from pipnet.pipnet import PIPNet, get_network, PIPNetBYOL
from util.log import Log
import torch.nn as nn
from util.args import get_args, save_args, get_optimizer_nn
from util.data import get_dataloaders, SubsetSequentialSampler, unshuffle_dataloader, create_filtered_dataloader
from util.func import init_weights_xavier
from pipnet.train import train_pipnet, test_pipnet
# from pipnet.test import eval_pipnet, get_thresholds, eval_ood
from util.eval_cub_csv import eval_prototypes_cub_parts_csv, get_topk_cub, get_proto_patches_cub
import torch
from util.vis_pipnet import visualize, visualize_topk
from util.visualize_prediction import vis_pred, vis_pred_experiments
import sys, os
import random
import numpy as np
from shutil import copy
import matplotlib.pyplot as plt
from copy import deepcopy

from omegaconf import OmegaConf
from util.node import Node
import shutil
from util.phylo_utils import construct_phylo_tree, construct_discretized_phylo_tree
from util.custom_losses import WeightedCrossEntropyLoss, WeightedNLLLoss, FocalLossWrapper
from util.vis_hpipnet import save_images_topk
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data import DataLoader, SubsetRandomSampler

import time
import wandb
from collections import Counter

# # Set CUDA_LAUNCH_BLOCKING=1
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def copy_files(src_dir, dest_dir, extensions, skip_folders=None):
	"""
	Copies all .py files from src_dir to dest_dir while preserving directory structure.
	"""
	for root, dirs, files in os.walk(src_dir):
		skip_this_folder = False
		for skip_folder in skip_folders:
			if os.path.commonprefix([root, os.path.join(src_dir, skip_folder)]) == os.path.join(src_dir, skip_folder):
				skip_this_folder = True
		if skip_this_folder:
			continue
		for file in files:
			if file.split('.')[-1] in extensions:
				src_file_path = os.path.join(root, file)
				dest_file_path = os.path.join(dest_dir, os.path.relpath(src_file_path, src_dir))
				if os.path.commonprefix([os.path.abspath(src_file_path), os.path.abspath(dest_dir)]) != os.path.abspath(dest_dir):
					dest_file_dir = os.path.dirname(dest_file_path)
					if not os.path.exists(dest_file_dir):
						os.makedirs(dest_file_dir)
					shutil.copy(src_file_path, dest_file_path)

def run_pipnet(args=None):
    time_ = time.time()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    print('manual_seed', (time.time()-time_)/60)

    args = args or get_args()
    assert args.batch_size > 1

    if args.training_wheels == 'y':
        # args.wandb = 'n'
        # args.copy_files = 'n'
        pass

    if args.wandb == 'n':
        os.environ['WANDB_DISABLED'] = 'true'
        print('Disabled wand')

    args.protopool = "n"

    if (args.align_pf == 'y') and not ('y' in args.softmax or args.gumbel_softmax == 'y'):
        raise Exception('Use align_pf loss only when softmax or gumbel softmax is turned on')

    if (args.minmaximize == 'y') and (args.protopool == 'y'):
        raise Exception('Only use minmaximize loss when args.protopool == "n"')
    
    if ('y' in args.tanh_desc) and (args.protopool == 'y'):
        print('protopool', args.protopool)
        print('tanh_desc', args.tanh_desc)
        print('state_dict_dir_fullmodel', args.state_dict_dir_fullmodel)
        raise Exception('Only use tanh_desc loss when args.protopool == "n"')
 
    # Create a logger
    log = Log(args.log_dir)
    print("Log dir: ", args.log_dir, flush=True)
    # Log the run arguments

    time_ = time.time()
    save_args(args, log.metadata_dir)
    print('save_args', (time.time()-time_)/60)

    if args.copy_files == 'y':
        time_ = time.time()
        copy_files(src_dir=os.getcwd(), dest_dir=os.path.join(args.log_dir, 'source_clone'), \
                    extensions=['py', 'yaml', '.ipynb', '.sh'], skip_folders=['runs', 'wandb', 'SLURM'])
        print('copy_files', (time.time()-time_)/60)
    else:
        print('Disabled copy_files')
    
    # os.environ['WANDB_MODE'] = 'offline'
    # os.environ['WANDB_DIR'] = args.log_dir
    
    time_ = time.time()
    wandb_run = wandb.init(project="pipnet", name=os.path.basename(args.log_dir), config=vars(args), reinit=False)
    print('wandb_run', (time.time()-time_)/60)

    if args.phylo_config:
        phylo_config = OmegaConf.load(args.phylo_config)

    if args.phylo_config:
		# construct the phylo tree
        if phylo_config.phyloDistances_string == 'None':
            root = construct_phylo_tree(phylo_config.phylogeny_path)
            print('-'*25 + ' No discretization ' + '-'*25)
        else:
            root = construct_discretized_phylo_tree(phylo_config.phylogeny_path, phylo_config.phyloDistances_string)
            print('-'*25 + ' Discretized ' + '-'*25)
    else:
        # construct the tree (original hierarchy as described in the paper)
        root = Node("root")
        root.add_children(['animal','vehicle','everyday_object','weapon','scuba_diver'])
        root.add_children_to('animal',['non_primate','primate'])
        root.add_children_to('non_primate',['African_elephant','giant_panda','lion'])
        root.add_children_to('primate',['capuchin','gibbon','orangutan'])
        root.add_children_to('vehicle',['ambulance','pickup','sports_car'])
        root.add_children_to('everyday_object',['laptop','sandal','wine_bottle'])
        root.add_children_to('weapon',['assault_rifle','rifle'])
        # flat root
        # root.add_children(['scuba_diver','African_elephant','giant_panda','lion','capuchin','gibbon','orangutan','ambulance','pickup','sports_car','laptop','sandal','wine_bottle','assault_rifle','rifle'])
    root.assign_all_descendents()

    if ('y' in args.OOD_ent) and (not args.bias):
        raise Exception('Recommended to use with bias in classification layer when using OOD_ent')

    # set pretrain epochs zero if align and uni are not used
    if (args.align == 'n') and (args.uni == 'n') and (args.align_pf == 'n') and (args.tanh == 'n') and (args.byol.split('|')[0] == 'n') and (args.epochs_pretrain > 0):
        raise Exception('Do not pretrain if not using any pretrain specific losses like align, uni, byol etc.')

    # update num of protos per node based on num_protos_per_descendant
    if args.num_features == 0 and args.num_protos_per_descendant == 0 and args.num_protos_per_child == 0:
        raise Exception('Either of num_features or num_protos_per_descendant or num_protos_per_child must be greater than zero')
    for node in root.nodes_with_children():
        node.set_num_protos(num_protos_per_descendant=args.num_protos_per_descendant,\
                            num_protos_per_child=args.num_protos_per_child,\
                            min_protos=args.num_features,\
                            split_protos=('protopool' in args) and (args.protopool == 'n'))


    gpu_list = args.gpu_ids.split(',')
    device_ids = []
    if args.gpu_ids!='':
        for m in range(len(gpu_list)):
            device_ids.append(int(gpu_list[m]))
    
    global device
    if not args.disable_cuda and torch.cuda.is_available():
        if len(device_ids)==1:
            device = torch.device('cuda:{}'.format(args.gpu_ids))
        elif len(device_ids)==0:
            device = torch.device('cuda')
            print("CUDA device set without id specification", flush=True)
            device_ids.append(torch.cuda.current_device())
        else:
            print("This code should work with multiple GPU's but we didn't test that, so we recommend to use only 1 GPU.", flush=True)
            device_str = ''
            for d in device_ids:
                device_str+=str(d)
                device_str+=","
            device = torch.device('cuda:'+str(device_ids[0]))
    else:
        device = torch.device('cpu')
     
    # Log which device was actually used
    print("Device used: ", device, "with id", device_ids, flush=True)
    
    # Obtain the dataset and dataloaders
    trainloader, trainloader_pretraining, trainloader_normal, trainloader_normal_augment, projectloader, testloader, test_projectloader, classes = get_dataloaders(args, device, OOD=False)
    if args.OOD_dataset:
        trainloader_OOD, trainloader_pretraining_OOD, trainloader_normal_OOD, trainloader_normal_augment_OOD, projectloader_OOD, testloader_OOD, test_projectloader_OOD, _ = get_dataloaders(args, device, OOD=True)
        print('-'*25 + 'Using OOD data' + '-'*25)
    else:
        trainloader_OOD = trainloader_pretraining_OOD = trainloader_normal_OOD = trainloader_normal_augment_OOD = projectloader_OOD = testloader_OOD = test_projectloader_OOD = None
        print('-'*25 + 'Not using OOD data' + '-'*25)
    if args.validation_size == 0.:
        print("Classes: ", testloader.dataset.class_to_idx, flush=True)
    else:
        print("Classes: ", str(classes), flush=True)

    if ('leave_out_classes' in args) and (args.leave_out_classes != ''):
        with open(args.leave_out_classes, 'r') as file:
            leave_out_classes = [line.strip() for line in file]

        leave_out_loader = testloader
        classes_to_keep = leave_out_classes
        idx_of_classes_to_keep = set()
        name2label = leave_out_loader.dataset.class_to_idx # param
        label2name = {label:name for name, label in name2label.items()}
        for label in label2name:
            # NOTE: Keeping the left out classes here
            if label2name[label] in classes_to_keep:
                idx_of_classes_to_keep.add(label)

        target_indices = []
        for i in range(len(leave_out_loader.dataset)):
            *_, label = leave_out_loader.dataset[i]
            if label in idx_of_classes_to_keep:
                target_indices.append(i)
        sampler = SubsetRandomSampler(target_indices)
        to_shuffle = False

        leave_out_loader = create_filtered_dataloader(leave_out_loader, sampler)

        # # Verifying if the classes are left out 
        # unique_labels = set()
        # for xs1, xs2, ys in trainloader:
        #     unique_labels.update(ys.tolist())
        # print("trainloader Unique Labels:", unique_labels)

        # unique_labels = set()
        # for xs, ys in projectloader:
        #     unique_labels.update(ys.tolist())
        # print("projectloader Unique Labels:", unique_labels)

    print("Node count:", len(root.nodes_with_children()))

    # NOTE: Commented because previously weightage was based on num samples
    # Now it is based on num of descendants
    # # Obtain the number of images per class - 
    # temp_dataset = trainloader.dataset.dataset.dataset
    # idx_to_class = {v: k for k, v in temp_dataset.class_to_idx.items()}
    # class_counts = Counter({class_name: 0 for class_name in temp_dataset.classes})
    # for *_, targets in trainloader:
    #     targets = targets.numpy() if not isinstance(targets, list) else targets
    #     class_names = [idx_to_class[idx] for idx in targets]
    #     class_counts.update(class_names)

    # Set loss weightage for each node if args.weighted_ce_loss == 'y'
    if ('weighted_ce_loss' in args) and (args.weighted_ce_loss == 'y'):
        for node in root.nodes_with_children():
            # node.set_loss_weightage(class_size_count=class_counts)
            node.set_loss_weightage_using_descendants_count()

    # for node in root.nodes_with_children():
    #     print(node.name, node.num_images_of_each_child)
    
    # Create a convolutional network based on arguments and add 1x1 conv layer
    feature_net, add_on_layers, pool_layer, classification_layers, num_prototypes = get_network(len(classes), args, root)

    # Create a PIP-Net
    if args.byol.split('|')[0] == 'y':
        net = PIPNetBYOL(num_classes=len(classes),
                            num_prototypes=num_prototypes,
                            feature_net = feature_net,
                            args = args,
                            add_on_layers = add_on_layers,
                            pool_layer = pool_layer,
                            classification_layers = classification_layers,
                            num_parent_nodes = len(root.nodes_with_children()),
                            root = root
                            )
    else:
        net = PIPNet(num_classes=len(classes),
                        num_prototypes=num_prototypes,
                        feature_net = feature_net,
                        args = args,
                        add_on_layers = add_on_layers,
                        pool_layer = pool_layer,
                        classification_layers = classification_layers,
                        num_parent_nodes = len(root.nodes_with_children()),
                        root = root
                        )
    net = net.to(device=device)
    net = nn.DataParallel(net, device_ids = device_ids)    

    # print(net)

    optimizer_net, optimizer_classifier, params_to_freeze, params_to_train, params_backbone = get_optimizer_nn(net, args)   

    # Initialize or load model
    with torch.no_grad():
        if args.state_dict_dir_net != '':
            raise Exception('Do not use this, use state_dict_dir_backbone for loading pretrained ._net')
            checkpoint = torch.load(args.state_dict_dir_net,map_location=device)
            net.load_state_dict(checkpoint['model_state_dict'],strict=True) 
            print("Pretrained network loaded", flush=True)
            net.module._multiplier.requires_grad = False
            try:
                optimizer_net.load_state_dict(checkpoint['optimizer_net_state_dict']) 
            except:
                print('-'*25, 'Unable to load optimizer_net_state_dict')
                pass

            loading_pretrained_only_model = False
            for attr in dir(net.module):
                if attr.endswith('_classification'):
                    # assume that the linear classification layer is not yet trained (e.g. when loading a pretrained backbone only)
                    if torch.mean(getattr(net.module, attr).weight).item() > 1.0 \
                        and torch.mean(getattr(net.module, attr).weight).item() < 3.0 \
                            and torch.count_nonzero(torch.relu(getattr(net.module, attr).weight-1e-5)).float().item() > 0.8*(getattr(net.module, attr).weight.shape[0] * getattr(net.module, attr).weight.shape[1]): 
                        print(f"We assume that the {attr} layer is not yet trained. We re-initialize it...", flush=True)
                        torch.nn.init.normal_(getattr(net.module, attr).weight, mean=1.0,std=0.1) 
                        torch.nn.init.constant_(net.module._multiplier, val=2.)
                        print(f"{attr} layer initialized with mean", torch.mean(getattr(net.module, attr).weight).item(), flush=True)
                        if args.bias:
                            torch.nn.init.constant_(getattr(net.module, attr).bias, val=0.)
                        loading_pretrained_only_model = True
            if loading_pretrained_only_model and 'optimizer_classifier_state_dict' in checkpoint.keys():
                optimizer_classifier.load_state_dict(checkpoint['optimizer_classifier_state_dict'])

        elif args.state_dict_dir_backbone != '':
            checkpoint = torch.load(args.state_dict_dir_backbone,map_location=device)
            # load backbone 'module._net' from checkpoint
            filtered_checkpoint_dict = {key:val for key, val in checkpoint['model_state_dict'].items() if (key.startswith('module._net') or key.startswith('module._add_on'))}
            if ('byol' in args) and args.byol == 'y':
                filtered_checkpoint_dict.update({key:val for key, val in checkpoint['model_state_dict'].items() if key.startswith('module._projector')})
                filtered_checkpoint_dict.update({key:val for key, val in checkpoint['model_state_dict'].items() if key.startswith('module._predictor')})
                filtered_checkpoint_dict.update({key:val for key, val in checkpoint['model_state_dict'].items() if key.startswith('module._target_feature_net')})
                filtered_checkpoint_dict.update({key:val for key, val in checkpoint['model_state_dict'].items() if key.startswith('module._target_projector')})
            net.load_state_dict(filtered_checkpoint_dict,strict=False) 
            print(f"Backbone and add-on loaded from {args.state_dict_dir_backbone}", flush=True)

            # NOTE: This makes sense only for hypersphere style contrastive learning
            # # initialize add on
            # # net.module._add_on.apply(init_weights_xavier)
            # for attr in dir(net.module):
            #     if attr.endswith('_add_on'):
            #         getattr(net.module, attr).apply(init_weights_xavier)

            # # initialize classification
            # for attr in dir(net.module):
            #     if attr.endswith('_classification'):
            #         torch.nn.init.normal_(getattr(net.module, attr).weight, mean=1.0,std=0.1) 
            #         if args.bias:
            #             torch.nn.init.constant_(getattr(net.module, attr).bias, val=0.)
            #         print(f"{attr} layer initialized with mean", torch.mean(getattr(net.module, attr).weight).item(), flush=True)
            
            # initialize multiplier
            torch.nn.init.constant_(net.module._multiplier, val=2.)
            net.module._multiplier.requires_grad = False

        elif args.state_dict_dir_fullmodel != '':
            checkpoint = torch.load(args.state_dict_dir_fullmodel,map_location=str(device))
            net.load_state_dict(checkpoint['model_state_dict'], strict=True)
            try:
                optimizer_net.load_state_dict(checkpoint['optimizer_net_state_dict'])
            except:
                print('-'*25, 'Unable to load optimizer_net_state_dict')
                pass
            
            try:
                optimizer_classifier.load_state_dict(checkpoint['optimizer_classifier_state_dict'])
            except:
                print('-'*25, 'Unable to load optimizer_classifier_state_dict')
                pass

            print("/*"*20, f"Full model loaded from {args.state_dict_dir_fullmodel}", flush=True)

            # initialize multiplier
            torch.nn.init.constant_(net.module._multiplier, val=2.)
            net.module._multiplier.requires_grad = False

        else:
            # initialize add on
            # net.module._add_on.apply(init_weights_xavier)
            for attr in dir(net.module):
                if attr.endswith('_add_on'):
                    getattr(net.module, attr).apply(init_weights_xavier)

            # # initialize classification
            # for attr in dir(net.module):
            #     if attr.endswith('_classification'):
            #         torch.nn.init.normal_(getattr(net.module, attr).weight, mean=1.0,std=0.1) 
            #         if args.bias:
            #             torch.nn.init.constant_(getattr(net.module, attr).bias, val=0.)
            #         print(f"{attr} layer initialized with mean", torch.mean(getattr(net.module, attr).weight).item(), flush=True)

            # initialize multiplier
            torch.nn.init.constant_(net.module._multiplier, val=2.)
            net.module._multiplier.requires_grad = False


    # Define classification loss function and scheduler
    # if args.weighted_ce_loss == 'n' input weights during forward call will be none and the output will be unweighted mean
    if ('focal_loss' in args) and (args.focal_loss == 'y'):
        criterion = FocalLossWrapper(device=device, alpha=1, gamma=args.focal_loss_gamma, reduction='mean').to(device)
    else:
        criterion = WeightedNLLLoss(device=device).to(device)

    scheduler_net = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_net, T_max=len(trainloader_pretraining)*args.epochs_pretrain, eta_min=args.lr_block/100., last_epoch=-1)

    # Forward one batch through the backbone to get the latent output size
    with torch.no_grad():
        xs1, _, _ = next(iter(trainloader))
        xs1 = xs1.to(device)
        features, proto_features, _, _ = net(xs1, inference=True)
        wshape = proto_features['root'].shape[-1]
        args.wshape = wshape #needed for calculating image patch size
        print("Output shape: ", proto_features['root'].shape, flush=True)
    
    if net.module._num_classes == 2:
        # Create a csv log for storing the test accuracy, F1-score, mean train accuracy and mean loss for each epoch
        log.create_log('log_epoch_overview', 'epoch', 'test_top1_acc', 'test_f1', 'almost_sim_nonzeros', 'local_size_all_classes','almost_nonzeros_pooled', 'num_nonzero_prototypes', 'mean_train_acc', 'mean_train_loss_during_epoch')
        print("Your dataset only has two classes. Is the number of samples per class similar? If the data is imbalanced, we recommend to use the --weighted_loss flag to account for the imbalance.", flush=True)
    else:
        # Create a csv log for storing the test accuracy (top 1 and top 5), mean train accuracy and mean loss for each epoch
        log.create_log('log_epoch_overview', 'epoch', 'test_top1_acc', 'test_top5_acc', 'almost_sim_nonzeros', 'local_size_all_classes','almost_nonzeros_pooled', 'num_nonzero_prototypes', 'mean_train_acc', 'mean_train_loss_during_epoch')
    
    if args.byol.split('|')[0] == 'y':
        if len(args.byol.split('|')) > 1:
            byol_tau_base = float(args.byol.split('|')[1])
        else:
            byol_tau_base = 0.9995
        if len(args.byol.split('|')) > 2:
            byol_tau_max = float(args.byol.split('|')[2])
        else:
            byol_tau_max = 1.0
    
    lrs_pretrain_net = []
    # ------------------------- PRETRAINING PROTOTYPES PHASE -------------------------
    max_pretraining_steps = len(trainloader_pretraining) * args.epochs_pretrain
    step_info_pretraining = {'current_step': 0, 'max_training_steps': max_pretraining_steps}
    for epoch in range(1, args.epochs_pretrain+1):
        for param in params_to_train:
            param.requires_grad = True
        for attr in dir(net.module):
            if attr.endswith('_add_on'):
                for param in getattr(net.module, attr).parameters():
                    param.requires_grad = True
        for attr in dir(net.module):
            if attr.endswith('_classification'):
                for param in getattr(net.module, attr).parameters():
                    param.requires_grad = False
        for param in params_to_freeze:
            param.requires_grad = True # can be set to False when you want to freeze more layers
        for param in params_backbone:
            param.requires_grad = False #can be set to True when you want to train whole backbone (e.g. if dataset is very different from ImageNet)
        
        # # IMPORTANT: Remove this later added for freezing backbone entirely during training
        # print('IMPORTANT: Fully frozen backbone')
        # for attr in dir(net.module):
        #     if attr.endswith('_add_on'):
        #         for param in getattr(net.module, attr).parameters():
        #             param.requires_grad = True
        # for attr in dir(net.module):
        #     if attr.endswith('_classification'):
        #         for param in getattr(net.module, attr).parameters():
        #             param.requires_grad = False
        # for param in params_to_train:
        #     param.requires_grad = False
        # for param in params_to_freeze:
        #     param.requires_grad = False # can be set to False when you want to freeze more layers
        # for param in params_backbone:
        #     param.requires_grad = False #can be set to True when you want to train whole backbone (e.g. if dataset is very different from ImageNet)

        print("\nPretrain Epoch", epoch, "with batch size", trainloader_pretraining.batch_size, flush=True)

        # Pretrain prototypes
        if args.byol.split('|')[0]== 'y':
            train_info, log_dict = train_pipnet(net, trainloader_pretraining, optimizer_net, optimizer_classifier, \
                                    scheduler_net, None, criterion, epoch, args.epochs_pretrain, device, \
                                    pretrain=True, finetune=False, kernel_orth=args.kernel_orth == 'y', \
                                    tanh_desc= ('y' in args.tanh_desc), align=args.align == 'y', uni=args.uni == 'y', align_pf=args.align_pf == 'y', tanh=args.tanh == 'y',\
                                    minmaximize=args.minmaximize == 'y', cluster_desc=args.cluster_desc == 'y', sep_desc=args.sep_desc == 'y', subspace_sep=args.subspace_sep == 'y', \
                                    byol=True, byol_tau_base=byol_tau_base, byol_tau_max=byol_tau_max, step_info=step_info_pretraining, \
                                    wandb_run=wandb_run, log=log, args=args)
        else:
            train_info, log_dict = train_pipnet(net, trainloader_pretraining, optimizer_net, optimizer_classifier, \
                                                scheduler_net, None, criterion, epoch, args.epochs_pretrain, device, \
                                                pretrain=True, finetune=False, kernel_orth=args.kernel_orth == 'y', \
                                                tanh_desc=('y' in args.tanh_desc), align=args.align == 'y', uni=args.uni == 'y', align_pf=args.align_pf == 'y', tanh=args.tanh == 'y',\
                                                minmaximize=args.minmaximize == 'y', cluster_desc=args.cluster_desc == 'y', sep_desc=args.sep_desc == 'y', subspace_sep=args.subspace_sep == 'y',\
                                                 wandb_run=wandb_run, log=log, args=args)
        # wandb_run.log(log_dict, step=epoch)
        # test_info = test_pipnet(net, trainloader_pretraining, optimizer_net, optimizer_classifier, scheduler_net, None, criterion, epoch, args.epochs_pretrain, device, pretrain=True, finetune=False)
        lrs_pretrain_net+=train_info['lrs_net']
        plt.clf()
        plt.plot(lrs_pretrain_net)
        plt.savefig(os.path.join(args.log_dir,'lr_pretrain_net.png'))
        log.log_values('log_epoch_overview', epoch, "n.a.", "n.a.", "n.a.", "n.a.", "n.a.", "n.a.", "n.a.", train_info['loss'])
    
    if args.state_dict_dir_net == '':
        net.eval()
        torch.save({'model_state_dict': net.state_dict(), 'optimizer_net_state_dict': optimizer_net.state_dict()}, os.path.join(os.path.join(args.log_dir, 'checkpoints'), 'net_pretrained'))
        net.train()
    # with torch.no_grad():
    #     if 'convnext' in args.net and args.epochs_pretrain > 0:
    #         for node in root.nodes_with_children():
    #             topks = visualize_topk(net, projectloader, node.num_children(), device, f'visualised_pretrained_prototypes_topk/{node.name}', args, node=node)
    
    # ------------------------- SECOND TRAINING PHASE -------------------------
    # re-initialize optimizers and schedulers for second training phase
    optimizer_net, optimizer_classifier, params_to_freeze, params_to_train, params_backbone = get_optimizer_nn(net, args)            
    scheduler_net = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_net, T_max=len(trainloader)*args.epochs, eta_min=args.lr_net/100.)
    # scheduler for the classification layer is with restarts, such that the model can re-active zeroed-out prototypes. Hence an intuitive choice. 
    if args.epochs<=30:
        scheduler_classifier = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_classifier, T_0=5, eta_min=0.001, T_mult=1, verbose=False)
    else:
        scheduler_classifier = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_classifier, T_0=10, eta_min=0.001, T_mult=1, verbose=False)
    # for param in net.module.parameters():
    #     param.requires_grad = False
    # for attr in dir(net.module):
    #     if attr.endswith('_classification'):
    #         for param in getattr(net.module, attr).parameters():
    #             param.requires_grad = True
    
    frozen = True
    lrs_net = []
    lrs_classifier = []
    
    max_training_steps = len(trainloader) * args.epochs
    step_info_training = {'current_step': 0, 'max_training_steps': max_training_steps}
    for epoch in range(1, args.epochs + 1):                      
        epochs_to_finetune = args.epochs_finetune #3 #during finetuning, only train classification layer and freeze rest. usually done for a few epochs (at least 1, more depends on size of dataset)
        if epoch <= args.epochs_finetune_classifier:
            for attr in dir(net.module):
                if attr.endswith('_classification'):
                    for param in getattr(net.module, attr).parameters():
                        param.requires_grad = True
            for attr in dir(net.module):
                if attr.endswith('_add_on'):
                    for param in getattr(net.module, attr).parameters():
                        param.requires_grad = False # True # False
            for param in params_to_train:
                param.requires_grad = False
            for param in params_to_freeze:
                param.requires_grad = False
            for param in params_backbone:
                param.requires_grad = False
            finetune = True
        # else:
        #     # IMPORTANT: Remove this later, added for freezing backbone entirely during training
        #     finetune = False
        #     print('IMPORTANT: Fully frozen backbone')
        #     for attr in dir(net.module):
        #         if attr.endswith('_add_on'):
        #             for param in getattr(net.module, attr).parameters():
        #                 param.requires_grad = True
        #     for attr in dir(net.module):
        #         if attr.endswith('_classification'):
        #             for param in getattr(net.module, attr).parameters():
        #                 param.requires_grad = True
        #     for param in params_to_train:
        #         param.requires_grad = False
        #     for param in params_to_freeze:
        #         param.requires_grad = False # can be set to False when you want to freeze more layers
        #     for param in params_backbone:
        #         param.requires_grad = False #can be set to True when you want to train whole backbone (e.g. if dataset is very different from ImageNet)

        elif epoch <= epochs_to_finetune: # and (args.epochs_pretrain > 0 or args.state_dict_dir_net != ''):
            # for param in net.module._add_on.parameters():
            #     param.requires_grad = False
            for attr in dir(net.module):
                if attr.endswith('_classification'):
                    for param in getattr(net.module, attr).parameters():
                        param.requires_grad = True
            for attr in dir(net.module):
                if attr.endswith('_add_on'):
                    for param in getattr(net.module, attr).parameters():
                        param.requires_grad = True # False
            for param in params_to_train:
                param.requires_grad = False
            for param in params_to_freeze:
                param.requires_grad = False
            for param in params_backbone:
                param.requires_grad = False
            finetune = True
        elif epoch > args.epochs_finetune_mask_prune:
            for attr in dir(net.module):
                if attr.endswith('_classification'):
                    for param in getattr(net.module, attr).parameters():
                        param.requires_grad = False
            for attr in dir(net.module):
                if attr.endswith('_add_on'):
                    for param in getattr(net.module, attr).parameters():
                        param.requires_grad = False # False
            for param in params_to_train:
                param.requires_grad = False
            for param in params_to_freeze:
                param.requires_grad = False
            for param in params_backbone:
                param.requires_grad = False
            for attr in dir(net.module):
                if attr.endswith('_proto_presence'):
                    param = getattr(net.module, attr)
                    param.requires_grad = True
        else:
            finetune=False          
            if frozen:
                # unfreeze backbone
                if epoch>(args.freeze_epochs):
                    # for param in net.module._add_on.parameters():
                    #     param.requires_grad = True
                    for attr in dir(net.module):
                        if attr.endswith('_add_on'):
                            for param in getattr(net.module, attr).parameters():
                                param.requires_grad = True
                    for param in params_to_freeze:
                        param.requires_grad = True
                    for param in params_to_train:
                        param.requires_grad = True
                    for param in params_backbone:
                        param.requires_grad = True   
                    frozen = False
                # freeze first layers of backbone, train rest
                else:
                    for param in params_to_freeze:
                        param.requires_grad = True #Can be set to False if you want to train fewer layers of backbone
                    # for param in net.module._add_on.parameters():
                    #     param.requires_grad = True
                    for attr in dir(net.module):
                        if attr.endswith('_add_on'):
                            for param in getattr(net.module, attr).parameters():
                                param.requires_grad = True
                    for param in params_to_train:
                        param.requires_grad = True
                    for param in params_backbone:
                        param.requires_grad = False
        
        print("\n Epoch", epoch, "frozen:", frozen, flush=True)            
        if (epoch==args.epochs or epoch%30==0) and args.epochs>1:
            # # SET SMALL WEIGHTS TO ZERO
            # with torch.no_grad():
            #     torch.set_printoptions(profile="full")
            #     for attr in dir(net.module):
            #         if attr.endswith('_classification'):
            #             getattr(net.module, attr).weight.copy_(torch.clamp(getattr(net.module, attr).weight.data - 0.001, min=0.)) 
            #             print(f"{attr} weights: ", getattr(net.module, attr).weight[getattr(net.module, attr).weight.nonzero(as_tuple=True)], \
            #                   (getattr(net.module, attr).weight[getattr(net.module, attr).weight.nonzero(as_tuple=True)]).shape, flush=True)
            #             if args.bias:
            #                 print(f"{attr} bias: ", getattr(net.module, attr).bias, flush=True)
            #     torch.set_printoptions(profile="default")

            for node in root.nodes_with_children():
                classification_weights = getattr(net.module, '_'+node.name+'_classification').weight
                node_label_to_children = {label: name for name, label in node.children_to_labels.items()}
                for class_label in range(classification_weights.shape[0]):
                    class_name = node_label_to_children[class_label]
                    print(f'Num protos for {node.name} class', class_name, torch.nonzero(classification_weights[class_label, :] > 1e-3).shape[0])

        if args.byol.split('|')[0]== 'y':
            train_info, log_dict = train_pipnet(net, trainloader, optimizer_net, optimizer_classifier, \
                                    scheduler_net, scheduler_classifier, criterion, epoch, \
                                        args.epochs, device, pretrain=False, finetune=finetune, \
                                        train_loader_OOD=trainloader_OOD, kernel_orth=args.kernel_orth == 'y',\
                                            tanh_desc=('y' in args.tanh_desc), align=args.align == 'y', uni=args.uni == 'y', align_pf=args.align_pf == 'y', tanh=args.tanh == 'y',\
                                            minmaximize=args.minmaximize == 'y', cluster_desc=args.cluster_desc == 'y', sep_desc=args.sep_desc == 'y', subspace_sep=args.subspace_sep == 'y', \
                                            byol=True, byol_tau_base=byol_tau_base, byol_tau_max=byol_tau_max, step_info=step_info_training, \
                                                wandb_run=wandb_run, pretrain_epochs=args.epochs_pretrain, log=log, args=args)
        else:
            train_info, log_dict = train_pipnet(net, trainloader, optimizer_net, optimizer_classifier, \
                                    scheduler_net, scheduler_classifier, criterion, epoch, \
                                        args.epochs, device, pretrain=False, finetune=finetune, \
                                        train_loader_OOD=trainloader_OOD, kernel_orth=args.kernel_orth == 'y',\
                                            tanh_desc=('y' in args.tanh_desc), align=args.align == 'y', uni=args.uni == 'y', align_pf=args.align_pf == 'y', tanh=args.tanh == 'y',\
                                            minmaximize=args.minmaximize == 'y', cluster_desc=args.cluster_desc == 'y', sep_desc=args.sep_desc == 'y', subspace_sep=args.subspace_sep == 'y', \
                                            wandb_run=wandb_run, pretrain_epochs=args.epochs_pretrain, log=log, args=args)
        # wandb_run.log(log_dict, step=epoch + args.epochs_pretrain)
        
        if (epoch==args.epochs or epoch%5==0) and args.epochs>1:
            if args.byol.split('|')[0] == 'y':
                test_info, log_dict = test_pipnet(net, testloader, optimizer_net, optimizer_classifier, \
                                        scheduler_net, scheduler_classifier, criterion, epoch, \
                                            args.epochs, device, pretrain=False, finetune=finetune, \
                                            test_loader_OOD=testloader_OOD, kernel_orth=args.kernel_orth == 'y', \
                                                tanh_desc=('y' in args.tanh_desc), align=args.align == 'y', uni=args.uni == 'y', align_pf=args.align_pf == 'y', tanh=args.tanh == 'y',\
                                                minmaximize=args.minmaximize == 'y', cluster_desc=args.cluster_desc == 'y', sep_desc=args.sep_desc == 'y', byol=True, byol_tau_base=0.9995, step_info=step_info_training, \
                                                wandb_run=wandb_run, pretrain_epochs=args.epochs_pretrain, log=log, args=args)
            else:
                test_info, log_dict = test_pipnet(net, testloader, optimizer_net, optimizer_classifier, \
                                        scheduler_net, scheduler_classifier, criterion, epoch, \
                                            args.epochs, device, pretrain=False, finetune=finetune, \
                                            test_loader_OOD=testloader_OOD, kernel_orth=args.kernel_orth == 'y', \
                                                tanh_desc=('y' in args.tanh_desc), cluster_desc=args.cluster_desc == 'y', sep_desc=args.sep_desc == 'y', align=args.align == 'y', uni=args.uni == 'y', \
                                                align_pf=args.align_pf == 'y', tanh=args.tanh == 'y',\
                                                minmaximize=args.minmaximize == 'y', wandb_run=wandb_run, pretrain_epochs=args.epochs_pretrain, log=log, args=args)
                if args.leave_out_classes.strip() != '':
                    test_info, log_dict = test_pipnet(net, leave_out_loader, optimizer_net, optimizer_classifier, \
                                            scheduler_net, scheduler_classifier, criterion, epoch, \
                                                args.epochs, device, pretrain=False, finetune=finetune, \
                                                test_loader_OOD=testloader_OOD, kernel_orth=args.kernel_orth == 'y', \
                                                    tanh_desc=('y' in args.tanh_desc), cluster_desc=args.cluster_desc == 'y', sep_desc=args.sep_desc == 'y', align=args.align == 'y', uni=args.uni == 'y', \
                                                    align_pf=args.align_pf == 'y', tanh=args.tanh == 'y',\
                                                    minmaximize=args.minmaximize == 'y', wandb_run=wandb_run, pretrain_epochs=args.epochs_pretrain, \
                                                    log=log, args=args, leave_out_classes=leave_out_classes)

        # wandb_run.log(log_dict, step=epoch + args.epochs_pretrain)
        # test_info = test_pipnet(net, testloader, criterion, epoch, device, progress_prefix= 'Test Epoch', wandb_logging=True, wandb_log_subdir = 'test')
        lrs_net+=train_info['lrs_net']
        lrs_classifier+=train_info['lrs_class']
        # Evaluate model - not doing this for now requires modification in test.py
        # eval_info = eval_pipnet(net, testloader, epoch, device, log)
        # log.log_values('log_epoch_overview', epoch, eval_info['top1_accuracy'], eval_info['top5_accuracy'], eval_info['almost_sim_nonzeros'], eval_info['local_size_all_classes'], eval_info['almost_nonzeros'], eval_info['num non-zero prototypes'], train_info['train_accuracy'], train_info['loss'])
        
        with torch.no_grad():
            net.eval()
            torch.save({'model_state_dict': net.state_dict(), 'optimizer_net_state_dict': optimizer_net.state_dict(), 'optimizer_classifier_state_dict': optimizer_classifier.state_dict()}, os.path.join(os.path.join(args.log_dir, 'checkpoints'), 'net_trained'))

            if epoch%5 == 0:
                # visualize prototypes
                # for node in root.nodes_with_children():
                #     topks = visualize_topk(net, projectloader, node.num_children(), device, f'visualised_prototypes_topk_ep={epoch}/{node.name}', args, node=node)
                # save model
                net.eval()
                torch.save({'model_state_dict': net.state_dict(), 'optimizer_net_state_dict': optimizer_net.state_dict(), 'optimizer_classifier_state_dict': optimizer_classifier.state_dict()}, os.path.join(os.path.join(args.log_dir, 'checkpoints'), 'net_trained_%s'%str(epoch)))            
        
            # save learning rate in figure
            plt.clf()
            plt.plot(lrs_net)
            plt.savefig(os.path.join(args.log_dir,'lr_net.png'))
            plt.clf()
            plt.plot(lrs_classifier)
            plt.savefig(os.path.join(args.log_dir,'lr_class.png'))
                
    net.eval()
    torch.save({'model_state_dict': net.state_dict(), 'optimizer_net_state_dict': optimizer_net.state_dict(), 'optimizer_classifier_state_dict': optimizer_classifier.state_dict()}, os.path.join(os.path.join(args.log_dir, 'checkpoints'), 'net_trained_last'))

    # for node in root.nodes_with_children():
    #     topks = visualize_topk(net, projectloader, node.num_children(), device, f'visualised_prototypes_topk/{node.name}', args, node=node)
    #     # set weights of prototypes that are never really found in projection set to 0
    #     set_to_zero = []
    #     classification_layer = getattr(net.module, '_'+node.name+'_classification')
    #     if topks:
    #         for prot in topks.keys():
    #             found = False
    #             for (i_id, score) in topks[prot]:
    #                 if score > 0.1:
    #                     found = True
    #             if not found:
    #                 torch.nn.init.zeros_(classification_layer.weight[:,prot])
    #                 set_to_zero.append(prot)
    #         print(f"Weights of prototypes of node {node.name}", set_to_zero, "are set to zero because it is never detected with similarity>0.1 in the training set", flush=True)

    #     # Not doing this for now requires modification in test.py
    #     # eval_info = eval_pipnet(net, testloader, "notused"+str(args.epochs), device, log)
    #     # log.log_values('log_epoch_overview', "notused"+str(args.epochs), eval_info['top1_accuracy'], eval_info['top5_accuracy'], eval_info['almost_sim_nonzeros'], eval_info['local_size_all_classes'], eval_info['almost_nonzeros'], eval_info['num non-zero prototypes'], "n.a.", "n.a.")

    #     print(f"classifier weights {node.name}: ", classification_layer.weight, flush=True)
    #     print(f"Classifier weights nonzero {node.name}: ", classification_layer.weight[classification_layer.weight.nonzero(as_tuple=True)], (classification_layer.weight[classification_layer.weight.nonzero(as_tuple=True)]).shape, flush=True)
    #     print(f"Classifier bias {node.name}: ", classification_layer.bias, flush=True)

    # Print weights and relevant prototypes per class
    # for node in root.nodes_with_children():
    #     classification_layer = getattr(net.module, '_'+node.name+'_classification')
    #     coarse_label_to_name = {label:name for name, label in node.children_to_labels.items()}
    #     print(f"Node: {node.name}, Class -> Prototypes")
    #     for c in range(classification_layer.weight.shape[0]):
    #         relevant_ps = []
    #         proto_weights = classification_layer.weight[c,:]
    #         for p in range(classification_layer.weight.shape[1]):
    #             if proto_weights[p]> 1e-3:
    #                 relevant_ps.append((p, proto_weights[p].item()))
    #         if args.validation_size == 0.:
    #             print("Class", c, "(", coarse_label_to_name[c], "):","has", len(relevant_ps),"relevant prototypes: ", relevant_ps, flush=True)
    #             # print("Class", c, "(", list(testloader.dataset.class_to_idx.keys())[list(testloader.dataset.class_to_idx.values()).index(c)],"):","has", len(relevant_ps),"relevant prototypes: ", relevant_ps, flush=True)

    #     print(f"Node: {node.name}, Prototypes -> Class")
    #     for p in range(classification_layer.weight.shape[1]):
    #         relevant_classes = []
    #         proto_weights = classification_layer.weight[:,p]
    #         for c in range(classification_layer.weight.shape[0]):
    #             if proto_weights[c]> 1e-3:
    #                 relevant_classes.append((c, proto_weights[c].item()))
    #         if relevant_classes:
    #             print("Prototype", p, " present in", len(relevant_classes), "classes: ", [coarse_label_to_name[rc[0]] for rc in relevant_classes], flush=True)

    # Evaluate prototype purity        
    # if args.dataset == 'CUB-200-2011':
    #     projectset_img0_path = projectloader.dataset.samples[0][0]
    #     project_path = os.path.split(os.path.split(projectset_img0_path)[0])[0].split("dataset")[0]
    #     parts_loc_path = os.path.join(project_path, "parts/part_locs.txt")
    #     parts_name_path = os.path.join(project_path, "parts/parts.txt")
    #     imgs_id_path = os.path.join(project_path, "images.txt")
    #     cubthreshold = 0.5 

    #     net.eval()
    #     print("\n\nEvaluating cub prototypes for training set", flush=True)        
    #     csvfile_topk = get_topk_cub(net, projectloader, 10, 'train_'+str(epoch), device, args)
    #     eval_prototypes_cub_parts_csv(csvfile_topk, parts_loc_path, parts_name_path, imgs_id_path, 'train_topk_'+str(epoch), args, log)
        
    #     csvfile_all = get_proto_patches_cub(net, projectloader, 'train_all_'+str(epoch), device, args, threshold=cubthreshold)
    #     eval_prototypes_cub_parts_csv(csvfile_all, parts_loc_path, parts_name_path, imgs_id_path, 'train_all_thres'+str(cubthreshold)+'_'+str(epoch), args, log)
        
    #     print("\n\nEvaluating cub prototypes for test set", flush=True)
    #     csvfile_topk = get_topk_cub(net, test_projectloader, 10, 'test_'+str(epoch), device, args)
    #     eval_prototypes_cub_parts_csv(csvfile_topk, parts_loc_path, parts_name_path, imgs_id_path, 'test_topk_'+str(epoch), args, log)
    #     cubthreshold = 0.5
    #     csvfile_all = get_proto_patches_cub(net, test_projectloader, 'test_'+str(epoch), device, args, threshold=cubthreshold)
    #     eval_prototypes_cub_parts_csv(csvfile_all, parts_loc_path, parts_name_path, imgs_id_path, 'test_all_thres'+str(cubthreshold)+'_'+str(epoch), args, log)
        
    # visualize predictions - not doing this for now
    # visualize(net, projectloader, len(classes), device, 'visualised_prototypes', args)
    # testset_img0_path = test_projectloader.dataset.samples[0][0]
    # test_path = os.path.split(os.path.split(testset_img0_path)[0])[0]
    # vis_pred(net, test_path, classes, device, args) 
    # if args.extra_test_image_folder != '':
    #     if os.path.exists(args.extra_test_image_folder):   
    #         vis_pred_experiments(net, args.extra_test_image_folder, classes, device, args)


    # EVALUATE OOD DETECTION - not doing this for now
    # ood_datasets = ["CARS", "CUB-200-2011", "pets"]
    # for percent in [95.]:
    #     print("\nOOD Evaluation for epoch", epoch,"with percent of", percent, flush=True)
    #     _, _, _, class_thresholds = get_thresholds(net, testloader, epoch, device, percent, log)
    #     print("Thresholds:", class_thresholds, flush=True)
    #     # Evaluate with in-distribution data
    #     id_fraction = eval_ood(net, testloader, epoch, device, class_thresholds)
    #     print("ID class threshold ID fraction (TPR) with percent",percent,":", id_fraction, flush=True)
        
    #     # Evaluate with out-of-distribution data
    #     for ood_dataset in ood_datasets:
    #         if ood_dataset != args.dataset:
    #             print("\n OOD dataset: ", ood_dataset,flush=True)
    #             ood_args = deepcopy(args)
    #             ood_args.dataset = ood_dataset
    #             _, _, _, _, _,ood_testloader, _, _ = get_dataloaders(ood_args, device)
                
    #             id_fraction = eval_ood(net, ood_testloader, epoch, device, class_thresholds)
    #             print(args.dataset, "- OOD", ood_dataset, "class threshold ID fraction (FPR) with percent",percent,":", id_fraction, flush=True)                

    print("Done!", flush=True)

    args.batch_size = 1
    trainloader, trainloader_pretraining, trainloader_normal, trainloader_normal_augment, projectloader, testloader, test_projectloader, classes = get_dataloaders(args, device, OOD=False)

    if len(classes) <= 60:

        for loadername in args.viz_loader.split(','):

            if loadername == 'projectloader':
                foldername = f'descendent_specific_topk_heatmap_{loadername}_ep=last'
                save_images_topk(args, unshuffle_dataloader(projectloader), net, root, save_path=args.log_dir, \
                                    foldername=foldername, find_non_descendants=False, device=device)
                print("Done visualizing descendants! " + loadername, flush=True)
                save_images_topk(args, unshuffle_dataloader(projectloader), net, root, save_path=args.log_dir, \
                                foldername=foldername, find_non_descendants=True, device=device)
                print("Done visualizing non-descendants!" + loadername, flush=True)

            if loadername == 'testloader':
                foldername = f'descendent_specific_topk_heatmap_{loadername}_ep=last'
                save_images_topk(args, unshuffle_dataloader(testloader), net, root, save_path=args.log_dir, \
                                    foldername=foldername, find_non_descendants=False, device=device)
                print("Done visualizing descendants! " + loadername, flush=True)
                save_images_topk(args, unshuffle_dataloader(testloader), net, root, save_path=args.log_dir, \
                                foldername=foldername, find_non_descendants=True, device=device)
                print("Done visualizing non-descendants!" + loadername, flush=True)

            elif loadername == 'test_projectloader':
                foldername = f'descendent_specific_topk_heatmap_{loadername}_ep=last'
                save_images_topk(args, unshuffle_dataloader(test_projectloader), net, root, save_path=args.log_dir, \
                                    foldername=foldername, find_non_descendants=False, device=device)
                print("Done visualizing descendants! " + loadername, flush=True)
                save_images_topk(args, unshuffle_dataloader(test_projectloader), net, root, save_path=args.log_dir, \
                                foldername=foldername, find_non_descendants=True, device=device)
                print("Done visualizing non-descendants!" + loadername, flush=True)
    else:
        print('Skipping visualization as there are too many classes')


class Tee(object):
    def __init__(self, name, mode, outstream):
        self.file = open(name, mode)
        self.stdout = outstream
    def __del__(self):
        self.file.close()
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()

if __name__ == '__main__':
    time_ = time.time()
    args = get_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    print('manual_seed', (time.time()-time_)/60)

    time_ = time.time()
    print_dir = os.path.join(args.log_dir,'out.txt')
    tqdm_dir = os.path.join(args.log_dir,'tqdm.txt')
    if not os.path.isdir(args.log_dir):
        os.mkdir(args.log_dir)
    
    # sys.stdout.close()
    # sys.stderr.close()
    # sys.stdout = open(print_dir, 'a')
    # sys.stderr = open(tqdm_dir, 'a')

    sys.stdout = Tee(print_dir, 'a', sys.stdout)
    sys.stderr = Tee(tqdm_dir, 'a', sys.stderr)
    print('stderr', (time.time()-time_)/60)

    time_ = time.time()
    run_pipnet(args)
    print('Finished in', int(time.time()-time_)//60, 'minutes')
    
    # sys.stdout.close()
    # sys.stderr.close()
