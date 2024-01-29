import os
import argparse
import pickle
import numpy as np
import random
import torch
import torch.optim
from torchlars import LARS

"""
    Utility functions for handling parsed arguments

"""
def get_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser('Train a PIP-Net')
    parser.add_argument('--dataset',
                        type=str,
                        default='CUB-200-2011',
                        help='Data set on PIP-Net should be trained')
    parser.add_argument('--OOD_dataset',
                        type=str,
                        default=None,
                        help='Data set on PIP-Net should be trained')
    parser.add_argument('--validation_size',
                        type=float,
                        default=0.,
                        help='Split between training and validation set. Can be zero when there is a separate test or validation directory. Should be between 0 and 1. Used for partimagenet (e.g. 0.2)')
    parser.add_argument('--net',
                        type=str,
                        default='convnext_tiny_26',
                        help='Base network used as backbone of PIP-Net. Default is convnext_tiny_26 with adapted strides to output 26x26 latent representations. Other option is convnext_tiny_13 that outputs 13x13 (smaller and faster to train, less fine-grained). Pretrained network on iNaturalist is only available for resnet50_inat. Options are: resnet18, resnet34, resnet50, resnet50_inat, resnet101, resnet152, convnext_tiny_26 and convnext_tiny_13.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='Batch size when training the model using minibatch gradient descent. Batch size is multiplied with number of available GPUs')
    parser.add_argument('--batch_size_pretrain',
                        type=int,
                        default=128,
                        help='Batch size when pretraining the prototypes (first training stage)')
    parser.add_argument('--epochs',
                        type=int,
                        default=60,
                        help='The number of epochs PIP-Net should be trained (second training stage)')
    parser.add_argument('--epochs_pretrain',
                        type=int,
                        default = 10,
                        help='Number of epochs to pre-train the prototypes (first training stage). Recommended to train at least until the align loss < 1'
                        )
    parser.add_argument('--epochs_finetune',
                        type=int,
                        default=5,
                        help='The number of epochs PIP-Net should be finetuned (second training stage)')
    parser.add_argument('--optimizer',
                        type=str,
                        default='Adam',
                        help='The optimizer that should be used when training PIP-Net')
    parser.add_argument('--lr',
                        type=float,
                        default=0.05, 
                        help='The optimizer learning rate for training the weights from prototypes to classes')
    parser.add_argument('--lr_block',
                        type=float,
                        default=0.0005, 
                        help='The optimizer learning rate for training the last conv layers of the backbone')
    parser.add_argument('--lr_net',
                        type=float,
                        default=0.0005, 
                        help='The optimizer learning rate for the backbone. Usually similar as lr_block.') 
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.0,
                        help='Weight decay used in the optimizer')
    parser.add_argument('--disable_cuda',
                        action='store_true',
                        help='Flag that disables GPU usage if set')
    parser.add_argument('--log_dir',
                        type=str,
                        default='./runs/run_pipnet',
                        help='The directory in which train progress should be logged')
    parser.add_argument('--num_features',
                        type=int,
                        default = 0,
                        help='Number of prototypes. When zero (default) the number of prototypes is the number of output channels of backbone. If this value is set, then a 1x1 conv layer will be added. Recommended to keep 0, but can be increased when number of classes > num output channels in backbone.')
    parser.add_argument('--image_size',
                        type=int,
                        default=224,
                        help='Input images will be resized to --image_size x --image_size (square). Code only tested with 224x224, so no guarantees that it works for different sizes.')
    parser.add_argument('--state_dict_dir_net',
                        type=str,
                        default='',
                        help='The directory containing a state dict with a pretrained PIP-Net. E.g., ./runs/run_pipnet/checkpoints/net_pretrained')
    parser.add_argument("--state_dict_dir_backbone",
                        type=str,
                        default='', 
                        help='The directory containing a state dict with a pretrained PIP-Net. Only the backbone i.e. "_net" will be loaded')
    parser.add_argument('--freeze_epochs',
                        type=int,
                        default = 10,
                        help='Number of epochs where pretrained features_net will be frozen while training classification layer (and last layer(s) of backbone)'
                        )
    parser.add_argument('--dir_for_saving_images',
                        type=str,
                        default='visualization_results',
                        help='Directoy for saving the prototypes and explanations')
    parser.add_argument('--disable_pretrained',
                        action='store_true',
                        help='When set, the backbone network is initialized with random weights instead of being pretrained on another dataset).'
                        )
    parser.add_argument('--weighted_loss',
                        action='store_true',
                        help='Flag that weights the loss based on the class balance of the dataset. Recommended to use when data is imbalanced. ')
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help='Random seed. Note that there will still be differences between runs due to nondeterminism. See https://pytorch.org/docs/stable/notes/randomness.html')
    parser.add_argument('--gpu_ids',
                        type=str,
                        default='',
                        help='ID of gpu. Can be separated with comma')
    parser.add_argument('--num_workers',
                        type=int,
                        default=8,
                        help='Num workers in dataloaders.')
    parser.add_argument('--bias',
                        action='store_true',
                        help='Flag that indicates whether to include a trainable bias in the linear classification layer.'
                        )
    parser.add_argument('--add_on_bias',
                        action='store_true',
                        help='Flag that indicates whether to include a trainable bias to the add_on layer.'
                        )
    parser.add_argument('--extra_test_image_folder',
                        type=str,
                        default='./experiments',
                        help='Folder with images that PIP-Net will predict and explain, that are not in the training or test set. E.g. images with 2 objects or OOD image. Images should be in subfolder. E.g. images in ./experiments/images/, and argument --./experiments')
    parser.add_argument("--phylo_config",
                        type=str,
                        default=None, 
                        help='path to the yaml file containing "phylogeny_path" and "phyloDistances_string"') # "./configs/cub27_phylogeny.yaml"
    parser.add_argument("--experiment_note",
                        type=str,
                        default='No note', 
                        help='Note on the experiment')
    parser.add_argument('--kernel_orth',
                        type=str,
                        default='n',
                        help='(y/n) Flag that indicates whether to apply kernel orthogonality on the last conv kernels (prototypes).'
                        )
    parser.add_argument('--num_protos_per_descendant',
                        type=int,
                        default=4,
                        help='Used for deciding the num of protos to assign for each node based on the number of descendants.')
    parser.add_argument('--copy_files',
                        type=str,
                        default='y',
                        help='(y/n) Flag that indicates whether to copy all py, sh, ipynb, yaml files.'
                        )
    parser.add_argument('--tanh_desc',
                        type=str,
                        default='y',
                        help='(y/n) Flag that indicates whether to use tanh descendant loss or not.'
                        )
    parser.add_argument('--align',
                        type=str,
                        default='y',
                        help='(y/n) Flag that indicates whether to use align loss or not.'
                        )
    parser.add_argument('--uni',
                        type=str,
                        default='y',
                        help='(y/n) Flag that indicates whether to use uni loss or not.'
                        )
    parser.add_argument('--align_pf',
                        type=str,
                        default='n',
                        help='(y/n) Flag that indicates whether to use align_pf (CARL style align_pf) loss or not.'
                        )
    parser.add_argument('--minmaximize',
                        type=str,
                        default='n',
                        help='(y/n) Flag that indicates whether to use minmaximize loss or not.'
                        )
    parser.add_argument('--cluster_desc',
                        type=str,
                        default='n',
                        help='(y/n) Flag that indicates whether to use descendant specific cluster (overspecificity) loss or not.'
                        )
    parser.add_argument('--sep_desc',
                        type=str,
                        default='n',
                        help='(y/n) Flag that indicates whether to use descendant specific seperation loss or not.'
                        )
    parser.add_argument('--subspace_sep',
                        type=str,
                        default='n',
                        help='(y/n) Flag that indicates whether to use subspace seperation loss or not.'
                        )
    parser.add_argument('--softmax',
                        type=str,
                        default='n',
                        help='(y/n or y|softmax_tau) Flag that indicates whether to use softmax on the inner product between prototype and latent patch. Takes precedence over gumbel_softmax'
                        )
    parser.add_argument('--gumbel_softmax',
                        type=str,
                        default='n',
                        help='(y/n) Flag that indicates whether to use gumbel_softmax on the inner product between prototype and latent patch'
                        )
    parser.add_argument('--gs_tau',
                        type=float,
                        default=0.5,
                        help='Temperature to use with gumbel softmax')
    parser.add_argument('--multiply_cs_softmax',
                        type=str,
                        default='n',
                        help='(y/n) Flag that indicates whether to multiply cosine similarity and softmax. Must have anyone of softmax or gumbel_softmax turned ON'
                        )
    parser.add_argument('--unitconv2d',
                        type=str,
                        default='n',
                        help='(y/n) Flag that indicates whether to use UnitConv2D or nn.Conv2D between prototypes and features'
                        )
    parser.add_argument('--projectconv2d',
                        type=str,
                        default='n',
                        help='(y/n) Flag that indicates whether to use UnitConv2D or nn.Conv2D between prototypes and features'
                        )
    parser.add_argument('--focal',
                        type=str,
                        default='n',
                        help='(y/n) Flag that indicates whether to use focal similarity'
                        )
    parser.add_argument('--wandb',
                        type=str,
                        default='y',
                        help='(y/n) Flag to enable/disable wandb logging'
                        )
    parser.add_argument('--training_wheels',
                        type=str,
                        default='n',
                        help='(y/n) Flag to do trial run of the code'
                        )
    parser.add_argument('--weighted_ce_loss',
                        type=str,
                        default='n',
                        help='(y/n) Flag to indicate whether to use weighted loss for classification. This actually uses weighted NLLLoss'
                        )
    parser.add_argument('--protopool',
                        type=str,
                        default='y',
                        help='(y/n) If yes all prototypes are common to all child classes'
                        )
    parser.add_argument('--focal_loss',
                        type=str,
                        default='n',
                        help='(y/n) Flag to indicate focal loss'
                        )
    parser.add_argument('--focal_loss_gamma',
                        type=float,
                        default=2.0,
                        help='Gamma for focal loss')
    parser.add_argument('--stage4_reducer_net',
                        type=str,
                        default='',
                        help='Architecture of the reducer net defined in in,out|in,out format'
                        )
    parser.add_argument('--basic_cnext_gaussian_multiplier',
                        type=str,
                        default='',
                        help='Basic gaussian multiplier that does multiplies after guassian by a fixed value, set sigma and factor ex 3,4|sigma|factor'
                        )
    parser.add_argument('--sg_before_protos',
                        type=str,
                        default='n',
                        help='Stop gradient before the prototype layer'
                        )
    parser.add_argument('--viz_loader',
                        type=str,
                        default='projectloader,test_loader,test_projectloader',
                        help='Currently not used'
                        )
    parser.add_argument('--leave_out_classes',
                        type=str,
                        default='',
                        help='Comma seperated list of class names'
                        )
    parser.add_argument('--byol',
                        type=str,
                        default='n',
                        help='BYOL style contrastive learning applied to patches'
                        )
    parser.add_argument('--disable_transform2',
                        type=str,
                        default='n',
                        help='Disables the second transform which affects color, contrast, saturations etc.'
                        )

    
    args = parser.parse_args()
    if len(args.log_dir.split('/'))>2:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)


    return args


def save_args(args: argparse.Namespace, directory_path: str) -> None:
    """
    Save the arguments in the specified directory as
        - a text file called 'args.txt'
        - a pickle file called 'args.pickle'
    :param args: The arguments to be saved
    :param directory_path: The path to the directory where the arguments should be saved
    """
    # If the specified directory does not exists, create it
    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)
    # Save the args in a text file
    with open(directory_path + '/args.txt', 'w') as f:
        for arg in vars(args):
            val = getattr(args, arg)
            if isinstance(val, str):  # Add quotation marks to indicate that the argument is of string type
                val = f"'{val}'"
            f.write('{}: {}\n'.format(arg, val))
    # Pickle the args for possible reuse
    with open(directory_path + '/args.pickle', 'wb') as f:
        pickle.dump(args, f)          \


class CombinedOptimizer(torch.optim.Optimizer):
    def __init__(self, optimizers):
        self.optimizers = optimizers
        self.param_groups = [param_group for optimizer in optimizers for param_group in optimizer.param_groups]

    def zero_grad(self, set_to_none=False):
        for optimizer in self.optimizers:
            if type(optimizer) == LARS:
                optimizer.zero_grad()
            else:
                optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        for optimizer in self.optimizers:
            optimizer.step(closure)

    def state_dict(self):
        return [optimizer.state_dict() for optimizer in self.optimizers]

    def load_state_dict(self, state_dicts):
        for optimizer, state_dict in zip(self.optimizers, state_dicts):
            optimizer.load_state_dict(state_dict)                                                                     

# class CombinedOptimizer(torch.optim.Optimizer):
#     def __init__(self, optimizers):
#         self.optimizers = optimizers

#     def zero_grad(self):
#         for optimizer in self.optimizers:
#             optimizer.zero_grad()

#     def step(self):
#         for optimizer in self.optimizers:
#             optimizer.step()

def is_bias_or_batchnorm(param_name):
    return 'bias' in param_name or 'BatchNorm' in param_name

# Create a parameter group excluding bias and batchnorm parameters for weight decay
def exclude_bias_and_batchnorm(named_params):
    params = []
    excluded_params = []
    for name, param in named_params.items():
        if not param.requires_grad:
            continue  # Skip frozen weights
        if is_bias_or_batchnorm(name):
            # Exclude from LARS adaptation and weight decay
            excluded_params.append(param)
        else:
            # Include in LARS adaptation and weight decay
            params.append(param)
    return params, excluded_params

def get_optimizer_nn_byol(net, args: argparse.Namespace) -> torch.optim.Optimizer:

    #create parameter groups
    params_to_freeze = {}
    params_to_train = {}
    params_backbone = {}
    # set up optimizer
    if 'resnet50' in args.net: 
        # freeze resnet50 except last convolutional layer
        for name,param in net.module._net.named_parameters():
            if 'layer4.2' in name:
                params_to_train[name] = param
            elif 'layer4' in name or 'layer3' in name:
                params_to_freeze[name] = param
            elif 'layer2' in name:
                params_backbone[name] = param
            else: #such that model training fits on one gpu. 
                param.requires_grad = False
                # params_backbone.append(param)
    elif 'resnet18' in args.net: 
        # freeze resnet50 except last convolutional layer
        for name,param in net.module._net.named_parameters():
            if 'layer4.1' in name:
                params_to_train[name] = param
            elif 'layer4' in name or 'layer3' in name:
                params_to_freeze[name] = param
            # elif 'layer2' in name:
            #     params_backbone.append(param)
            else: #such that model training fits on one gpu. 
                # param.requires_grad = False
                params_backbone[name] = param
    elif 'resnet34' in args.net: 
        # freeze resnet50 except last convolutional layer
        for name,param in net.module._net.named_parameters():
            if 'layer4.2' in name:
                params_to_train[name] = param
            elif 'layer4' in name or 'layer3' in name:
                params_to_freeze[name] = param
            # elif 'layer2' in name:
            #     params_backbone.append(param)
            else: #such that model training fits on one gpu. 
                # param.requires_grad = False
                params_backbone[name] = param
    elif 'convnext' in args.net:
        print("chosen network is convnext", flush=True)
        for name,param in net.module._net.named_parameters():
            if 'features.7.2' in name: 
                params_to_train[name] = param
            elif 'stage4_reducer' in name:
                params_to_train[name] = param
            elif 'features.7' in name or 'features.6' in name:
                params_to_freeze[name] = param
            # CUDA MEMORY ISSUES? COMMENT LINE 202-203 AND USE THE FOLLOWING LINES INSTEAD
            # elif 'features.5' in name or 'features.4' in name:
            #     params_backbone.append(param)
            # else:
            #     param.requires_grad = False
            else:
                params_backbone[name] = param
    else:
        print("Network is not ResNet or ConvNext.", flush=True)     

    classification_weight = []
    classification_bias = []
    for attr in dir(net.module):
        if attr.endswith('_classification'):
            for name, param in getattr(net.module, attr).named_parameters():
                if 'weight' in name:
                    classification_weight.append(param)
                elif 'multiplier' in name:
                    param.requires_grad = False # TBC remove/move this if the parameter is moved to pipnet
                else:
                    if args.bias:
                        classification_bias.append(param)

    paramlist_classifier = [
            {"params": classification_weight, "lr": args.lr, "weight_decay_rate": args.weight_decay},
            {"params": classification_bias, "lr": args.lr, "weight_decay_rate": 0},
    ]
    
    paramlist_add_on = []
    for attr in dir(net.module):
        if attr.endswith('_add_on'):
            paramlist_add_on.append({"params": getattr(net.module, attr).parameters(), "lr": args.lr_block*10., "weight_decay_rate": args.weight_decay})
    
    params_backbone_lars, params_backbone_non_lars = exclude_bias_and_batchnorm(params_backbone)
    params_to_freeze_lars, params_to_freeze_non_lars = exclude_bias_and_batchnorm(params_to_freeze)
    params_to_train_lars, params_to_train_non_lars = exclude_bias_and_batchnorm(params_to_train)
    params_projector_lars, params_projector_non_lars = exclude_bias_and_batchnorm({name: param for name, param in net.module._projector.named_parameters()})
    params_predictor_lars, params_predictor_non_lars = exclude_bias_and_batchnorm({name: param for name, param in net.module._predictor.named_parameters()})

    paramlist_net_non_lars = [{"params": params_backbone_non_lars, "lr": args.lr_net, "weight_decay_rate": args.weight_decay},
                                 {"params": params_to_freeze_non_lars, "lr": args.lr_block, "weight_decay_rate": args.weight_decay},
                                 {"params": params_to_train_non_lars, "lr": args.lr_block, "weight_decay_rate": args.weight_decay},
                                 {"params": params_projector_non_lars, "lr": args.lr_block*10, "weight_decay_rate": args.weight_decay},
                                 {"params": params_predictor_non_lars, "lr": args.lr_block*10, "weight_decay_rate": args.weight_decay}]
    
    paramlist_net_lars = [{"params": params_backbone_lars, "lr": args.lr_net, "weight_decay_rate": args.weight_decay},
                            {"params": params_to_freeze_lars, "lr": args.lr_block, "weight_decay_rate": args.weight_decay},
                            {"params": params_to_train_lars, "lr": args.lr_block, "weight_decay_rate": args.weight_decay},
                            {"params": params_projector_lars, "lr": args.lr_block*10, "weight_decay_rate": args.weight_decay},
                            {"params": params_predictor_lars, "lr": args.lr_block*10, "weight_decay_rate": args.weight_decay}]

    optimizer_net_lars = LARS(optimizer=torch.optim.SGD(paramlist_net_lars, lr=0.1, momentum=0.9, weight_decay=1.5e-6), trust_coef=1e-3)
    optimizer_net_non_lars = torch.optim.SGD(paramlist_net_non_lars, lr=0.1, momentum=0.9, weight_decay=0.0)
    if args.optimizer == 'Adam':
        optimizer_add_on = torch.optim.AdamW(paramlist_add_on,lr=args.lr,weight_decay=args.weight_decay)
        optimizer_classifier = torch.optim.AdamW(paramlist_classifier,lr=args.lr,weight_decay=args.weight_decay)
    else:
        raise ValueError("this optimizer type is not implemented")
    
    optimizer_net = CombinedOptimizer([optimizer_net_lars, optimizer_net_non_lars, optimizer_add_on])

    return optimizer_net, optimizer_classifier, list(params_to_freeze.values()), list(params_to_train.values()), list(params_backbone.values())

def get_optimizer_nn_byol2(net, args: argparse.Namespace) -> torch.optim.Optimizer:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    #create parameter groups
    params_to_freeze = []
    params_to_train = []
    params_backbone = []
    # set up optimizer
    if 'resnet50' in args.net: 
        # freeze resnet50 except last convolutional layer
        for name,param in net.module._net.named_parameters():
            if 'layer4.2' in name:
                params_to_train.append(param)
            elif 'layer4' in name or 'layer3' in name:
                params_to_freeze.append(param)
            elif 'layer2' in name:
                params_backbone.append(param)
            else: #such that model training fits on one gpu. 
                param.requires_grad = False
                # params_backbone.append(param)
    elif 'resnet18' in args.net: 
        # freeze resnet50 except last convolutional layer
        for name,param in net.module._net.named_parameters():
            if 'layer4.1' in name:
                params_to_train.append(param)
            elif 'layer4' in name or 'layer3' in name:
                params_to_freeze.append(param)
            # elif 'layer2' in name:
            #     params_backbone.append(param)
            else: #such that model training fits on one gpu. 
                # param.requires_grad = False
                params_backbone.append(param)
    elif 'resnet34' in args.net: 
        # freeze resnet50 except last convolutional layer
        for name,param in net.module._net.named_parameters():
            if 'layer4.2' in name:
                params_to_train.append(param)
            elif 'layer4' in name or 'layer3' in name:
                params_to_freeze.append(param)
            # elif 'layer2' in name:
            #     params_backbone.append(param)
            else: #such that model training fits on one gpu. 
                # param.requires_grad = False
                params_backbone.append(param)
    elif 'convnext' in args.net:
        print("chosen network is convnext", flush=True)
        for name,param in net.module._net.named_parameters():
            if 'features.7.2' in name: 
                params_to_train.append(param)
            elif 'stage4_reducer' in name:
                params_to_train.append(param)
            elif 'features.7' in name or 'features.6' in name:
                params_to_freeze.append(param)
            # CUDA MEMORY ISSUES? COMMENT LINE 202-203 AND USE THE FOLLOWING LINES INSTEAD
            # elif 'features.5' in name or 'features.4' in name:
            #     params_backbone.append(param)
            # else:
            #     param.requires_grad = False
            else:
                params_backbone.append(param)
    else:
        print("Network is not ResNet or ConvNext.", flush=True)     

    classification_weight = []
    classification_bias = []
    for attr in dir(net.module):
        if attr.endswith('_classification'):
            for name, param in getattr(net.module, attr).named_parameters():
                if 'weight' in name:
                    classification_weight.append(param)
                elif 'multiplier' in name:
                    param.requires_grad = False # TBC remove/move this if the parameter is moved to pipnet
                else:
                    if args.bias:
                        classification_bias.append(param)
    
    paramlist_net = [
            {"params": params_backbone, "lr": args.lr_net, "weight_decay_rate": args.weight_decay},
            {"params": params_to_freeze, "lr": args.lr_block, "weight_decay_rate": args.weight_decay},
            {"params": params_to_train, "lr": args.lr_block, "weight_decay_rate": args.weight_decay},
            {"params": net.module._predictor.parameters(), "lr": args.lr_block*10., "weight_decay_rate": args.weight_decay},
            {"params": net.module._projector.parameters(), "lr": args.lr_block*10., "weight_decay_rate": args.weight_decay}]
            # {"params": net.module._add_on.parameters(), "lr": args.lr_block*10., "weight_decay_rate": args.weight_decay}]
    for attr in dir(net.module):
        if attr.endswith('_add_on'):
            paramlist_net.append({"params": getattr(net.module, attr).parameters(), "lr": args.lr_block*10., "weight_decay_rate": args.weight_decay})
            
    paramlist_classifier = [
            {"params": classification_weight, "lr": args.lr, "weight_decay_rate": args.weight_decay},
            {"params": classification_bias, "lr": args.lr, "weight_decay_rate": 0},
    ]
    
    
    if args.optimizer == 'Adam':
        optimizer_net = torch.optim.AdamW(paramlist_net,lr=args.lr,weight_decay=args.weight_decay)
        optimizer_classifier = torch.optim.AdamW(paramlist_classifier,lr=args.lr,weight_decay=args.weight_decay)
        return optimizer_net, optimizer_classifier, params_to_freeze, params_to_train, params_backbone
    else:
        raise ValueError("this optimizer type is not implemented")

def get_optimizer_nn_byol3(net, args: argparse.Namespace) -> torch.optim.Optimizer:

    """
    Not training batch norm weight and bias in batchnorm layer
    No SGD or LARS
    """

    #create parameter groups
    params_to_freeze = {}
    params_to_train = {}
    params_backbone = {}
    # set up optimizer
    if 'resnet50' in args.net: 
        # freeze resnet50 except last convolutional layer
        for name,param in net.module._net.named_parameters():
            if 'layer4.2' in name:
                params_to_train[name] = param
            elif 'layer4' in name or 'layer3' in name:
                params_to_freeze[name] = param
            elif 'layer2' in name:
                params_backbone[name] = param
            else: #such that model training fits on one gpu. 
                param.requires_grad = False
                # params_backbone.append(param)
    elif 'resnet18' in args.net: 
        # freeze resnet50 except last convolutional layer
        for name,param in net.module._net.named_parameters():
            if 'layer4.1' in name:
                params_to_train[name] = param
            elif 'layer4' in name or 'layer3' in name:
                params_to_freeze[name] = param
            # elif 'layer2' in name:
            #     params_backbone.append(param)
            else: #such that model training fits on one gpu. 
                # param.requires_grad = False
                params_backbone[name] = param
    elif 'resnet34' in args.net: 
        # freeze resnet50 except last convolutional layer
        for name,param in net.module._net.named_parameters():
            if 'layer4.2' in name:
                params_to_train[name] = param
            elif 'layer4' in name or 'layer3' in name:
                params_to_freeze[name] = param
            # elif 'layer2' in name:
            #     params_backbone.append(param)
            else: #such that model training fits on one gpu. 
                # param.requires_grad = False
                params_backbone[name] = param
    elif 'convnext' in args.net:
        print("chosen network is convnext", flush=True)
        for name,param in net.module._net.named_parameters():
            if 'features.7.2' in name: 
                params_to_train[name] = param
            elif 'stage4_reducer' in name:
                params_to_train[name] = param
            elif 'features.7' in name or 'features.6' in name:
                params_to_freeze[name] = param
            # CUDA MEMORY ISSUES? COMMENT LINE 202-203 AND USE THE FOLLOWING LINES INSTEAD
            # elif 'features.5' in name or 'features.4' in name:
            #     params_backbone.append(param)
            # else:
            #     param.requires_grad = False
            else:
                params_backbone[name] = param
    else:
        print("Network is not ResNet or ConvNext.", flush=True)     

    classification_weight = []
    classification_bias = []
    for attr in dir(net.module):
        if attr.endswith('_classification'):
            for name, param in getattr(net.module, attr).named_parameters():
                if 'weight' in name:
                    classification_weight.append(param)
                elif 'multiplier' in name:
                    param.requires_grad = False # TBC remove/move this if the parameter is moved to pipnet
                else:
                    if args.bias:
                        classification_bias.append(param)

    paramlist_classifier = [
            {"params": classification_weight, "lr": args.lr, "weight_decay_rate": args.weight_decay},
            {"params": classification_bias, "lr": args.lr, "weight_decay_rate": 0},
    ]
    
    paramlist_add_on = []
    for attr in dir(net.module):
        if attr.endswith('_add_on'):
            paramlist_add_on.append({"params": getattr(net.module, attr).parameters(), "lr": args.lr_block*10., "weight_decay_rate": args.weight_decay})
    
    params_backbone_lars, params_backbone_non_lars = exclude_bias_and_batchnorm(params_backbone)
    params_to_freeze_lars, params_to_freeze_non_lars = exclude_bias_and_batchnorm(params_to_freeze)
    params_to_train_lars, params_to_train_non_lars = exclude_bias_and_batchnorm(params_to_train)

    params_projector_non_lars = [param for name, param in net.module._projector.named_parameters()][2:4] # [name for name, param in net.module._projector.named_parameters()]
    params_projector_lars = [param for name, param in net.module._projector.named_parameters()][:2]
    params_projector_lars += [param for name, param in net.module._projector.named_parameters()][4:]

    params_predictor_non_lars = [param for name, param in net.module._predictor.named_parameters()][2:4]
    params_predictor_lars = [param for name, param in net.module._predictor.named_parameters()][:2]
    params_predictor_lars += [param for name, param in net.module._predictor.named_parameters()][4:]

    # params_projector_lars, params_projector_non_lars = exclude_bias_and_batchnorm({name: param for name, param in net.module._projector.named_parameters()})
    # params_predictor_lars, params_predictor_non_lars = exclude_bias_and_batchnorm({name: param for name, param in net.module._predictor.named_parameters()})

    paramlist_net_non_lars = [{"params": params_backbone_non_lars, "lr": args.lr_net, "weight_decay_rate": args.weight_decay},
                                 {"params": params_to_freeze_non_lars, "lr": args.lr_block, "weight_decay_rate": args.weight_decay},
                                 {"params": params_to_train_non_lars, "lr": args.lr_block, "weight_decay_rate": args.weight_decay},
                                 {"params": params_projector_non_lars, "lr": args.lr_block*10, "weight_decay_rate": args.weight_decay},
                                 {"params": params_predictor_non_lars, "lr": args.lr_block*10, "weight_decay_rate": args.weight_decay}]
    
    paramlist_net_lars = [{"params": params_backbone_lars, "lr": args.lr_net, "weight_decay_rate": args.weight_decay},
                            {"params": params_to_freeze_lars, "lr": args.lr_block, "weight_decay_rate": args.weight_decay},
                            {"params": params_to_train_lars, "lr": args.lr_block, "weight_decay_rate": args.weight_decay},
                            {"params": params_projector_lars, "lr": args.lr_block*10, "weight_decay_rate": args.weight_decay},
                            {"params": params_predictor_lars, "lr": args.lr_block*10, "weight_decay_rate": args.weight_decay}]

    optimizer_net_lars = LARS(optimizer=torch.optim.SGD(paramlist_net_lars, lr=0.1, momentum=0.9, weight_decay=1.5e-6), trust_coef=1e-3)
    # optimizer_net_non_lars = torch.optim.SGD(paramlist_net_non_lars, lr=0.1, momentum=0.9, weight_decay=0.0)
    if args.optimizer == 'Adam':
        optimizer_add_on = torch.optim.AdamW(paramlist_add_on,lr=args.lr,weight_decay=args.weight_decay)
        optimizer_classifier = torch.optim.AdamW(paramlist_classifier,lr=args.lr,weight_decay=args.weight_decay)
    else:
        raise ValueError("this optimizer type is not implemented")
    
    optimizer_net = CombinedOptimizer([optimizer_net_lars, optimizer_add_on])
    # optimizer_net = CombinedOptimizer([optimizer_net_lars, optimizer_net_non_lars, optimizer_add_on])

    return optimizer_net, optimizer_classifier, list(params_to_freeze.values()), list(params_to_train.values()), list(params_backbone.values())

def get_optimizer_nn_byol4(net, args: argparse.Namespace) -> torch.optim.Optimizer:

    """
    Not training batch norm weight and bias in batchnorm layer
    No SGD or LARS
    """

    #create parameter groups
    params_to_freeze = {}
    params_to_train = {}
    params_backbone = {}
    # set up optimizer
    if 'resnet50' in args.net: 
        # freeze resnet50 except last convolutional layer
        for name,param in net.module._net.named_parameters():
            if 'layer4.2' in name:
                params_to_train[name] = param
            elif 'layer4' in name or 'layer3' in name:
                params_to_freeze[name] = param
            elif 'layer2' in name:
                params_backbone[name] = param
            else: #such that model training fits on one gpu. 
                param.requires_grad = False
                # params_backbone.append(param)
    elif 'resnet18' in args.net: 
        # freeze resnet50 except last convolutional layer
        for name,param in net.module._net.named_parameters():
            if 'layer4.1' in name:
                params_to_train[name] = param
            elif 'layer4' in name or 'layer3' in name:
                params_to_freeze[name] = param
            # elif 'layer2' in name:
            #     params_backbone.append(param)
            else: #such that model training fits on one gpu. 
                # param.requires_grad = False
                params_backbone[name] = param
    elif 'resnet34' in args.net: 
        # freeze resnet50 except last convolutional layer
        for name,param in net.module._net.named_parameters():
            if 'layer4.2' in name:
                params_to_train[name] = param
            elif 'layer4' in name or 'layer3' in name:
                params_to_freeze[name] = param
            # elif 'layer2' in name:
            #     params_backbone.append(param)
            else: #such that model training fits on one gpu. 
                # param.requires_grad = False
                params_backbone[name] = param
    elif 'convnext' in args.net:
        print("chosen network is convnext", flush=True)
        for name,param in net.module._net.named_parameters():
            if 'features.7.2' in name: 
                params_to_train[name] = param
            elif 'stage4_reducer' in name:
                params_to_train[name] = param
            elif 'features.7' in name or 'features.6' in name:
                params_to_freeze[name] = param
            # CUDA MEMORY ISSUES? COMMENT LINE 202-203 AND USE THE FOLLOWING LINES INSTEAD
            # elif 'features.5' in name or 'features.4' in name:
            #     params_backbone.append(param)
            # else:
            #     param.requires_grad = False
            else:
                params_backbone[name] = param
    else:
        print("Network is not ResNet or ConvNext.", flush=True)     

    classification_weight = []
    classification_bias = []
    for attr in dir(net.module):
        if attr.endswith('_classification'):
            for name, param in getattr(net.module, attr).named_parameters():
                if 'weight' in name:
                    classification_weight.append(param)
                elif 'multiplier' in name:
                    param.requires_grad = False # TBC remove/move this if the parameter is moved to pipnet
                else:
                    if args.bias:
                        classification_bias.append(param)

    paramlist_classifier = [
            {"params": classification_weight, "lr": args.lr, "weight_decay_rate": args.weight_decay},
            {"params": classification_bias, "lr": args.lr, "weight_decay_rate": 0},
    ]
    
    paramlist_add_on = []
    for attr in dir(net.module):
        if attr.endswith('_add_on'):
            paramlist_add_on.append({"params": getattr(net.module, attr).parameters(), "lr": args.lr_block*10., "weight_decay_rate": args.weight_decay})
    
    params_backbone_lars, params_backbone_non_lars = exclude_bias_and_batchnorm(params_backbone)
    params_to_freeze_lars, params_to_freeze_non_lars = exclude_bias_and_batchnorm(params_to_freeze)
    params_to_train_lars, params_to_train_non_lars = exclude_bias_and_batchnorm(params_to_train)

    params_projector_non_lars = [param for name, param in net.module._projector.named_parameters()][2:4] # [name for name, param in net.module._projector.named_parameters()]
    params_projector_lars = [param for name, param in net.module._projector.named_parameters()][:2]
    params_projector_lars += [param for name, param in net.module._projector.named_parameters()][4:]

    params_predictor_non_lars = [param for name, param in net.module._predictor.named_parameters()][2:4]
    params_predictor_lars = [param for name, param in net.module._predictor.named_parameters()][:2]
    params_predictor_lars += [param for name, param in net.module._predictor.named_parameters()][4:]

    # params_projector_lars, params_projector_non_lars = exclude_bias_and_batchnorm({name: param for name, param in net.module._projector.named_parameters()})
    # params_predictor_lars, params_predictor_non_lars = exclude_bias_and_batchnorm({name: param for name, param in net.module._predictor.named_parameters()})

    paramlist_net_non_lars = [{"params": params_backbone_non_lars, "lr": args.lr_net, "weight_decay_rate": args.weight_decay},
                                 {"params": params_to_freeze_non_lars, "lr": args.lr_block, "weight_decay_rate": args.weight_decay},
                                 {"params": params_to_train_non_lars, "lr": args.lr_block, "weight_decay_rate": args.weight_decay},
                                 {"params": params_projector_non_lars, "lr": args.lr_block*10, "weight_decay_rate": args.weight_decay},
                                 {"params": params_predictor_non_lars, "lr": args.lr_block*10, "weight_decay_rate": args.weight_decay}]
    
    paramlist_net_lars = [{"params": params_backbone_lars, "lr": args.lr_net, "weight_decay_rate": args.weight_decay},
                            {"params": params_to_freeze_lars, "lr": args.lr_block, "weight_decay_rate": args.weight_decay},
                            {"params": params_to_train_lars, "lr": args.lr_block, "weight_decay_rate": args.weight_decay},
                            {"params": params_projector_lars, "lr": args.lr_block*10, "weight_decay_rate": args.weight_decay},
                            {"params": params_predictor_lars, "lr": args.lr_block*10, "weight_decay_rate": args.weight_decay}]

    # optimizer_net_lars = LARS(optimizer=torch.optim.SGD(paramlist_net_lars, lr=0.1, momentum=0.9, weight_decay=1.5e-6), trust_coef=1e-3)
    optimizer_net_adamW = torch.optim.AdamW(paramlist_net_lars,lr=args.lr,weight_decay=args.weight_decay)
    # optimizer_net_non_lars = torch.optim.SGD(paramlist_net_non_lars, lr=0.1, momentum=0.9, weight_decay=0.0)
    if args.optimizer == 'Adam':
        optimizer_add_on = torch.optim.AdamW(paramlist_add_on,lr=args.lr,weight_decay=args.weight_decay)
        optimizer_classifier = torch.optim.AdamW(paramlist_classifier,lr=args.lr,weight_decay=args.weight_decay)
    else:
        raise ValueError("this optimizer type is not implemented")
    
    optimizer_net = CombinedOptimizer([optimizer_net_adamW, optimizer_add_on])
    # optimizer_net = CombinedOptimizer([optimizer_net_lars, optimizer_net_non_lars, optimizer_add_on])

    return optimizer_net, optimizer_classifier, list(params_to_freeze.values()), list(params_to_train.values()), list(params_backbone.values())


def get_optimizer_nn(net, args: argparse.Namespace) -> torch.optim.Optimizer:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.byol == 'y':
        return get_optimizer_nn_byol2(net, args)

    #create parameter groups
    params_to_freeze = []
    params_to_train = []
    params_backbone = []
    # set up optimizer
    if 'resnet50' in args.net: 
        # freeze resnet50 except last convolutional layer
        for name,param in net.module._net.named_parameters():
            if 'layer4.2' in name:
                params_to_train.append(param)
            elif 'layer4' in name or 'layer3' in name:
                params_to_freeze.append(param)
            elif 'layer2' in name:
                params_backbone.append(param)
            else: #such that model training fits on one gpu. 
                param.requires_grad = False
                # params_backbone.append(param)
    elif 'resnet18' in args.net: 
        # freeze resnet50 except last convolutional layer
        for name,param in net.module._net.named_parameters():
            if 'layer4.1' in name:
                params_to_train.append(param)
            elif 'layer4' in name or 'layer3' in name:
                params_to_freeze.append(param)
            # elif 'layer2' in name:
            #     params_backbone.append(param)
            else: #such that model training fits on one gpu. 
                # param.requires_grad = False
                params_backbone.append(param)
    elif 'resnet34' in args.net: 
        # freeze resnet50 except last convolutional layer
        for name,param in net.module._net.named_parameters():
            if 'layer4.2' in name:
                params_to_train.append(param)
            elif 'layer4' in name or 'layer3' in name:
                params_to_freeze.append(param)
            # elif 'layer2' in name:
            #     params_backbone.append(param)
            else: #such that model training fits on one gpu. 
                # param.requires_grad = False
                params_backbone.append(param)
    elif 'convnext' in args.net:
        print("chosen network is convnext", flush=True)
        for name,param in net.module._net.named_parameters():
            if 'features.7.2' in name: 
                params_to_train.append(param)
            elif 'stage4_reducer' in name:
                params_to_train.append(param)
            elif 'features.7' in name or 'features.6' in name:
                params_to_freeze.append(param)
            # CUDA MEMORY ISSUES? COMMENT LINE 202-203 AND USE THE FOLLOWING LINES INSTEAD
            # elif 'features.5' in name or 'features.4' in name:
            #     params_backbone.append(param)
            # else:
            #     param.requires_grad = False
            else:
                params_backbone.append(param)
    else:
        print("Network is not ResNet or ConvNext.", flush=True)     

    classification_weight = []
    classification_bias = []
    for attr in dir(net.module):
        if attr.endswith('_classification'):
            for name, param in getattr(net.module, attr).named_parameters():
                if 'weight' in name:
                    classification_weight.append(param)
                elif 'multiplier' in name:
                    param.requires_grad = False # TBC remove/move this if the parameter is moved to pipnet
                else:
                    if args.bias:
                        classification_bias.append(param)
    
    paramlist_net = [
            {"params": params_backbone, "lr": args.lr_net, "weight_decay_rate": args.weight_decay},
            {"params": params_to_freeze, "lr": args.lr_block, "weight_decay_rate": args.weight_decay},
            {"params": params_to_train, "lr": args.lr_block, "weight_decay_rate": args.weight_decay}]
            # {"params": net.module._add_on.parameters(), "lr": args.lr_block*10., "weight_decay_rate": args.weight_decay}]
    for attr in dir(net.module):
        if attr.endswith('_add_on'):
            paramlist_net.append({"params": getattr(net.module, attr).parameters(), "lr": args.lr_block*10., "weight_decay_rate": args.weight_decay})
            
    paramlist_classifier = [
            {"params": classification_weight, "lr": args.lr, "weight_decay_rate": args.weight_decay},
            {"params": classification_bias, "lr": args.lr, "weight_decay_rate": 0},
    ]
    
    
    if args.optimizer == 'Adam':
        optimizer_net = torch.optim.AdamW(paramlist_net,lr=args.lr,weight_decay=args.weight_decay)
        optimizer_classifier = torch.optim.AdamW(paramlist_classifier,lr=args.lr,weight_decay=args.weight_decay)
        return optimizer_net, optimizer_classifier, params_to_freeze, params_to_train, params_backbone
    else:
        raise ValueError("this optimizer type is not implemented")

