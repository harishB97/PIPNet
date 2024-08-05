from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import math
import numpy as np
import torch.nn as nn
import torch.distributed as dist

from collections import defaultdict
from torchmetrics.functional import f1_score, recall, precision
from torchvision.datasets.folder import ImageFolder

from util.log import Log

from pipnet.pipnet import functional_UnitConv2D

import os
import gc

import pdb

# import wandb

OOD_LABEL = -1

def entropy_loss(probs):
    """
    Calculate the entropy of a probability distribution.
    :param probs: Tensor of shape (N, C) where N is the batch size and C is the number of classes.
                  Each row should be a probability distribution over classes for a sample, i.e., sum to 1.
    :return: Scalar tensor representing the mean entropy across the batch.
    """
    probs = torch.clamp(probs, min=1e-9)  # Prevent log(0)
    batch_entropy = -torch.sum(probs * torch.log(probs), dim=1)
    return torch.mean(batch_entropy)

def ema(byol_tau, online_network, target_network):
    with torch.no_grad():  # Ensure no gradients are computed for this operation
        for target_param, online_param in zip(target_network.parameters(), online_network.parameters()):
            target_param.data = byol_tau * target_param.data + (1 - byol_tau) * online_param.data

def findCorrespondingToMax(base, target):
    output, indices = F.max_pool2d(base, kernel_size=(base.shape[-1], base.shape[-2]), return_indices=True)# these are logits
    tensor_flattened = target.view(target.shape[0], target.shape[1], -1)
    indices_flattened = indices.view(target.shape[0], target.shape[1], -1)
    corresponding_values_in_target = torch.gather(tensor_flattened, 2, indices_flattened)
    corresponding_values_in_target = corresponding_values_in_target.view(target.shape[0],\
                                     target.shape[1], 1, 1)
    pooled_target = corresponding_values_in_target
    return pooled_target

def check_and_update_weights_dist(model, threshold=1e-3, add_value=0.1):
    rank = dist.get_rank()
    if rank == 0:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name.endswith('_classification'):
                    param.data[param.data <= threshold] += add_value
    # Ensure all processes wait until rank 0 finishes updating weights
    dist.barrier()
    # Broadcast the updated parameters from rank 0 to all processes
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

def check_and_update_weights(model, threshold=1e-3, add_value=0.1):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith('_classification'):
                param.data[param.data <= threshold] += add_value

def train_pipnet(net, train_loader, optimizer_net, optimizer_classifier, scheduler_net, scheduler_classifier, criterion, \
                 epoch, nr_epochs, device, pretrain=False, finetune=False, progress_prefix: str = 'Train Epoch', wandb_logging=True, \
                 train_loader_OOD=None, kernel_orth=False, tanh_desc=False, align=True, uni=True, align_pf=False, tanh=False, \
                 minmaximize=False, cluster_desc=False, sep_desc=False, subspace_sep=False, byol=False, byol_tau_base=0.9995, byol_tau_max=1., step_info=None, wandb_run=None, \
                 pretrain_epochs=0, log:Log=None, args=None, dist_training=False):

    root = net.module.root
    dataset = train_loader.dataset
    while type(dataset) != ImageFolder:
        dataset = dataset.dataset
    name2label = dataset.class_to_idx
    label2name = {label:name for name, label in name2label.items()}
    label2name[OOD_LABEL] = 'OOD'

    wandb_log_subdir = 'train' if not pretrain else ('pretrain')

    # node_accuracy = defaultdict(lambda: {'n_examples': 0, 'n_correct': 0, 'accuracy': None, 'preds': None, 'children': defaultdict(lambda: {'n_examples': 0, 'n_correct': 0})})
    node_accuracy = {}
    for node in root.nodes_with_children():
        node_accuracy[node.name] = {'n_examples': 0, 'n_correct': 0, 'accuracy': None, 'f1': None, 'preds': torch.empty(0, node.num_children()).cpu(), 'gts': torch.empty(0).cpu()}
        node_accuracy[node.name]['children'] = defaultdict(lambda: {'n_examples': 0, 'n_correct': 0})

    # Make sure the model is in train mode
    net.train()
    
    if pretrain:
        # Disable training of classification layer
        # net.module._classification.requires_grad = False
        for attr in dir(net.module):
            if attr.endswith('_classification'):
                getattr(net.module, attr).requires_grad = False
        progress_prefix = 'Pretrain Epoch'
    else:
        # Enable training of classification layer (disabled in case of pretraining)
        # net.module._classification.requires_grad = True
        for attr in dir(net.module):
            if attr.endswith('_classification'):
                getattr(net.module, attr).requires_grad = True
    
    # Store info about the procedure
    train_info = dict()
    total_loss = 0.
    total_acc = 0.
    class_loss_ep_mean = 0.
    a_loss_pf_ep_mean = 0.
    a_loss_ep_mean = 0.
    tanh_loss_ep_mean = 0.
    minmaximize_loss_ep_mean = 0.
    OOD_loss_ep_mean = 0.
    kernel_orth_loss_ep_mean = 0.
    uni_loss_ep_mean = 0.
    byol_loss_ep_mean = 0.
    cluster_desc_loss_ep_mean = 0.
    sep_desc_loss_ep_mean = 0.
    tanh_desc_loss_ep_mean = 0.
    subspace_sep_loss_ep_mean = 0.
    conc_log_ip_loss_ep_mean = 0.

    iters = len(train_loader)
    # Show progress on progress bar. 
    train_iter = tqdm(enumerate(train_loader),
                    total=len(train_loader),
                    desc=progress_prefix+'%s'%epoch,
                    mininterval=2.,
                    ncols=0)
    
    train_OOD_iter = iter(train_loader_OOD) if train_loader_OOD else None
    OOD_loss_required = True if train_loader_OOD else False
    
    count_param=0
    for name, param in net.named_parameters():
        if param.requires_grad:
            count_param+=1           
    print("Number of parameters that require gradient: ", count_param, flush=True)

    if pretrain:
        align_pf_weight = (epoch/nr_epochs)*1.
        # unif_weight = 0.5
        byol_weight = 0.5
        align_weight = 0.5 #3.
        unif_weight = 3.
        t_weight = 5.
        mm_weight = 0.
        cl_weight = 0.
        # optional losses
        OOD_loss_weight = 0.
        orth_weight = 0.5
        cluster_desc_weight = 0.8
        sep_desc_weight = 0.08
        subspace_sep_weight = 1e-2
    else:
        align_pf_weight = 5. 
        # unif_weight = 2. # 0.
        byol_weight = 2
        align_weight = 0.5 #3. 
        unif_weight = 3. #3. # 0.
        t_weight = 2.
        mm_weight = 2.
        cl_weight = args.cl_weight # 2.
        # optional losses
        OOD_loss_weight = 0.2
        orth_weight = 0.5
        cluster_desc_weight = 0.8
        sep_desc_weight = 0.08
        subspace_sep_weight = 1e-2


    print("Align weight: ", align_weight, "Align (CARL) weight: ", align_pf_weight, "Unif weight: ", unif_weight, ", Tanh-desc weight: ", t_weight, "Class weight:", cl_weight, "OOD_loss weight", OOD_loss_weight, flush=True)
    print("Pretrain?", pretrain, "Finetune?", finetune, flush=True)

    # maps, node_name -> loss_name -> list_of_loss_values_corresponding_to_each_step
    # node_wise_losses = defaultdict(lambda: defaultdict(list))
    node_wise_losses = {}
    for node in root.nodes_with_children():
        node_wise_losses[node.name] = {}
        node_wise_losses[node.name]['class_loss'] = []
        # node_wise_losses[node.name]['a_loss'] = []
        node_wise_losses[node.name]['tanh_loss'] = []
        node_wise_losses[node.name]['minmaximize_loss'] = []
        node_wise_losses[node.name]['OOD_loss'] = []
        node_wise_losses[node.name]['kernel_orth_loss'] = []
        # node_wise_losses[node.name]['uni_loss'] = []

    
    lrs_net = []
    lrs_class = []
    n_fine_correct = 0
    n_samples = 0
    # Iterate through the data set to update leaves, prototypes and network
    for i, (xs1, xs2, ys) in train_iter:       
        
        xs1, xs2, ys = xs1.to(device), xs2.to(device), ys.to(device)

        if train_OOD_iter:
            xs1_OOD, xs2_OOD, ys_OOD = next(train_OOD_iter)
            ys_OOD.fill_(OOD_LABEL)
            xs1_OOD, xs2_OOD, ys_OOD = xs1_OOD.to(device), xs2_OOD.to(device), ys_OOD.to(device)
            xs = torch.cat([xs1, xs2, xs1_OOD, xs2_OOD])
            ys = torch.cat([ys, ys, ys_OOD, ys_OOD])
        else:
            xs = torch.cat([xs1, xs2])
            ys = torch.cat([ys, ys])
       
        # Reset the gradients
        optimizer_classifier.zero_grad(set_to_none=True)
        optimizer_net.zero_grad(set_to_none=True)

        additional_network_outputs = {}
        
        
        # Perform a forward pass through the network
        if byol:
            online_network_out, target_network_out, features, proto_features, pooled, out = net(xs)
            additional_network_outputs['online_network_out'] = online_network_out
            additional_network_outputs['target_network_out'] = target_network_out
        else:
            features, proto_features, pooled, out = net(xs)
        
        loss, class_loss_dict, a_loss, tanh_loss_dict, minmaximize_loss_dict, OOD_loss_dict, kernel_orth_loss_dict, \
        uni_loss, avg_class_loss, avg_a_loss_pf, avg_tanh_loss, avg_minmaximize_loss, avg_OOD_loss, avg_kernel_orth_loss,\
        byol_loss, avg_cluster_desc_loss, avg_sep_desc_loss, avg_tanh_desc_loss, avg_subspace_sep_loss, avg_conc_log_ip_loss, acc = \
            calculate_loss(epoch, net, additional_network_outputs, features, proto_features, pooled, out, ys, align_weight=align_weight, align_pf_weight=align_pf_weight, \
                            t_weight=t_weight, mm_weight=mm_weight, unif_weight=unif_weight, cl_weight=cl_weight, OOD_loss_weight=OOD_loss_weight, orth_weight=orth_weight, \
                            cluster_desc_weight=cluster_desc_weight, sep_desc_weight=sep_desc_weight, subspace_sep_weight=subspace_sep_weight, byol_weight=byol_weight,
                            net_normalization_multiplier=net.module._multiplier, pretrain=pretrain, finetune=finetune, \
                           criterion=criterion, train_iter=train_iter, print=True, EPS=1e-8, root=root, label2name=label2name, node_accuracy=node_accuracy, \
                           OOD_loss_required=OOD_loss_required, kernel_orth=kernel_orth, tanh_desc=tanh_desc, align=align, uni=uni, align_pf=align_pf, tanh=tanh, minmaximize=minmaximize,\
                            cluster_desc=cluster_desc, sep_desc=sep_desc, subspace_sep=subspace_sep, byol=byol, args=args, device=device)

        # breakpoint() 
        # getattr(net.module, '_144+147_classification').weight
        # getattr(net.module, '_031+032_classification').weight
        # getattr(net.module, '_084+063_classification').weight
        # getattr(net.module, '_107+030_classification').weight
        

        
        # print(f"GPU Memory Usage: 0: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        # print(f"GPU Memory Usage: 0: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB, 1: {torch.cuda.memory_allocated(1) / 1024**2:.2f} MB")

        # with torch.no_grad():
        #     features1, features2 = features.chunk(2)
        #     flattened_meanpooled_features1 = flatten_tensor(features1)
        #     flattened_meanpooled_features2 = flatten_tensor(features2)
        #     normalized_flattened_meanpooled_features1 = F.normalize(flattened_meanpooled_features1, p=2, dim=1)
        #     normalized_flattened_meanpooled_features2 = F.normalize(flattened_meanpooled_features2, p=2, dim=1)
        #     true_uni_loss = (uniform_loss(normalized_flattened_meanpooled_features1) \
        #                 + uniform_loss(normalized_flattened_meanpooled_features2)) / 2.
        true_uni_loss = torch.tensor(-5)

        # Compute the gradient
        loss.backward()



        
        # del features
        # del proto_features
        # del pooled
        # del a_loss

        for node_name, loss_value in class_loss_dict.items():
            node_wise_losses[node_name]['class_loss'].append(loss_value.item())
        
        # del class_loss_dict

        # for node_name, loss_value in a_loss_pf_dict.items():
        #     node_wise_losses[node_name]['a_loss'].append(loss_value.item())

        for node_name, loss_value in minmaximize_loss_dict.items():
            node_wise_losses[node_name]['minmaximize_loss'].append(loss_value.item())

        for node_name, loss_value in tanh_loss_dict.items():
            node_wise_losses[node_name]['tanh_loss'].append(loss_value.item())

        # del tanh_loss_dict

        for node_name, loss_value in OOD_loss_dict.items():
            node_wise_losses[node_name]['OOD_loss'].append(loss_value.item())

        # del OOD_loss_dict

        for node_name, loss_value in kernel_orth_loss_dict.items():
            node_wise_losses[node_name]['kernel_orth_loss'].append(loss_value.item())

        # del kernel_orth_loss_dict

        # for node_name, loss_value in uni_loss_dict.items():
        #     node_wise_losses[node_name]['uni_loss'].append(loss_value.item())

        # to be modified, not all avg losses will be none if they're not used, some use integer placeholder value, to be modified accordingly
        class_loss_ep_mean += avg_class_loss if avg_class_loss else -5
        tanh_loss_ep_mean += avg_tanh_loss if avg_tanh_loss else -5
        minmaximize_loss_ep_mean += avg_minmaximize_loss if avg_minmaximize_loss else -5
        OOD_loss_ep_mean += avg_OOD_loss if avg_OOD_loss else -5
        kernel_orth_loss_ep_mean += avg_kernel_orth_loss if avg_kernel_orth_loss else -5

        # modifying the these two because they are not calculated for each node anymore
        a_loss_ep_mean += a_loss # avg_a_loss_pf if avg_a_loss_pf else -5
        uni_loss_ep_mean += uni_loss # avg_uni_loss if avg_uni_loss else -5

        byol_loss_ep_mean += byol_loss
        cluster_desc_loss_ep_mean += avg_cluster_desc_loss
        sep_desc_loss_ep_mean += avg_sep_desc_loss
        tanh_desc_loss_ep_mean += avg_tanh_desc_loss
        subspace_sep_loss_ep_mean += avg_subspace_sep_loss
        conc_log_ip_loss_ep_mean += avg_conc_log_ip_loss
        if not pretrain:
            optimizer_classifier.step()   
            scheduler_classifier.step(epoch - 1 + (i/iters))
            lrs_class.append(scheduler_classifier.get_last_lr()[0])
     
        if not finetune:
            optimizer_net.step()
            scheduler_net.step() 
            lrs_net.append(scheduler_net.get_last_lr()[0])
        else:
            lrs_net.append(0.)
            
        with torch.no_grad():
            total_acc+=acc
            total_loss+=loss.item()

        # del loss
        # To avoid all values becoming zero
        if dist_training:
            check_and_update_weights_dist(net.module, threshold=1e-3, add_value=0.01)
        else:
            check_and_update_weights(net.module, threshold=1e-3, add_value=0.01)

        if byol:
            byol_tau = byol_tau_max - (((byol_tau_max - byol_tau_base) * (torch.cos(torch.tensor((torch.pi * step_info['current_step'])/step_info['max_training_steps'])) + 1)) / 2)
            # byol_tau = 0.9985
            step_info['current_step'] += 1
            ema(byol_tau, online_network=net.module._net, target_network=net.module._target_feature_net)
            ema(byol_tau, online_network=net.module._projector, target_network=net.module._target_projector)
            # net.module._target_feature_net = (byol_tau * net.module._target_feature_net) + ((1 - byol_tau) * net.module._net)
            # net.module._target_projector = (byol_tau * net.module._target_projector) + ((1 - byol_tau) * net.module._projector)

        # if not pretrain:
        #     with torch.no_grad():
        #         for attr in dir(net.module):
        #             if attr.endswith('_classification'):
        #                 classification_layer = getattr(net.module, attr)
        #                 classification_layer.weight.copy_(torch.clamp(classification_layer.weight.data - 1e-3, min=0.)) #set weights in classification layer < 1e-3 to zero
        #                 if classification_layer.bias is not None:
        #                     classification_layer.bias.copy_(torch.clamp(classification_layer.bias.data, min=0.))  
        #         # YTIR - keeping this because this was done in the original code, but why this is required this parameter is supposed to be constant & non-trainable
        #         net.module._multiplier.copy_(torch.clamp(net.module._multiplier.data, min=1.0))
        
        _, preds_joint = net.module.get_joint_distribution(out)
        preds_joint = preds_joint[ys != OOD_LABEL]
        _, fine_predicted = torch.max(preds_joint.data, 1)
        target = ys[ys != OOD_LABEL]
        fine_correct = fine_predicted == target
        n_fine_correct += fine_correct.sum().item()
        n_samples += target.size(0)

        # del out

        # del loss
        # del features
        # del proto_features
        # del pooled
        # del out
        # del class_loss_dict
        # del a_loss
        # del tanh_loss_dict
        # del OOD_loss_dict
        # del kernel_orth_loss_dict

        # del uni_loss
        # del avg_class_loss
        # del avg_a_loss_pf
        # del avg_tanh_loss
        # del avg_OOD_loss
        # del avg_kernel_orth_loss
        # del acc
        
    class_loss_ep_mean /= float(i+1)
    tanh_loss_ep_mean /= float(i+1)
    minmaximize_loss_ep_mean /= float(i+1)
    OOD_loss_ep_mean /= float(i+1)
    kernel_orth_loss_ep_mean /= float(i+1)
    a_loss_ep_mean /= float(i+1)
    uni_loss_ep_mean /= float(i+1)
    byol_loss_ep_mean /= float(i+1)
    cluster_desc_loss_ep_mean /= float(i+1)
    sep_desc_loss_ep_mean /= float(i+1)
    tanh_desc_loss_ep_mean /= float(i+1)
    subspace_sep_loss_ep_mean /= float(i+1)
    conc_log_ip_loss_ep_mean /= float(i+1)

    train_info['fine_accuracy'] = n_fine_correct/n_samples
    train_info['train_accuracy'] = total_acc/float(i+1) # have to check what this is not sure how it is useful
    train_info['loss'] = total_loss/float(i+1)
    train_info['class_loss (mean over epoch nodes)'] = class_loss_ep_mean.item() if isinstance(class_loss_ep_mean, torch.Tensor) else class_loss_ep_mean
    train_info['kernel_orth_loss (mean over epoch nodes)'] = kernel_orth_loss_ep_mean.item() if isinstance(kernel_orth_loss_ep_mean, torch.Tensor) else kernel_orth_loss_ep_mean
    train_info['tanh_loss (mean over epoch nodes)'] = tanh_loss_ep_mean.item() if isinstance(tanh_loss_ep_mean, torch.Tensor) else tanh_loss_ep_mean
    train_info['minmaximize_loss (mean over epoch nodes)'] = minmaximize_loss_ep_mean.item() if isinstance(minmaximize_loss_ep_mean, torch.Tensor) else minmaximize_loss_ep_mean
    train_info['OOD_loss (mean over epoch nodes)'] = OOD_loss_ep_mean.item() if isinstance(OOD_loss_ep_mean, torch.Tensor) else OOD_loss_ep_mean
    train_info['a_loss (mean over epoch)'] = a_loss_ep_mean.item() if isinstance(a_loss_ep_mean, torch.Tensor) else a_loss_ep_mean
    train_info['uni_loss (mean over epoch)'] = uni_loss_ep_mean.item() if isinstance(uni_loss_ep_mean, torch.Tensor) else uni_loss_ep_mean
    train_info['true_uni_loss (without averaging)'] = true_uni_loss.item()
    train_info['lrs_net'] = lrs_net
    train_info['lrs_class'] = lrs_class

    # Logging to CSV
    # logging epoch level metrics
    try:
        log.create_log(f'epoch_wise_metrics_train', 'epoch', 'fine_accuracy', \
                        'loss', 'class_loss (mean over epoch nodes)',\
                        'kernel_orth_loss (mean over epoch nodes)',\
                        'tanh_loss (mean over epoch nodes)',\
                        'minmaximize_loss (mean over epoch nodes)',\
                        'OOD_loss (mean over epoch nodes)',\
                        'a_loss (mean over epoch)',\
                        'uni_loss (mean over epoch)',\
                        'true_uni_loss (without averaging)')
    except Exception as e:
        pass
    log.log_values(f'epoch_wise_metrics_train', epoch if pretrain else (epoch+pretrain_epochs), n_fine_correct/n_samples,
                   train_info['loss'], train_info['class_loss (mean over epoch nodes)'],
                   train_info['kernel_orth_loss (mean over epoch nodes)'],
                   train_info['tanh_loss (mean over epoch nodes)'],
                   train_info['minmaximize_loss (mean over epoch nodes)'],
                   train_info['OOD_loss (mean over epoch nodes)'],
                   train_info['a_loss (mean over epoch)'],
                   train_info['uni_loss (mean over epoch)'],
                   train_info['true_uni_loss (without averaging)'])
    
    # Loggin to WandB
    log_dict = {}
    if wandb_logging:
        log_dict[wandb_log_subdir + "/epoch loss"] = train_info['loss']
        log_dict[wandb_log_subdir + "/fine_accuracy"] = train_info['fine_accuracy']
        log_dict[wandb_log_subdir + "/class_loss"] = class_loss_ep_mean
        log_dict[wandb_log_subdir + "/tanh_loss"] = tanh_loss_ep_mean
        log_dict[wandb_log_subdir + "/minmaximize_loss"] = minmaximize_loss_ep_mean
        log_dict[wandb_log_subdir + "/OOD_loss"] = OOD_loss_ep_mean
        log_dict[wandb_log_subdir + "/kernel_orth_loss"] = kernel_orth_loss_ep_mean
        log_dict[wandb_log_subdir + "/a_loss_pf"] = a_loss_ep_mean
        log_dict[wandb_log_subdir + "/uni_loss"] = uni_loss_ep_mean
        log_dict[wandb_log_subdir + "/true_uni_loss"] = true_uni_loss.item()
        if byol:
            log_dict[wandb_log_subdir + "/byol_tau"] = byol_tau
        log_dict[wandb_log_subdir + "/byol_loss"] = byol_loss_ep_mean
        log_dict[wandb_log_subdir + "/cluster_desc_loss"] = cluster_desc_loss_ep_mean
        log_dict[wandb_log_subdir + "/sep_desc_loss"] = sep_desc_loss_ep_mean
        log_dict[wandb_log_subdir + "/tanh_desc_loss"] = tanh_desc_loss_ep_mean
        log_dict[wandb_log_subdir + "/subspace_sep_loss"] = subspace_sep_loss_ep_mean
        log_dict[wandb_log_subdir + "/conc_log_ip_loss"] = conc_log_ip_loss_ep_mean
        # wandb_run.log({wandb_log_subdir + "/epoch loss": train_info['loss']}, step=epoch)
        # wandb_run.log({wandb_log_subdir + "/epoch lrs_net": train_info['lrs_net']})
        # wandb_run.log({wandb_log_subdir + "/epoch lrs_class": train_info['lrs_class']})

    for node_name in node_accuracy:
        node_accuracy[node_name]['accuracy'] = round((node_accuracy[node_name]['n_correct'] / node_accuracy[node_name]['n_examples']) * 100, 2)
        node_accuracy[node_name]['f1'] = f1_score(node_accuracy[node_name]["preds"], node_accuracy[node_name]["gts"].to(torch.int), average='weighted', num_classes=net.module.root.get_node(node_name).num_children()).item()
        node_accuracy[node_name]['f1'] = round(node_accuracy[node_name]['f1'] * 100, 2)
        if wandb_logging:
            log_dict[wandb_log_subdir + f"/node_wise/acc:{node_name}"] = node_accuracy[node_name]['accuracy']
            log_dict[wandb_log_subdir + f"/node_wise/f1:{node_name}"] = node_accuracy[node_name]['f1']

    if wandb_logging:
        for node_name in node_wise_losses:
            for loss_name in node_wise_losses[node_name]:
                log_dict[wandb_log_subdir + f"/node_wise_{loss_name}/{node_name}"] = np.mean(node_wise_losses[node_name][loss_name])

        wandb_run.log(log_dict, step=epoch if pretrain else (epoch+pretrain_epochs))
    # wandb_run.log(log_dict, step=epoch)

    # Logging to console
    # train_info['node_accuracy'] = node_accuracy
    print('\tFine accuracy:', round(train_info['fine_accuracy'], 2))
    # for node_name in node_accuracy:
    #     acc = node_accuracy[node_name]["accuracy"]
    #     f1 = node_accuracy[node_name]['f1']
    #     samples = node_accuracy[node_name]["n_examples"]
    #     log_string = f'\tNode name: {node_name}, acc: {acc}, f1:{f1}, samples: {samples}'
    #     for child in net.module.root.get_node(node_name).children:
    #         child_n_correct = node_accuracy[node_name]['children'][child.name]['n_correct']
    #         child_n_examples = node_accuracy[node_name]['children'][child.name]['n_examples']
    #         log_string += ", " + f'{child.name}={child_n_correct}/{child_n_examples}={round(torch.divide(child_n_correct, child_n_examples).item(), 2)}'
    #     print(log_string)


    # Logging CSV
    # logging node level metrics
    # create a log csv file for each node to log the different loss values
    log_sub_dir = 'node_wise_metrics_train'
    os.makedirs(os.path.join(log.log_dir, log_sub_dir), exist_ok=True)
    for node_name in node_wise_losses:
        loss_names = sorted(list(node_wise_losses[node_name].keys()))
        try:
            log.create_log(f'{log_sub_dir}/{node_name}_losses', 'epoch', *loss_names)
        except Exception as e:
            pass

        epoch_losses = [] # contains mean over each step for each loss
        for loss_name in loss_names:
            if len(node_wise_losses[node_name][loss_name]) != 0:
                epoch_losses.append(np.mean(node_wise_losses[node_name][loss_name]))
            else:
                epoch_losses.append('n.a')
        log.log_values(f'{log_sub_dir}/{node_name}_losses', epoch if pretrain else (epoch+pretrain_epochs), *epoch_losses)

    

    return train_info, log_dict


def test_pipnet(net, test_loader, optimizer_net, optimizer_classifier, scheduler_net, scheduler_classifier, criterion, \
                epoch, nr_epochs, device, pretrain=False, finetune=False, progress_prefix: str = 'Test Epoch', wandb_logging=True, \
                test_loader_OOD=None, kernel_orth=False, tanh_desc=False, align=True, uni=True, align_pf=False, tanh=False, \
                minmaximize=False, cluster_desc=False, sep_desc=False, subspace_sep=False, byol=False, byol_tau_base=0.9995, step_info=None, \
                    wandb_run=None, pretrain_epochs=0, log:Log=None, args=None, apply_overspecificity_mask=False, leave_out_classes=None, \
                        path_prob_softmax_tau=1):

    root = net.module.root
    dataset = test_loader.dataset
    while type(dataset) != ImageFolder:
        dataset = dataset.dataset
    name2label = dataset.class_to_idx
    label2name = {label:name for name, label in name2label.items()}
    label2name[OOD_LABEL] = 'OOD'

    wandb_log_subdir = 'test'

    # node_accuracy = defaultdict(lambda: {'n_examples': 0, 'n_correct': 0, 'accuracy': None, 'preds': None, 'children': defaultdict(lambda: {'n_examples': 0, 'n_correct': 0})})
    node_accuracy = {}
    for node in root.nodes_with_children():
        node_accuracy[node.name] = {'n_examples': 0, 'n_correct': 0, 'accuracy': None, 'f1': None, 'preds': torch.empty(0, node.num_children()).cpu(), 'gts': torch.empty(0).cpu()}
        node_accuracy[node.name]['children'] = defaultdict(lambda: {'n_examples': 0, 'n_correct': 0})

    # Make sure the model is in eval mode
    net.eval()
    
    # Store info about the procedure
    test_info = dict()
    total_loss = 0.
    total_acc = 0.
    class_loss_ep_mean = 0.
    a_loss_pf_ep_mean = 0.
    a_loss_ep_mean = 0.
    tanh_loss_ep_mean = 0.
    minmaximize_loss_ep_mean = 0.
    OOD_loss_ep_mean = 0.
    kernel_orth_loss_ep_mean = 0.
    uni_loss_ep_mean = 0.
    byol_loss_ep_mean = 0.
    cluster_desc_loss_ep_mean = 0.
    sep_desc_loss_ep_mean = 0.
    tanh_desc_loss_ep_mean = 0.
    subspace_sep_loss_ep_mean = 0.
    conc_log_ip_loss_ep_mean = 0.

    iters = len(test_loader)
    # Show progress on progress bar. 
    test_iter = tqdm(enumerate(test_loader),
                    total=len(test_loader),
                    desc=progress_prefix+'%s'%epoch,
                    mininterval=2.,
                    ncols=0)
    
    test_OOD_iter = iter(test_loader_OOD) if test_loader_OOD else None
    OOD_loss_required = True if test_loader_OOD else False

    if pretrain:
        align_pf_weight = (epoch/nr_epochs)*1.
        # unif_weight = 0.5
        byol_weight = 0.5
        align_weight = 3.
        unif_weight = 3.
        t_weight = 5.
        mm_weight = 0.
        cl_weight = 0.
        # optional losses
        OOD_loss_weight = 0.
        orth_weight = 0.1
        cluster_desc_weight = 0.8
        sep_desc_weight = 0.08
        subspace_sep_weight = 1e-7
    else:
        align_pf_weight = 5. 
        # unif_weight = 2. # 0.
        byol_weight = 0.5
        align_weight = 3. 
        unif_weight = 3. # 0.
        t_weight = 2.
        mm_weight = 2.
        cl_weight = args.cl_weight # 2.
        # optional losses
        OOD_loss_weight = 0.2
        orth_weight = 0.1
        cluster_desc_weight = 0.8
        sep_desc_weight = 0.08
        subspace_sep_weight = 1e-7

    # maps, node_name -> loss_name -> list_of_loss_values_corresponding_to_each_step
    # node_wise_losses = defaultdict(lambda: defaultdict(list))
    node_wise_losses = {}
    for node in root.nodes_with_children():
        node_wise_losses[node.name] = {}
        node_wise_losses[node.name]['class_loss'] = []
        node_wise_losses[node.name]['a_loss'] = []
        node_wise_losses[node.name]['tanh_loss'] = []
        node_wise_losses[node.name]['minmaximize_loss'] = []
        node_wise_losses[node.name]['OOD_loss'] = []
        node_wise_losses[node.name]['kernel_orth_loss'] = []
        node_wise_losses[node.name]['uni_loss'] = []

    
    lrs_net = []
    lrs_class = []
    n_fine_correct = 0
    n_samples = 0
    
    with torch.no_grad():
        # Iterate through the data set to update leaves, prototypes and network
        for i, (xs, ys) in test_iter:       
            
            xs, ys = xs.to(device), ys.to(device)

            if test_OOD_iter:
                xs_OOD, ys_OOD = next(test_OOD_iter)
                ys_OOD.fill_(OOD_LABEL)
                xs_OOD, ys_OOD = xs_OOD.to(device), ys_OOD.to(device)
                xs = torch.cat([xs, xs, xs_OOD])
                ys = torch.cat([ys, ys, ys_OOD])
            else:
                xs = torch.cat([xs, xs])
                ys = torch.cat([ys, ys])

            additional_network_outputs = {}
            
            # Perform a forward pass through the network
            if byol:
                online_network_out, target_network_out, features, proto_features, pooled, out = net(xs)
                additional_network_outputs['online_network_out'] = online_network_out
                additional_network_outputs['target_network_out'] = target_network_out
            else:
                features, proto_features, pooled, out = net(xs, apply_overspecificity_mask=apply_overspecificity_mask)

            loss, class_loss_dict, a_loss, tanh_loss_dict, minmaximize_loss_dict, OOD_loss_dict, kernel_orth_loss_dict, \
            uni_loss, avg_class_loss, avg_a_loss_pf, avg_tanh_loss, avg_minmaximize_loss, avg_OOD_loss, avg_kernel_orth_loss, \
            byol_loss, avg_cluster_desc_loss, avg_sep_desc_loss, avg_tanh_desc_loss, avg_subspace_sep_loss, avg_conc_log_ip_loss, acc = \
                calculate_loss(epoch, net, additional_network_outputs, features, proto_features, pooled, out, ys, align_weight=align_weight, align_pf_weight=align_pf_weight, \
                                t_weight=t_weight, mm_weight=mm_weight, unif_weight=unif_weight, cl_weight=cl_weight, OOD_loss_weight=OOD_loss_weight, orth_weight=orth_weight, \
                                cluster_desc_weight=cluster_desc_weight, sep_desc_weight=sep_desc_weight, subspace_sep_weight=subspace_sep_weight, byol_weight=byol_weight, \
                                net_normalization_multiplier=net.module._multiplier, pretrain=pretrain, finetune=finetune, \
                            criterion=criterion, train_iter=test_iter, print=True, EPS=1e-8, root=root, label2name=label2name, node_accuracy=node_accuracy, \
                            OOD_loss_required=OOD_loss_required, kernel_orth=kernel_orth, tanh_desc=tanh_desc, align=align, uni=uni, align_pf=align_pf, tanh=tanh, minmaximize=minmaximize,\
                            cluster_desc=cluster_desc, sep_desc=sep_desc, subspace_sep=subspace_sep, byol=byol, train=False, args=args, device=device)
            
            # print(f"GPU Memory Usage: 0:{torch.cuda.memory_allocated(0) / 1024**2:.2f} MB, 1:{torch.cuda.memory_allocated(1) / 1024**2:.2f} MB")

            for node_name, loss_value in class_loss_dict.items():
                node_wise_losses[node_name]['class_loss'].append(loss_value.item())
            
            # for node_name, loss_value in a_loss_pf_dict.items():
            #     node_wise_losses[node_name]['a_loss'].append(loss_value.item())

            for node_name, loss_value in minmaximize_loss_dict.items():
                node_wise_losses[node_name]['minmaximize_loss'].append(loss_value.item())

            for node_name, loss_value in tanh_loss_dict.items():
                node_wise_losses[node_name]['tanh_loss'].append(loss_value.item())

            for node_name, loss_value in OOD_loss_dict.items():
                node_wise_losses[node_name]['OOD_loss'].append(loss_value.item())

            for node_name, loss_value in kernel_orth_loss_dict.items():
                node_wise_losses[node_name]['kernel_orth_loss'].append(loss_value.item())

            # for node_name, loss_value in uni_loss_dict.items():
            #     node_wise_losses[node_name]['uni_loss'].append(loss_value.item())

            class_loss_ep_mean += avg_class_loss if avg_class_loss else -5
            tanh_loss_ep_mean += avg_tanh_loss if avg_tanh_loss else -5
            minmaximize_loss_ep_mean += avg_minmaximize_loss if avg_minmaximize_loss else -5
            OOD_loss_ep_mean += avg_OOD_loss if avg_OOD_loss else -5
            kernel_orth_loss_ep_mean += avg_kernel_orth_loss if avg_kernel_orth_loss else -5
            
            # modifying the these two because they are not calculated for each node anymore
            a_loss_ep_mean += a_loss # avg_a_loss_pf if avg_a_loss_pf else -5
            uni_loss_ep_mean += uni_loss # avg_uni_loss if avg_uni_loss else -5

            byol_loss_ep_mean += byol_loss
            cluster_desc_loss_ep_mean += avg_cluster_desc_loss
            sep_desc_loss_ep_mean += avg_sep_desc_loss
            tanh_desc_loss_ep_mean += avg_tanh_desc_loss
            subspace_sep_loss_ep_mean += avg_subspace_sep_loss
            conc_log_ip_loss_ep_mean += avg_conc_log_ip_loss
                
            total_acc+=acc # DUMMY can be removed
            total_loss+=loss.item()

            # print(ys)
            
            _, preds_joint = net.module.get_joint_distribution(out, leave_out_classes=leave_out_classes, apply_overspecificity_mask=apply_overspecificity_mask, softmax_tau=path_prob_softmax_tau)
            preds_joint = preds_joint[ys != OOD_LABEL]
            _, fine_predicted = torch.max(preds_joint.data, 1)
            target = ys[ys != OOD_LABEL]
            fine_correct = fine_predicted == target
            n_fine_correct += fine_correct.sum().item()
            n_samples += target.size(0)

            # print(n_fine_correct / n_samples)
            # print(preds_joint.shape)
            # pdb.set_trace()

    class_loss_ep_mean /= float(i+1)
    a_loss_pf_ep_mean /= float(i+1)
    tanh_loss_ep_mean /= float(i+1)
    minmaximize_loss_ep_mean /= float(i+1)
    OOD_loss_ep_mean /= float(i+1)
    kernel_orth_loss_ep_mean /= float(i+1)
    uni_loss_ep_mean /= float(i+1)
    byol_loss_ep_mean /= float(i+1)
    cluster_desc_loss_ep_mean /= float(i+1)
    sep_desc_loss_ep_mean /= float(i+1)
    tanh_desc_loss_ep_mean /= float(i+1)
    subspace_sep_loss_ep_mean /= float(i+1)
    conc_log_ip_loss_ep_mean /= float(i+1)


    test_info['fine_accuracy'] = n_fine_correct/n_samples
    test_info['accuracy'] = total_acc/float(i+1)
    test_info['loss'] = total_loss/float(i+1)
    test_info['class_loss (mean over epoch nodes)'] = class_loss_ep_mean.item() if isinstance(class_loss_ep_mean, torch.Tensor) else class_loss_ep_mean
    test_info['kernel_orth_loss (mean over epoch nodes)'] = kernel_orth_loss_ep_mean.item() if isinstance(kernel_orth_loss_ep_mean, torch.Tensor) else kernel_orth_loss_ep_mean
    test_info['tanh_loss (mean over epoch nodes)'] = tanh_loss_ep_mean.item() if isinstance(tanh_loss_ep_mean, torch.Tensor) else tanh_loss_ep_mean
    test_info['minmaximize_loss (mean over epoch nodes)'] = minmaximize_loss_ep_mean.item() if isinstance(minmaximize_loss_ep_mean, torch.Tensor) else minmaximize_loss_ep_mean
    test_info['OOD_loss (mean over epoch nodes)'] = OOD_loss_ep_mean.item() if isinstance(OOD_loss_ep_mean, torch.Tensor) else OOD_loss_ep_mean
    test_info['a_loss (mean over epoch)'] = a_loss_ep_mean.item() if isinstance(a_loss_ep_mean, torch.Tensor) else a_loss_ep_mean
    test_info['uni_loss (mean over epoch)'] = uni_loss_ep_mean.item() if isinstance(uni_loss_ep_mean, torch.Tensor) else uni_loss_ep_mean

    # Logging to CSV
    # logging epoch level metrics
    try:
        log.create_log(f'epoch_wise_metrics_test', 'epoch', 'fine_accuracy', \
                        'loss', 'class_loss (mean over epoch nodes)',\
                        'kernel_orth_loss (mean over epoch nodes)',\
                        'tanh_loss (mean over epoch nodes)',\
                        'minmaximize_loss (mean over epoch nodes)',\
                        'OOD_loss (mean over epoch nodes)',\
                        'a_loss (mean over epoch)',\
                        'uni_loss (mean over epoch)')
    except Exception as e:
        pass
    log.log_values(f'epoch_wise_metrics_test', epoch if pretrain else (epoch+pretrain_epochs), n_fine_correct/n_samples,
                   test_info['loss'], test_info['class_loss (mean over epoch nodes)'],
                   test_info['kernel_orth_loss (mean over epoch nodes)'],
                   test_info['tanh_loss (mean over epoch nodes)'],
                   test_info['minmaximize_loss (mean over epoch nodes)'],
                   test_info['OOD_loss (mean over epoch nodes)'],
                   test_info['a_loss (mean over epoch)'],
                   test_info['uni_loss (mean over epoch)'])

    log_dict = {}
    if wandb_logging:
        log_dict[wandb_log_subdir + "/epoch loss"] = test_info['loss']
        log_dict[wandb_log_subdir + "/fine_accuracy"] = test_info['fine_accuracy']
        log_dict[wandb_log_subdir + "/class_loss"] = class_loss_ep_mean
        log_dict[wandb_log_subdir + "/tanh_loss"] = tanh_loss_ep_mean
        log_dict[wandb_log_subdir + "/minmaximize_loss"] = minmaximize_loss_ep_mean
        log_dict[wandb_log_subdir + "/OOD_loss"] = OOD_loss_ep_mean
        log_dict[wandb_log_subdir + "/kernel_orth_loss"] = kernel_orth_loss_ep_mean
        log_dict[wandb_log_subdir + "/a_loss_pf"] = a_loss_ep_mean
        log_dict[wandb_log_subdir + "/uni_loss"] = uni_loss_ep_mean
        log_dict[wandb_log_subdir + "/byol_loss"] = byol_loss_ep_mean
        log_dict[wandb_log_subdir + "/cluster_desc_loss"] = cluster_desc_loss_ep_mean
        log_dict[wandb_log_subdir + "/sep_desc_loss"] = sep_desc_loss_ep_mean
        log_dict[wandb_log_subdir + "/tanh_desc_loss"] = tanh_desc_loss_ep_mean
        log_dict[wandb_log_subdir + "/subspace_sep_loss"] = subspace_sep_loss_ep_mean
        log_dict[wandb_log_subdir + "/conc_log_ip_loss"] = conc_log_ip_loss_ep_mean
    # log_dict[wandb_log_subdir + "/uni_loss"] = uni_loss_ep_mean
    # wandb_run.log({wandb_log_subdir + "/epoch loss": train_info['loss']}, step=epoch)
    # wandb_run.log({wandb_log_subdir + "/epoch lrs_net": train_info['lrs_net']})
    # wandb_run.log({wandb_log_subdir + "/epoch lrs_class": train_info['lrs_class']})

    for node_name in node_accuracy:
        if node_accuracy[node_name]['n_examples'] == 0:
            node_accuracy[node_name]['accuracy'] = node_accuracy[node_name]['f1'] = float('inf')
            log_dict[wandb_log_subdir + f"/node_wise/acc:{node_name}"] = node_accuracy[node_name]['accuracy']
            log_dict[wandb_log_subdir + f"/node_wise/f1:{node_name}"] = node_accuracy[node_name]['f1']
            continue
        node_accuracy[node_name]['accuracy'] = round((node_accuracy[node_name]['n_correct'] / node_accuracy[node_name]['n_examples']) * 100, 2)
        node_accuracy[node_name]['f1'] = f1_score(node_accuracy[node_name]["preds"], node_accuracy[node_name]["gts"].to(torch.int), \
                                                    average='weighted', num_classes=net.module.root.get_node(node_name).num_children()).item()
        node_accuracy[node_name]['f1'] = round(node_accuracy[node_name]['f1'] * 100, 2)
        # if wandb_logging:
        log_dict[wandb_log_subdir + f"/node_wise/acc:{node_name}"] = node_accuracy[node_name]['accuracy']
        log_dict[wandb_log_subdir + f"/node_wise/f1:{node_name}"] = node_accuracy[node_name]['f1']
    if wandb_logging:
        for node_name in node_wise_losses:
            for loss_name in node_wise_losses[node_name]:
                log_dict[wandb_log_subdir + f"/node_wise_{loss_name}/{node_name}"] = np.mean(node_wise_losses[node_name][loss_name])

        wandb_run.log(log_dict, step=epoch if pretrain else (epoch+pretrain_epochs))

    test_info['node_accuracy'] = node_accuracy
    print('\tFine accuracy:', round(test_info['fine_accuracy'], 4))
    for node_name in node_accuracy:
        acc = node_accuracy[node_name]["accuracy"]
        f1 = node_accuracy[node_name]['f1']
        samples = node_accuracy[node_name]["n_examples"]
        if int(samples) == 0:
            continue
        log_string = f'\tNode name: {node_name}, acc: {acc}, f1:{f1}, samples: {samples}'
        for child in net.module.root.get_node(node_name).children:
            child_n_correct = node_accuracy[node_name]['children'][child.name]['n_correct']
            child_n_examples = node_accuracy[node_name]['children'][child.name]['n_examples']
            log_string += ", " + f'{child.name}={child_n_correct}/{child_n_examples}={round(torch.divide(child_n_correct, child_n_examples).item(), 2)}'
        print(log_string)

    # create a log csv file for each node to log the different loss values
    if log:
        log_sub_dir = 'node_wise_metrics_val'
        os.makedirs(os.path.join(log.log_dir, log_sub_dir), exist_ok=True)
        for node_name in node_wise_losses:
            loss_names = sorted(list(node_wise_losses[node_name].keys()))
            try:
                log.create_log(f'{log_sub_dir}/{node_name}_losses', 'epoch', *loss_names)
            except Exception as e:
                pass

            epoch_losses = [] # contains mean over each step for each loss
            for loss_name in loss_names:
                if len(node_wise_losses[node_name][loss_name]) != 0:
                    epoch_losses.append(np.mean(node_wise_losses[node_name][loss_name]))
                else:
                    epoch_losses.append('n.a')
            log.log_values(f'{log_sub_dir}/{node_name}_losses', epoch if pretrain else (epoch+pretrain_epochs), *epoch_losses)
    
    return test_info, log_dict


def calculate_loss(epoch, net, additional_network_outputs, features, proto_features, pooled, out, ys, align_weight, align_pf_weight, t_weight, mm_weight, unif_weight, cl_weight, OOD_loss_weight, \
                    orth_weight, cluster_desc_weight, sep_desc_weight, subspace_sep_weight, byol_weight, net_normalization_multiplier, pretrain, finetune, criterion, train_iter, print=True, EPS=1e-10, root=None, \
                    label2name=None, node_accuracy=None, OOD_loss_required=False, kernel_orth=False, tanh_desc=False, align=True, uni=True, \
                        align_pf=False, tanh=False, minmaximize=False, cluster_desc=False, sep_desc=False, subspace_sep=False, byol=False, train=True, args=None, device=None):
    batch_names = [label2name[y.item()] for y in ys]

    normalize_by_node_count = True

    al_and_uni = 0
    cl_and_tanh_desc = 0
    mm_loss = 0
    loss = 0
    class_loss = {}
    a_loss_pf = {}
    tanh_loss = {}
    OOD_loss = {}
    kernel_orth_loss = {}
    uni_loss = {}
    tanh_desc_loss = {}
    minmaximize_loss = {}
    cluster_desc_loss = {}
    subspace_sep_loss = {}
    sep_desc_loss = {}
    conc_log_ip_loss = {}
    ant_conc_log_ip_loss = {}
    act_l1_loss = {}
    overspecifity_loss = {}
    mask_l1_loss = {}
    minimize_contrasting_set_loss = {}
    OOD_ent_loss = {}

    losses_used = []

    features1, features2 = features.chunk(2)

    if (not finetune) and byol:
        online_network_out = additional_network_outputs['online_network_out']
        target_network_out = additional_network_outputs['target_network_out']
        online_network_out_1, online_network_out_2 = online_network_out.chunk(2)
        target_network_out_1, target_network_out_2 = target_network_out.chunk(2)
        byol_loss = (regression_loss(online_network_out_1, target_network_out_2) + regression_loss(online_network_out_2, target_network_out_1)) / 2.
        loss += byol_weight * byol_loss
        losses_used.append('BYOL')
    else:
        byol_loss = torch.tensor(-5) # placeholder value

    if (not finetune) and align and uni:
        flattened_features1 = flatten_tensor(features1)
        flattened_features2 = flatten_tensor(features2)
        normalized_flattened_features1 = F.normalize(flattened_features1, p=2, dim=1)
        normalized_flattened_features2 = F.normalize(flattened_features2, p=2, dim=1)
        a_loss = align_loss_unit_space(normalized_flattened_features1, normalized_flattened_features2)

        uni_loss = (uniform_loss(normalized_flattened_features1) \
                    + uniform_loss(normalized_flattened_features2)) / 2.


        # loss += align_pf_weight * a_loss
        al_and_uni += align_weight * a_loss
        losses_used.append('AL')
        # loss += unif_weight * uni_loss
        al_and_uni += unif_weight * uni_loss
        losses_used.append('UNI')
    elif (not finetune) and align:
        flattened_features1 = flatten_tensor(features1)
        flattened_features2 = flatten_tensor(features2)
        normalized_flattened_features1 = F.normalize(flattened_features1, p=2, dim=1)
        normalized_flattened_features2 = F.normalize(flattened_features2, p=2, dim=1)
        a_loss = align_loss_unit_space(normalized_flattened_features1, normalized_flattened_features2)
        al_and_uni += align_weight * a_loss
        losses_used.append('AL')
    elif (not finetune) and uni:
        raise Exception('Uni can be used only along with align loss')

    else:
        a_loss = torch.tensor(-5) # placeholder value
        uni_loss = torch.tensor(-5) # placeholder value                    

    if args.sg_before_protos == 'y':
        features = features.clone().detach()

    for node in root.nodes_with_children():
        children_idx = torch.tensor([name in node.leaf_descendents for name in batch_names])
        batch_names_coarsest = [node.closest_descendent_for(name).name for name in batch_names if name in node.leaf_descendents]
        try:
            node_y = torch.tensor([node.children_to_labels[name] for name in batch_names_coarsest]).to(device)#.cuda()
        except:
            pdb.set_trace()

        if len(node_y) == 0:
            continue

        node_logits = out[node.name][children_idx]

        if (not pretrain) and ('y' in args.mask_prune_overspecific):
            assert args.protopool != 'y'

            overspecifity_loss_weight = 2.0 # 1.0 # 2.0
            mask_l1_loss_weight = 0.5

            if (len(args.mask_prune_overspecific.split('|')) > 1) and (epoch < int(args.mask_prune_overspecific.split('|')[1])):
                pass
            else:
                proto_presence = getattr(net.module, '_'+node.name+'_proto_presence')
                overspecifity_loss_current_node = 0.
                mask_l1_loss_current_node = 0.
                total_num_relevant_protos = 0. # sum of relevant protos for each class, no proto is relevant to more than one class so this would work

                for child_node in node.children:
                    classification_weights = getattr(net.module, '_'+node.name+'_classification').weight
                    child_class_idx = node.children_to_labels[child_node.name]
                    relevant_proto_idx = torch.nonzero(classification_weights[child_class_idx, :] > 1e-3).squeeze(-1)
                    num_relevant_protos = relevant_proto_idx.shape[0]
                    total_num_relevant_protos += num_relevant_protos

                    max_pooled_each_descendant = torch.empty(0, num_relevant_protos).to(device)#.cuda()
                    for descendant_name in child_node.leaf_descendents:
                        descendant_idx = torch.tensor([name == descendant_name for name in batch_names])
                        if torch.sum(descendant_idx).item() == 0: # no of descendants in batch is zero
                            continue
                        max_vals, _ = torch.max(pooled[node.name][descendant_idx][:, relevant_proto_idx], dim=0, keepdim=True) # [1, num_relevant_protos]
                        max_pooled_each_descendant = torch.cat([max_pooled_each_descendant, max_vals], dim=0)

                    if max_pooled_each_descendant.shape[0] == 0: # if none of the descendants are in the batch
                        continue

                    proto_presence = F.gumbel_softmax(proto_presence, tau=0.5, hard=False, dim=-1)

                    if (len(args.mask_prune_overspecific.split('|')) > 2): # is there is a boosting factor
                        boosting_factor = float(args.mask_prune_overspecific.split('|')[2])
                        if ('sg_before_masking' in args) and ('y' in args.sg_before_masking):
                            overspecifity_loss_current_node += (-1) * (torch.prod(torch.clamp(max_pooled_each_descendant.clone().detach() * boosting_factor, max=1.0), dim=0) * proto_presence[relevant_proto_idx, 1]).sum()
                        else:
                            overspecifity_loss_current_node += (-1) * (torch.prod(torch.clamp(max_pooled_each_descendant * boosting_factor, max=1.0), dim=0) * proto_presence[relevant_proto_idx, 1]).sum()

                    else: # without boosting factor
                        num_descendants_in_batch = max_pooled_each_descendant.shape[0]
                        if ('geometric_mean_overspecificity_score' in args) and ('y' in args.geometric_mean_overspecificity_score):
                            overspecificity_score = torch.prod(max_pooled_each_descendant.pow(1/num_descendants_in_batch), dim=0)
                        else:
                            overspecificity_score = torch.prod(max_pooled_each_descendant, dim=0)
                        if ('sg_before_masking' in args) and ('y' in args.sg_before_masking):
                            overspecificity_score = overspecificity_score.clone().detach()
                        overspecifity_loss_current_node += (-1) * (overspecificity_score * proto_presence[relevant_proto_idx, 1]).sum()

                        # if ('sg_before_masking' in args) and ('y' in args.sg_before_masking):
                        #     overspecifity_loss_current_node += (-1) * (torch.prod(max_pooled_each_descendant, dim=0).clone().detach() * proto_presence[relevant_proto_idx, 1]).sum()
                        # else:
                        #     overspecifity_loss_current_node += (-1) * (torch.prod(max_pooled_each_descendant, dim=0) * proto_presence[relevant_proto_idx, 1]).sum()
                    mask_l1_loss_current_node += proto_presence[relevant_proto_idx, 1].sum()


                overspecifity_loss_current_node /= total_num_relevant_protos
                mask_l1_loss_current_node /= total_num_relevant_protos

                overspecifity_loss[node.name] = (overspecifity_loss_weight * overspecifity_loss_current_node) \
                                                    / (len(root.nodes_with_children()) if normalize_by_node_count else 1.)

                mask_l1_loss[node.name] = (mask_l1_loss_weight * mask_l1_loss_current_node) \
                                                    / (len(root.nodes_with_children()) if normalize_by_node_count else 1.)

                loss += overspecifity_loss[node.name] + mask_l1_loss[node.name]
                if not 'MASK_PRUNING' in losses_used:
                    losses_used.append('MASK_PRUNING')

        if (not pretrain) and (not finetune) and ('y' in args.minimize_contrasting_set):
            if (len(args.minimize_contrasting_set.split('|')) > 2):
                minimize_contrasting_set_weight = float(args.minimize_contrasting_set.split('|')[2])
            else:
                minimize_contrasting_set_weight = 0.1

            TOPK = int(args.minimize_contrasting_set.split('|')[1]) if (len(args.minimize_contrasting_set.split('|')) > 1) else 1
            EPS=1e-12
            num_protos = pooled[node.name].shape[-1]
            classification_weights = getattr(net.module, '_'+node.name+'_classification').weight
            if args.protopool == 'n':
                node_y_expanded = node_y.unsqueeze(-1).repeat(1, num_protos) # [batch_size] to [batch_size, num_protos]
                minimize_contrasting_set_loss[node.name] = 0.
                max_of_contrasting_set = torch.empty(TOPK, 0).to(device)#.cuda()
                for child_node in node.children:
                    child_class_idx = node.children_to_labels[child_node.name]
                    relevant_proto_idx = torch.nonzero(classification_weights[child_class_idx, :] > 1e-5).squeeze(-1)

                    if len(relevant_proto_idx) == 0:
                        no_proto_node_names = [child_name for node in root.nodes_with_children() for child_name, child_class_idx in node.children_to_labels.items() if (len(torch.nonzero(getattr(net.module, '_'+node.name+'_classification').weight[child_class_idx, :] > 1e-5).squeeze(-1)) == 0)]
                        # likely to happen when leave_out_classes is used
                        # breakpoint()
                        print('Node', node.name, 'child_node', child_node.name)
                        assert child_node.is_leaf()
                        assert args.leave_out_classes.strip() != '' 
                        continue
                    
                    # look at data points that do not belong to child_node
                    relevant_data_idx = torch.nonzero(node_y != child_class_idx).squeeze(-1)

                    if len(relevant_data_idx) == 0:
                        continue # no data points that is not equal to child_class_idx present so skip this child_node

                    max_of_contrast, topk_idx = torch.topk(pooled[node.name][children_idx][relevant_data_idx, :][:, relevant_proto_idx], dim=0, k=TOPK)
                    max_of_contrasting_set = torch.cat([max_of_contrasting_set, max_of_contrast], dim=1)

                    # if torch.isnan(max_of_contrast.sum()):
                    #     breakpoint()
                if max_of_contrasting_set.numel() != 0:
                    loss += minimize_contrasting_set_weight * max_of_contrasting_set.mean() / (len(root.nodes_with_children()) if normalize_by_node_count else 1.)
                    if not 'MIN_CONT' in losses_used:
                        losses_used.append('MIN_CONT')
            else:
                raise Exception('Do not use ant_conc_log_ip loss when protopool is true')


        if (not finetune) and align_pf:
            # CARL align loss
            pf1, pf2 = proto_features[node.name][children_idx].chunk(2)
            embv2 = pf2.flatten(start_dim=2).permute(0,2,1).flatten(end_dim=1)
            embv1 = pf1.flatten(start_dim=2).permute(0,2,1).flatten(end_dim=1)
            a_loss_pf[node.name] = (align_loss(embv1, embv2.detach()) \
                                    + align_loss(embv2, embv1.detach())) / 2.
            # if torch.isnan(a_loss_pf[node.name]):
            #     breakpoint()
            loss += align_pf_weight * a_loss_pf[node.name] / (len(root.nodes_with_children()) if normalize_by_node_count else 1.)
            if not 'AL_PF' in losses_used:
                losses_used.append('AL_PF')

        if (not finetune) and tanh:
            if (args.tanh_during_second_phase == 'n') and (not pretrain):
                pass
            else:
                pooled1, pooled2 = pooled[node.name][children_idx].chunk(2)
                tanh_loss[node.name] = -(torch.log(torch.tanh(torch.sum(pooled1,dim=0))+EPS).mean() \
                                        + torch.log(torch.tanh(torch.sum(pooled2,dim=0))+EPS).mean()) / 2.
                # if torch.isnan(tanh_loss[node.name]):
                #     breakpoint()
                loss += t_weight * tanh_loss[node.name] / (len(root.nodes_with_children()) if normalize_by_node_count else 1.)
                if not 'TANH' in losses_used:
                    losses_used.append('TANH')

        if (not finetune) and (not pretrain) and tanh_desc:
            tanh_desc_weight =  float(args.tanh_desc.split('|')[1])
            # tanh_desc_weight =  0.05 # 0.1 # 0.2 # 2.0
            # tanh loss corresponding to every descendant species
            tanh_for_each_descendant = []
            for child_node in node.children:
                child_class_idx = node.children_to_labels[child_node.name]
                classification_weights = getattr(net.module, '_'+node.name+'_classification').weight
                if child_node.is_leaf(): # because leaf nodes do not have any descendants
                    descendant_idx = torch.tensor([name == child_node.name for name in batch_names])
                    relevant_proto_idx = torch.nonzero(classification_weights[child_class_idx, :] > 1e-3).squeeze(-1)
                    if len(relevant_proto_idx) == 0:
                        no_proto_node_names = [child_name for node in root.nodes_with_children() for child_name, child_class_idx in node.children_to_labels.items() if (len(torch.nonzero(getattr(net.module, '_'+node.name+'_classification').weight[child_class_idx, :] > 1e-5).squeeze(-1)) == 0)]
                        # likely to happen when leave_out_classes is used
                        # breakpoint()
                        # pdb.set_trace()
                        print('Node', node.name, 'child_node', child_node.name)
                        assert args.leave_out_classes.strip() != '' # this is likely to happen when using leave_out_classes
                        continue 
                    descendant_pooled1, descendant_pooled2 = pooled[node.name][descendant_idx][:, relevant_proto_idx].chunk(2)
                    descendant_tanh_loss = -(torch.log(torch.tanh(torch.sum(descendant_pooled1,dim=0))+EPS).mean() \
                                                        + torch.log(torch.tanh(torch.sum(descendant_pooled2,dim=0))+EPS).mean()) / 2.
                    tanh_for_each_descendant.append(descendant_tanh_loss)
                else:
                    for descendant_name in child_node.leaf_descendents:
                        descendant_idx = torch.tensor([name == descendant_name for name in batch_names])
                        relevant_proto_idx = torch.nonzero(classification_weights[child_class_idx, :] > 1e-3).squeeze(-1)
                        if len(relevant_proto_idx) == 0:
                            print('Node', node.name, 'child_node', child_node.name)
                            raise Exception('All values become zero' + 'Node ' + node.name + 'child_node ' + child_node.name)
                            # breakpoint()
                        descendant_pooled1, descendant_pooled2 = pooled[node.name][descendant_idx][:, relevant_proto_idx].chunk(2)
                        descendant_tanh_loss = -(torch.log(torch.tanh(torch.sum(descendant_pooled1,dim=0))+EPS).mean() \
                                                            + torch.log(torch.tanh(torch.sum(descendant_pooled2,dim=0))+EPS).mean()) / 2.
                        tanh_for_each_descendant.append(descendant_tanh_loss)
            tanh_desc_loss[node.name] = torch.mean(torch.stack(tanh_for_each_descendant), dim=0)

            if torch.isnan(tanh_desc_loss[node.name]):
                raise Exception('Tanh desc became nan')
                # breakpoint()

            # loss += t_weight * tanh_loss[node.name]
            cl_and_tanh_desc += tanh_desc_weight * tanh_desc_loss[node.name] / (len(root.nodes_with_children()) if normalize_by_node_count else 1.)
            if not 'TANH_DESC' in losses_used:
                losses_used.append('TANH_DESC')

        # may not be required
        if (not pretrain) and (not finetune) and kernel_orth:
            prototype_kernels = getattr(net.module, '_'+node.name+'_add_on')
            classification_layer = getattr(net.module, '_'+node.name+'_classification')
            # using any below because its a relevant prototype if it has strong connection to any one of the class
            relevant_prototype_kernels = prototype_kernels.weight[(classification_layer.weight > 0.001).any(dim=0)]
            try:
                kernel_orth_loss[node.name] = orth_dist(relevant_prototype_kernels, device=device)
            except:
                kernel_orth_loss[node.name] = 0.
                # breakpoint()

            # if torch.isnan(kernel_orth_loss[node.name]):
            #     breakpoint()
            loss += orth_weight * kernel_orth_loss[node.name] / (len(root.nodes_with_children()) if normalize_by_node_count else 1.)
            if not 'KO' in losses_used:
                losses_used.append('KO')

        if not pretrain:
            # finetuning or general training
            if ('pipnet_sparsity' in args) and (args.pipnet_sparsity == 'n'):
                softmax_inputs = node_logits
            else:
                softmax_inputs = torch.log1p(node_logits**net_normalization_multiplier)
            # softmax_tau = 0.2
            # softmax_inputs = softmax_inputs / softmax_tau
            class_loss[node.name] = criterion(softmax_inputs, \
                                                node_y, \
                                                node.weights) # * (len(node_y) / len(ys[ys != OOD_LABEL]))

            cl_and_tanh_desc += cl_weight * class_loss[node.name] / (len(root.nodes_with_children()) if normalize_by_node_count else 1.)
            if not 'CL' in losses_used:
                losses_used.append('CL')

            # may not be required
            if OOD_loss_required:
                not_children_idx = torch.tensor([name not in node.leaf_descendents for name in batch_names]) # includes OOD images as well as images belonging to other nodes
                OOD_logits = out[node.name][not_children_idx] # [sum(not_children_idx), node.num_children()]
                sigmoid_out = F.sigmoid(torch.log1p(OOD_logits**net_normalization_multiplier))
                OOD_loss[node.name] = F.binary_cross_entropy(sigmoid_out, torch.zeros_like(OOD_logits))
                # loss += OOD_loss_weight * OOD_loss[node.name]
                cl_and_tanh_desc += OOD_loss_weight * OOD_loss[node.name] / (len(root.nodes_with_children()) if normalize_by_node_count else 1.)
                if not 'OOD' in losses_used:
                    losses_used.append('OOD')

                
        # Our tanh-loss optimizes for uniformity and was sufficient for our experiments. However, if pretraining of the prototypes is not working well for your dataset, you may try to add another uniformity loss from https://www.tongzhouwang.info/hypersphere/ Just uncomment the following three lines
        # else:
        #     uni_loss[node.name] = (uniform_loss(F.normalize(pooled1+EPS,dim=1)) + uniform_loss(F.normalize(pooled2+EPS,dim=1)))/2.
        #     loss += unif_weight * uni_loss[node.name]

        # For debugging purpose
        node_accuracy[node.name]['n_examples'] += node_y.shape[0]
        _, node_coarsest_predicted = torch.max(node_logits.data, 1)
        node_accuracy[node.name]['n_correct'] += (node_y == node_coarsest_predicted).sum().item()
        for child in node.children:
            node_accuracy[node.name]['children'][child.name]['n_examples'] += (node_y == node.children_to_labels[child.name]).sum().item()
            node_accuracy[node.name]['children'][child.name]['n_correct'] += (node_coarsest_predicted[node_y == node.children_to_labels[child.name]] == node.children_to_labels[child.name]).sum().item()
            node_accuracy[node.name]['preds'] = torch.cat((node_accuracy[node.name]['preds'], node_logits.detach().cpu())).detach().cpu()#.numpy()
            node_accuracy[node.name]['gts'] = torch.cat((node_accuracy[node.name]['gts'], node_y.detach().cpu())).detach().cpu()#.numpy()

    # # calculate gradients
    # if (not finetune) and train:
    #     al_and_uni.backward() # gradient for all layers before add_on
    # if (not pretrain) and train:
    #     # cl_and_tanh_desc.backward() # gradient for all layer after and including add_on
    #     cl_and_tanh_desc.backward(inputs=features) # gradient for all layer after and including add_on

    if (not pretrain) and (not finetune) and minmaximize and train:
        layers_requiring_gradients = []
        for attr in dir(net.module):
            if attr.endswith('_classification'):
                layers_requiring_gradients.append(getattr(net.module, attr).weight)
                if getattr(net.module, attr).bias is not None:
                    layers_requiring_gradients.append(getattr(net.module, attr).bias)
            if attr.endswith('_add_on'):
                layers_requiring_gradients.append(getattr(net.module, attr).weight)
                if getattr(net.module, attr).bias is not None:
                    layers_requiring_gradients.append(getattr(net.module, attr).bias)
        mm_loss.backward(inputs=layers_requiring_gradients)

    # calculating overall loss
    loss += al_and_uni + cl_and_tanh_desc

    acc=0.
    # if not pretrain:
    #     ys_pred_max = torch.argmax(out, dim=1)
    #     correct = torch.sum(torch.eq(ys_pred_max, ys))
    #     acc = correct.item() / float(len(ys))
    if print: 
        with torch.no_grad():
            if len(OOD_ent_loss) > 0:
                avg_OOD_ent_loss = np.mean([loss_val.item() for node_name, loss_val in OOD_ent_loss.items()])
            else:
                avg_OOD_ent_loss = torch.tensor(-5) # placeholder value

            if len(overspecifity_loss) > 0:
                avg_overspecifity_loss = np.mean([loss_val.item() for node_name, loss_val in overspecifity_loss.items()])
            else:
                avg_overspecifity_loss = torch.tensor(-5) # placeholder value

            if len(mask_l1_loss) > 0:
                avg_mask_l1_loss = np.mean([loss_val.item() for node_name, loss_val in mask_l1_loss.items()])
            else:
                avg_mask_l1_loss = torch.tensor(-5) # placeholder value

            if len(act_l1_loss) > 0:
                avg_act_l1_loss = np.mean([loss_val.item() for node_name, loss_val in act_l1_loss.items()])
            else:
                avg_act_l1_loss = torch.tensor(-5) # placeholder value

            if len(ant_conc_log_ip_loss) > 0:
                avg_ant_conc_log_ip_loss = np.mean([loss_val.item() for node_name, loss_val in ant_conc_log_ip_loss.items()])
            else:
                avg_ant_conc_log_ip_loss = torch.tensor(-5) # placeholder value

            if len(conc_log_ip_loss) > 0:
                avg_conc_log_ip_loss = np.mean([loss_val.item() for node_name, loss_val in conc_log_ip_loss.items()])
            else:
                avg_conc_log_ip_loss = torch.tensor(-5) # placeholder value

            # dict will be empty if not used, so setting the average to a placeholder vale
            if len(cluster_desc_loss) > 0:
                avg_cluster_desc_loss = np.mean([loss_val.item() for node_name, loss_val in cluster_desc_loss.items()])
            else:
                avg_cluster_desc_loss = torch.tensor(-5) # placeholder value
            
            if len(sep_desc_loss) > 0:
                avg_sep_desc_loss = np.mean([loss_val.item() for node_name, loss_val in sep_desc_loss.items()])
            else:
                avg_sep_desc_loss = torch.tensor(-5) # placeholder value

            # dict will be empty if not used, so setting the average to a placeholder vale
            if len(subspace_sep_loss) > 0:
                avg_subspace_sep_loss = np.mean([loss_val.item() for node_name, loss_val in subspace_sep_loss.items()])
            else:
                avg_subspace_sep_loss = torch.tensor(-5) # placeholder value

            # dict will be empty if not used, so setting the average to a placeholder vale
            if len(a_loss_pf) > 0:
                avg_a_loss_pf = np.mean([node_a_loss_pf.item() for node_name, node_a_loss_pf in a_loss_pf.items()])
            else:
                avg_a_loss_pf = torch.tensor(-5) # placeholder value
            
            # dict will be empty if not used, so setting the average to a placeholder vale
            if len(tanh_loss) > 0:
                avg_tanh_loss = np.mean([node_tanh_loss.item() for node_name, node_tanh_loss in tanh_loss.items()])
            else:
                avg_tanh_loss = torch.tensor(-5) # placeholder value

            # dict will be empty if not used, so setting the average to a placeholder vale
            if len(minmaximize_loss) > 0:
                avg_minmaximize_loss = np.mean([node_minmaximize_loss.item() for node_name, node_minmaximize_loss in minmaximize_loss.items()])
            else:
                avg_minmaximize_loss = torch.tensor(-5) # placeholder value

            # dict will be empty if not used, so setting the average to a placeholder vale
            if len(tanh_desc_loss) > 0:
                avg_tanh_desc_loss = np.mean([loss_val.item() for node_name, loss_val in tanh_desc_loss.items()])
            else:
                avg_tanh_desc_loss = torch.tensor(-5) # placeholder value
                
            # avg_tanh_loss = np.mean([node_tanh_loss.item() for node_name, node_tanh_loss in tanh_loss.items()])
            # avg_tanh_desc_loss = np.mean([node_tanh_desc_loss.item() for node_name, node_tanh_desc_loss in tanh_desc_loss.items()])

            # optional loss, dict will be empty if not used, so setting the average to a placeholder vale
            if len(kernel_orth_loss) > 0:
                avg_kernel_orth_loss = np.mean([node_kernel_orth_loss.item() for node_name, node_kernel_orth_loss in kernel_orth_loss.items()])
            else:
                avg_kernel_orth_loss = -5 # placeholder value

            # optional loss, dict will be empty if not used, so setting the average to a placeholder vale
            # if len(uni_loss) > 0:
            #     avg_uni_loss = np.mean([node_uni_loss.item() for node_name, node_uni_loss in uni_loss.items()])
            # else:
            #     avg_uni_loss = -5
            
            avg_class_loss = None
            avg_OOD_loss = None

            # if pretrain:
            #     train_iter.set_postfix_str(
            #     f'L: {loss.item():.3f}, L_OVSP:{avg_overspecifity_loss.item():.3f}, L_MASKL1:{avg_mask_l1_loss.item():.3f}, L_ACT_L1:{avg_act_l1_loss.item():.3f}, L_CONC_LOG_IP:{avg_conc_log_ip_loss.item():.3f}, L_ANT_CONC_LOG_IP:{avg_ant_conc_log_ip_loss.item():.3f}, LA:{a_loss.item():.2f}, L_UNI:{uni_loss.item():.3f}, L_BYOL:{byol_loss.item():.3f}, losses_used:{"+".join(losses_used)}', refresh=False)
            # else:
            #     avg_class_loss = np.mean([node_class_loss.item() for node_name, node_class_loss in class_loss.items()])
            #     avg_OOD_loss = np.mean([node_OOD_loss.item() for node_name, node_OOD_loss in OOD_loss.items()]) if OOD_loss_required else -5
            #     if finetune:
            #         train_iter.set_postfix_str(
            #         f'L:{loss.item():.3f},LC:{avg_class_loss.item():.3f}, L_OVSP:{avg_overspecifity_loss.item():.3f}, L_MASKL1:{avg_mask_l1_loss.item():.3f}, L_ACT_L1:{avg_act_l1_loss.item():.3f}, L_CONC_LOG_IP:{avg_conc_log_ip_loss.item():.3f}, L_ANT_CONC_LOG_IP:{avg_ant_conc_log_ip_loss.item():.3f}, LA:{a_loss.item():.2f}, L_UNI:{uni_loss.item():.3f}, L_OOD:{avg_OOD_loss:.3f}, L_ORTH:{avg_kernel_orth_loss:.3f}, L_BYOL:{byol_loss.item():.3f}, L_CLUS_DESC:{avg_cluster_desc_loss.item():.3f}, L_SEP_DESC:{avg_sep_desc_loss.item():.3f}, LT_DESC:{avg_tanh_desc_loss.item():.3f}, L_SS:{avg_subspace_sep_loss.item():.3f}, losses_used:{"+".join(losses_used)}', refresh=False)
            #     else:
            #         train_iter.set_postfix_str(
            #         f'L:{loss.item():.3f},LC:{avg_class_loss.item():.3f}, L_OVSP:{avg_overspecifity_loss.item():.3f}, L_MASKL1:{avg_mask_l1_loss.item():.3f}, L_ACT_L1:{avg_act_l1_loss.item():.3f}, L_CONC_LOG_IP:{avg_conc_log_ip_loss.item():.3f}, L_ANT_CONC_LOG_IP:{avg_ant_conc_log_ip_loss.item():.3f}, LA:{a_loss.item():.2f}, L_UNI:{uni_loss.item():.3f}, LT:{avg_tanh_loss.item():.3f}, L_MM:{avg_minmaximize_loss.item():.3f}, L_OOD:{avg_OOD_loss:.3f}, L_ORTH:{avg_kernel_orth_loss:.3f}, L_BYOL:{byol_loss.item():.3f}, L_CLUS_DESC:{avg_cluster_desc_loss.item():.3f}, L_SEP_DESC:{avg_sep_desc_loss.item():.3f}, LT_DESC:{avg_tanh_desc_loss.item():.3f}, L_SS:{avg_subspace_sep_loss.item():.3f}, losses_used:{"+".join(losses_used)}', refresh=False)            

            if pretrain:
                train_iter.set_postfix_str(
                f'L: {loss.item():.3f}, L_OVSP:{avg_overspecifity_loss.item():.3f}, L_MASKL1:{avg_mask_l1_loss.item():.3f}, LA_PF:{avg_a_loss_pf.item():.2f}, LT:{avg_tanh_loss.item():.3f}, L_OOD_ENT:{avg_OOD_ent_loss.item():.3f}, losses_used:{"+".join(losses_used)}', refresh=False)
            else:
                avg_class_loss = np.mean([node_class_loss.item() for node_name, node_class_loss in class_loss.items()])
                avg_OOD_loss = np.mean([node_OOD_loss.item() for node_name, node_OOD_loss in OOD_loss.items()]) if OOD_loss_required else -5
                if finetune:
                    train_iter.set_postfix_str(
                    f'L:{loss.item():.3f},LC:{avg_class_loss.item():.3f}, L_OVSP:{avg_overspecifity_loss.item():.3f}, L_MASKL1:{avg_mask_l1_loss.item():.3f}, L_ORTH:{avg_kernel_orth_loss:.3f}, LT_DESC:{avg_tanh_desc_loss.item():.3f}, L_OOD_ENT:{avg_OOD_ent_loss.item():.3f}, losses_used:{"+".join(losses_used)}', refresh=False)
                else:
                    train_iter.set_postfix_str(
                    f'L:{loss.item():.3f},LC:{avg_class_loss.item():.3f}, L_OVSP:{avg_overspecifity_loss.item():.3f}, L_MASKL1:{avg_mask_l1_loss.item():.3f}, LA_PF:{avg_a_loss_pf.item():.3f}, LT:{avg_tanh_loss.item():.3f}, L_ORTH:{avg_kernel_orth_loss:.3f}, LT_DESC:{avg_tanh_desc_loss.item():.3f}, L_OOD_ENT:{avg_OOD_ent_loss.item():.3f}, losses_used:{"+".join(losses_used)}', refresh=False)            
    
    return loss, class_loss, a_loss, tanh_loss, minmaximize_loss, OOD_loss, kernel_orth_loss, uni_loss, avg_class_loss, avg_a_loss_pf, avg_tanh_loss, avg_minmaximize_loss, avg_OOD_loss, avg_kernel_orth_loss, byol_loss.item(), avg_cluster_desc_loss.item(), avg_sep_desc_loss.item(), avg_tanh_desc_loss.item(), avg_subspace_sep_loss.item(), avg_conc_log_ip_loss.item(), acc


def flatten_tensor(x):
    # converts B, C, H, W to B*H*W, C
    permuted_tensor = x.permute(0, 2, 3, 1)
    contiguous_tensor = permuted_tensor.contiguous()
    reshaped_tensor = contiguous_tensor.view(-1, contiguous_tensor.shape[-1])
    return reshaped_tensor

def cdist2(x, y):
    # |x_i - y_j|_2^2 = <x_i - y_j, x_i - y_j> = <x_i, x_i> + <y_j, y_j> - 2*<x_i, y_j>
    x_sq_norm = x.pow(2).sum(dim=-1, keepdim=True)
    y_sq_norm = y.pow(2).sum(dim=-1)
    x_dot_y = x @ y.transpose(-1,-2)
    sq_dist = x_sq_norm + y_sq_norm.unsqueeze(dim=-2) - 2*x_dot_y
    # For numerical issues
    sq_dist.clamp_(min=0.0)
    return torch.sqrt(sq_dist)

def pdist2(x):
    # |x_i - y_j|_2^2 = <x_i - y_j, x_i - y_j> = <x_i, x_i> + <y_j, y_j> - 2*<x_i, y_j>
    x_sq_norm = x.pow(2).sum(dim=-1, keepdim=True)
    x_dot_x = x @ x.transpose(-1,-2)
    sq_dist = 2*x_sq_norm - 2*x_dot_x
    # For numerical issues
    sq_dist.clamp_(min=0.0)
    upper_right_triangle_indices = torch.triu_indices(row=x.shape[0], col=x.shape[0], offset=1)
    dist = torch.sqrt(sq_dist)
    mask = torch.zeros_like(dist)
    mask[upper_right_triangle_indices[0], upper_right_triangle_indices[1]] = 1
    dist = dist * mask
    return dist

# Extra uniform loss from https://www.tongzhouwang.info/hypersphere/. Currently not used but you could try adding it if you want. 
def uniform_loss(x, t=2):
    # print("sum elements: ", torch.sum(torch.pow(x,2), dim=1).shape, torch.sum(torch.pow(x,2), dim=1)) #--> should be ones
    dist = torch.cdist(x.unsqueeze(0), x.unsqueeze(0), p=2.0).pow(2).mul(-t).exp()

    upper_right_triangle_indices = torch.triu_indices(row=x.shape[0], col=x.shape[0], offset=1)
    mask = torch.zeros_like(dist)
    mask[:, upper_right_triangle_indices[0], upper_right_triangle_indices[1]] = 1

    dist = dist * mask
    loss = ((dist.sum() / mask.sum()) + 1e-10).log()
    return loss

# Extra uniform loss from https://www.tongzhouwang.info/hypersphere/. Currently not used but you could try adding it if you want. 
def uniform_loss_orig(x, t=2):
    # print("sum elements: ", torch.sum(torch.pow(x,2), dim=1).shape, torch.sum(torch.pow(x,2), dim=1)) #--> should be ones
    loss = (torch.pdist(x, p=2).pow(2).mul(-t).exp().mean() + 1e-10).log()
    return loss

# alignment loss from https://www.tongzhouwang.info/hypersphere/
def align_loss_unit_space(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

# from https://gitlab.com/mipl/carl/-/blob/main/losses.py
def align_loss(inputs, targets, EPS=1e-12):
    assert inputs.shape == targets.shape
    assert targets.requires_grad == False
    
    loss = torch.einsum("nc,nc->n", [inputs, targets])
    loss = -torch.log(loss + EPS).mean()
    return loss

# from https://github.com/samaonline/Orthogonal-Convolutional-Neural-Networks/blob/master/imagenet/utils.py
def orth_dist(mat, stride=None, device=None):
    mat = mat.reshape((mat.shape[0], -1))
    if mat.shape[0] < mat.shape[1]:
        mat = mat.permute(1,0)
    return torch.norm( torch.t(mat)@mat - torch.eye(mat.shape[1]).to(device))

def regression_loss(x, y):
    norm_x, norm_y = F.normalize(x, p=2, dim=1), F.normalize(y, p=2, dim=1)
    # cosine_similarity = (norm_x * norm_y).sum(dim=-1)
    # loss = -2. * cosine_similarity.mean()
    loss = torch.sum((norm_x - norm_y) ** 2, dim=1)
    loss = loss.mean()
    return loss