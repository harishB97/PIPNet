from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import math
import numpy as np

from collections import defaultdict
from torchmetrics.functional import f1_score, recall, precision
from torchvision.datasets.folder import ImageFolder

import wandb

OOD_LABEL = -1

def train_pipnet(net, train_loader, optimizer_net, optimizer_classifier, scheduler_net, scheduler_classifier, criterion, epoch, nr_epochs, device, pretrain=False, finetune=False, progress_prefix: str = 'Train Epoch', wandb_logging=True, train_loader_OOD=None, kernel_orth=False):
    
    root = net.module.root
    dataset = train_loader.dataset
    while type(dataset) != ImageFolder:
        dataset = dataset.dataset
    name2label = dataset.class_to_idx
    label2name = {label:name for name, label in name2label.items()}
    label2name[OOD_LABEL] = 'OOD'

    wandb_log_subdir = 'train'

    # node_accuracy = defaultdict(lambda: {'n_examples': 0, 'n_correct': 0, 'accuracy': None, 'preds': None, 'children': defaultdict(lambda: {'n_examples': 0, 'n_correct': 0})})
    node_accuracy = {}
    for node in root.nodes_with_children():
        node_accuracy[node.name] = {'n_examples': 0, 'n_correct': 0, 'accuracy': None, 'f1': None, 'preds': torch.empty(0, node.num_children()).to(device), 'gts': torch.empty(0).to(device)}
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
    tanh_loss_ep_mean = 0.
    OOD_loss_ep_mean = 0.
    kernel_orth_loss_ep_mean = 0.
    uni_loss_ep_mean = 0.

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
        unif_weight = 0.5 #ignored
        t_weight = 5.
        cl_weight = 0.
        OOD_loss_weight = 0.
        orth_weight = 0.1
    else:
        align_pf_weight = 5. 
        t_weight = 2.
        unif_weight = 0.
        cl_weight = 2.
        OOD_loss_weight = 0.2
        orth_weight = 0.1

    
    print("Align weight: ", align_pf_weight, ", U_tanh weight: ", t_weight, "Class weight:", cl_weight, "OOD_loss weight", OOD_loss_weight, flush=True)
    print("Pretrain?", pretrain, "Finetune?", finetune, flush=True)
    
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
        
        # Perform a forward pass through the network
        proto_features, pooled, out = net(xs)
        
        loss, class_loss_dict, a_loss_pf_dict, tanh_loss_dict, OOD_loss_dict, kernel_orth_loss_dict, uni_loss_dict, avg_class_loss, avg_a_loss_pf, avg_tanh_loss, avg_OOD_loss, avg_kernel_orth_loss, avg_uni_loss, acc = \
            calculate_loss(net, proto_features, pooled, out, ys, align_pf_weight, t_weight, unif_weight, cl_weight, OOD_loss_weight, orth_weight, net.module._multiplier, pretrain, finetune, \
                           criterion, train_iter, print=True, EPS=1e-8, root=root, label2name=label2name, node_accuracy=node_accuracy, OOD_loss_required=OOD_loss_required, kernel_orth=kernel_orth)
        
        # Compute the gradient
        loss.backward()

        # to be modified, not all avg losses will be none if they're not, some use integer placeholder value, to be modified accordingly
        class_loss_ep_mean += avg_class_loss if avg_class_loss else 0.
        a_loss_pf_ep_mean += avg_a_loss_pf if avg_a_loss_pf else 0.
        tanh_loss_ep_mean += avg_tanh_loss if avg_tanh_loss else 0.
        OOD_loss_ep_mean += avg_OOD_loss if avg_OOD_loss else 0.
        kernel_orth_loss_ep_mean += avg_kernel_orth_loss if avg_kernel_orth_loss else 0.
        uni_loss_ep_mean += avg_uni_loss if avg_uni_loss else 0.

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

        if not pretrain:
            with torch.no_grad():
                for attr in dir(net.module):
                    if attr.endswith('_classification'):
                        classification_layer = getattr(net.module, attr)
                        classification_layer.weight.copy_(torch.clamp(classification_layer.weight.data - 1e-3, min=0.)) #set weights in classification layer < 1e-3 to zero
                        if classification_layer.bias is not None:
                            classification_layer.bias.copy_(torch.clamp(classification_layer.bias.data, min=0.))  
                # YTIR - keeping this because this was done in the original code, but why this is required this parameter is supposed to be constant & non-trainable
                net.module._multiplier.copy_(torch.clamp(net.module._multiplier.data, min=1.0))
        
        _, preds_joint = net.module.get_joint_distribution(out)
        preds_joint = preds_joint[ys != OOD_LABEL]
        _, fine_predicted = torch.max(preds_joint.data, 1)
        target = ys[ys != OOD_LABEL]
        fine_correct = fine_predicted == target
        n_fine_correct += fine_correct.sum().item()
        n_samples += target.size(0)

        


    train_info['fine_accuracy'] = n_fine_correct/n_samples

    train_info['train_accuracy'] = total_acc/float(i+1)
    train_info['loss'] = total_loss/float(i+1)
    train_info['lrs_net'] = lrs_net
    train_info['lrs_class'] = lrs_class

    class_loss_ep_mean /= float(i+1)
    a_loss_pf_ep_mean /= float(i+1)
    tanh_loss_ep_mean /= float(i+1)
    OOD_loss_ep_mean /= float(i+1)
    kernel_orth_loss_ep_mean /= float(i+1)
    uni_loss_ep_mean /= float(i+1)

    log_dict = {}
    if wandb_logging:
        log_dict[wandb_log_subdir + "/epoch loss"] = train_info['loss']
        log_dict[wandb_log_subdir + "/fine_accuracy"] = train_info['fine_accuracy']
        log_dict[wandb_log_subdir + "/class_loss"] = class_loss_ep_mean
        log_dict[wandb_log_subdir + "/a_loss_pf"] = a_loss_pf_ep_mean
        log_dict[wandb_log_subdir + "/tanh_loss"] = tanh_loss_ep_mean
        log_dict[wandb_log_subdir + "/OOD_loss"] = OOD_loss_ep_mean
        log_dict[wandb_log_subdir + "/kernel_orth_loss"] = kernel_orth_loss_ep_mean
        log_dict[wandb_log_subdir + "/uni_loss"] = uni_loss_ep_mean
        # wandb.log({wandb_log_subdir + "/epoch loss": train_info['loss']}, step=epoch)
        # wandb.log({wandb_log_subdir + "/epoch lrs_net": train_info['lrs_net']})
        # wandb.log({wandb_log_subdir + "/epoch lrs_class": train_info['lrs_class']})

    for node_name in node_accuracy:
        node_accuracy[node_name]['accuracy'] = round((node_accuracy[node_name]['n_correct'] / node_accuracy[node_name]['n_examples']) * 100, 2)
        node_accuracy[node_name]['f1'] = f1_score(node_accuracy[node_name]["preds"], node_accuracy[node_name]["gts"].to(torch.int), \
                                                    average='weighted', num_classes=net.module.root.get_node(node_name).num_children()).item()
        node_accuracy[node_name]['f1'] = round(node_accuracy[node_name]['f1'] * 100, 2)
        if wandb_logging:
            log_dict[wandb_log_subdir + f"/node_wise/acc:{node_name}"] = node_accuracy[node_name]['accuracy']
            log_dict[wandb_log_subdir + f"/node_wise/f1:{node_name}"] = node_accuracy[node_name]['f1']
    wandb.log(log_dict, step=epoch)

    train_info['node_accuracy'] = node_accuracy
    print('\tFine accuracy:', round(train_info['fine_accuracy'], 2))
    for node_name in node_accuracy:
        acc = node_accuracy[node_name]["accuracy"]
        f1 = node_accuracy[node_name]['f1']
        samples = node_accuracy[node_name]["n_examples"]
        log_string = f'\tNode name: {node_name}, acc: {acc}, f1:{f1}, samples: {samples}'
        for child in net.module.root.get_node(node_name).children:
            child_n_correct = node_accuracy[node_name]['children'][child.name]['n_correct']
            child_n_examples = node_accuracy[node_name]['children'][child.name]['n_examples']
            log_string += ", " + f'{child.name}={child_n_correct}/{child_n_examples}={round(child_n_correct/child_n_examples, 2)}'
        print(log_string)
    
    return train_info


def test_pipnet(net, test_loader, optimizer_net, optimizer_classifier, scheduler_net, scheduler_classifier, criterion, epoch, nr_epochs, device, pretrain=False, finetune=False, progress_prefix: str = 'Test Epoch', wandb_logging=True, test_loader_OOD=None, kernel_orth=False):
    
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
        node_accuracy[node.name] = {'n_examples': 0, 'n_correct': 0, 'accuracy': None, 'f1': None, 'preds': torch.empty(0, node.num_children()).to(device), 'gts': torch.empty(0).to(device)}
        node_accuracy[node.name]['children'] = defaultdict(lambda: {'n_examples': 0, 'n_correct': 0})

    # Make sure the model is in eval mode
    net.eval()
    
    # Store info about the procedure
    test_info = dict()
    total_loss = 0.
    total_acc = 0.
    class_loss_ep_mean = 0.
    a_loss_pf_ep_mean = 0.
    tanh_loss_ep_mean = 0.
    OOD_loss_ep_mean = 0.
    kernel_orth_loss_ep_mean = 0.
    uni_loss_ep_mean = 0.

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
        unif_weight = 0.5 #ignored
        t_weight = 5.
        cl_weight = 0.
        OOD_loss_weight = 0.
        orth_weight = 0.5
    else:
        align_pf_weight = 5. 
        t_weight = 2.
        unif_weight = 0.
        cl_weight = 2.
        OOD_loss_weight = 0.2
        orth_weight = 0.5

    
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
            
            # Perform a forward pass through the network
            proto_features, pooled, out = net(xs)
            
            loss, class_loss_dict, a_loss_pf_dict, tanh_loss_dict, OOD_loss_dict, kernel_orth_loss_dict, uni_loss_dict, avg_class_loss, avg_a_loss_pf, avg_tanh_loss, avg_OOD_loss, avg_kernel_orth_loss, avg_uni_loss, acc = \
            calculate_loss(net, proto_features, pooled, out, ys, align_pf_weight, t_weight, unif_weight, cl_weight, OOD_loss_weight, orth_weight, net.module._multiplier, pretrain, finetune, \
                           criterion, test_iter, print=True, EPS=1e-8, root=root, label2name=label2name, node_accuracy=node_accuracy, OOD_loss_required=OOD_loss_required, kernel_orth=kernel_orth)
            
            class_loss_ep_mean += avg_class_loss if avg_class_loss else 0.
            a_loss_pf_ep_mean += avg_a_loss_pf if avg_a_loss_pf else 0.
            tanh_loss_ep_mean += avg_tanh_loss if avg_tanh_loss else 0.
            OOD_loss_ep_mean += avg_OOD_loss if avg_OOD_loss else 0.
            kernel_orth_loss_ep_mean += avg_kernel_orth_loss if avg_kernel_orth_loss else 0.
            uni_loss_ep_mean += avg_uni_loss if avg_uni_loss else 0.
                
            total_acc+=acc # DUMMY can be removed
            total_loss+=loss.item()
            
            _, preds_joint = net.module.get_joint_distribution(out)
            preds_joint = preds_joint[ys != OOD_LABEL]
            _, fine_predicted = torch.max(preds_joint.data, 1)
            target = ys[ys != OOD_LABEL]
            fine_correct = fine_predicted == target
            n_fine_correct += fine_correct.sum().item()
            n_samples += target.size(0)

    test_info['fine_accuracy'] = n_fine_correct/n_samples

    test_info['accuracy'] = total_acc/float(i+1)
    test_info['loss'] = total_loss/float(i+1)

    class_loss_ep_mean /= float(i+1)
    a_loss_pf_ep_mean /= float(i+1)
    tanh_loss_ep_mean /= float(i+1)
    OOD_loss_ep_mean /= float(i+1)
    kernel_orth_loss_ep_mean /= float(i+1)
    uni_loss_ep_mean /= float(i+1)

    log_dict = {}
    if wandb_logging:
        log_dict[wandb_log_subdir + "/epoch loss"] = test_info['loss']
        log_dict[wandb_log_subdir + "/fine_accuracy"] = test_info['accuracy']
        log_dict[wandb_log_subdir + "/class_loss"] = class_loss_ep_mean
        log_dict[wandb_log_subdir + "/a_loss_pf"] = a_loss_pf_ep_mean
        log_dict[wandb_log_subdir + "/tanh_loss"] = tanh_loss_ep_mean
        log_dict[wandb_log_subdir + "/OOD_loss"] = OOD_loss_ep_mean
        log_dict[wandb_log_subdir + "/kernel_orth_loss"] = kernel_orth_loss_ep_mean
        log_dict[wandb_log_subdir + "/uni_loss"] = uni_loss_ep_mean
        # wandb.log({wandb_log_subdir + "/epoch loss": train_info['loss']}, step=epoch)
        # wandb.log({wandb_log_subdir + "/epoch lrs_net": train_info['lrs_net']})
        # wandb.log({wandb_log_subdir + "/epoch lrs_class": train_info['lrs_class']})

    for node_name in node_accuracy:
        node_accuracy[node_name]['accuracy'] = round((node_accuracy[node_name]['n_correct'] / node_accuracy[node_name]['n_examples']) * 100, 2)
        node_accuracy[node_name]['f1'] = f1_score(node_accuracy[node_name]["preds"], node_accuracy[node_name]["gts"].to(torch.int), \
                                                    average='weighted', num_classes=net.module.root.get_node(node_name).num_children()).item()
        node_accuracy[node_name]['f1'] = round(node_accuracy[node_name]['f1'] * 100, 2)
        if wandb_logging:
            log_dict[wandb_log_subdir + f"/node_wise/acc:{node_name}"] = node_accuracy[node_name]['accuracy']
            log_dict[wandb_log_subdir + f"/node_wise/f1:{node_name}"] = node_accuracy[node_name]['f1']
    wandb.log(log_dict, step=epoch)

    test_info['node_accuracy'] = node_accuracy
    print('\tFine accuracy:', round(test_info['fine_accuracy'], 2))
    for node_name in node_accuracy:
        acc = node_accuracy[node_name]["accuracy"]
        f1 = node_accuracy[node_name]['f1']
        samples = node_accuracy[node_name]["n_examples"]
        log_string = f'\tNode name: {node_name}, acc: {acc}, f1:{f1}, samples: {samples}'
        for child in net.module.root.get_node(node_name).children:
            child_n_correct = node_accuracy[node_name]['children'][child.name]['n_correct']
            child_n_examples = node_accuracy[node_name]['children'][child.name]['n_examples']
            log_string += ", " + f'{child.name}={child_n_correct}/{child_n_examples}={round(child_n_correct/child_n_examples, 2)}'
        print(log_string)
    
    return test_info


def calculate_loss(net, proto_features, pooled, out, ys, align_pf_weight, t_weight, unif_weight, cl_weight, OOD_loss_weight, orth_weight, net_normalization_multiplier, pretrain, finetune, criterion, train_iter, print=True, EPS=1e-10, root=None, label2name=None, node_accuracy=None, OOD_loss_required=False, kernel_orth=False):
    batch_names = [label2name[y.item()] for y in ys]
    loss = 0
    class_loss = {}
    a_loss_pf = {}
    tanh_loss = {}
    OOD_loss = {}
    kernel_orth_loss = {}
    uni_loss = {}
    for node in root.nodes_with_children():
        children_idx = torch.tensor([name in node.descendents for name in batch_names])
        batch_names_coarsest = [node.closest_descendent_for(name).name for name in batch_names if name in node.descendents]
        node_y = torch.tensor([node.children_to_labels[name] for name in batch_names_coarsest]).cuda()

        if len(node_y) == 0:
            continue

        # ys = torch.cat([node_y,node_y])

        # pooled1, pooled2 = pooled[node.name].chunk(2)
        # pooled1 = pooled1[children_idx]
        # pooled2 = pooled2[children_idx]

        pooled1, pooled2 = pooled[node.name][children_idx].chunk(2)

        # pf1, pf2 = proto_features[node.name].chunk(2)
        # pf1 = pf1[children_idx]
        # pf2 = pf2[children_idx]

        pf1, pf2 = proto_features[node.name][children_idx].chunk(2)

        # out[node.name] = out[node.name][torch.cat([children_idx,children_idx])] # since out will have 2*batch_size samples
        # node_logits = out[node.name][torch.cat([children_idx,children_idx])] # since out will have 2*batch_size samples
        node_logits = out[node.name][children_idx]

        embv2 = pf2.flatten(start_dim=2).permute(0,2,1).flatten(end_dim=1)
        embv1 = pf1.flatten(start_dim=2).permute(0,2,1).flatten(end_dim=1)
        
        a_loss_pf[node.name] = (align_loss(embv1, embv2.detach()) \
                                + align_loss(embv2, embv1.detach())) / 2.

        tanh_loss[node.name] = -(torch.log(torch.tanh(torch.sum(pooled1,dim=0))+EPS).mean() \
                                 + torch.log(torch.tanh(torch.sum(pooled2,dim=0))+EPS).mean()) / 2.
        
        if kernel_orth:
            prototype_kernels = getattr(net.module, '_'+node.name+'_add_on')
            classification_layer = getattr(net.module, '_'+node.name+'_classification')
            # using any below because its a relevant prototype if it has strong connection to any one of the class
            relevant_prototype_kernels = prototype_kernels.weight[(classification_layer.weight > 0.001).any(dim=0)]
            kernel_orth_loss[node.name] = orth_dist(relevant_prototype_kernels)

        if not finetune:
            # pretraining or general training
            loss += align_pf_weight * a_loss_pf[node.name]
            loss += t_weight * tanh_loss[node.name]
            if kernel_orth:
                loss += orth_weight * kernel_orth_loss[node.name]
        
        if not pretrain:
            # finetuning or general training
            softmax_inputs = torch.log1p(node_logits**net_normalization_multiplier)
            class_loss[node.name] = criterion(F.log_softmax((softmax_inputs),dim=1),node_y) # * (len(node_y) / len(ys[ys != OOD_LABEL]))
            loss += cl_weight * class_loss[node.name]

            if OOD_loss_required:
                not_children_idx = torch.tensor([name not in node.descendents for name in batch_names]) # includes OOD images as well as images belonging to other nodes
                OOD_logits = out[node.name][not_children_idx] # [sum(not_children_idx), node.num_children()]
                sigmoid_out = F.sigmoid(torch.log1p(OOD_logits**net_normalization_multiplier))
                OOD_loss[node.name] = F.binary_cross_entropy(sigmoid_out, torch.zeros_like(OOD_logits))
                loss += OOD_loss_weight * OOD_loss[node.name]
        # Our tanh-loss optimizes for uniformity and was sufficient for our experiments. However, if pretraining of the prototypes is not working well for your dataset, you may try to add another uniformity loss from https://www.tongzhouwang.info/hypersphere/ Just uncomment the following three lines
        else:
            uni_loss[node.name] = (uniform_loss(F.normalize(pooled1+EPS,dim=1)) + uniform_loss(F.normalize(pooled2+EPS,dim=1)))/2.
            loss += unif_weight * uni_loss[node.name]

        # For debugging purpose
        node_accuracy[node.name]['n_examples'] += node_y.shape[0]
        _, node_coarsest_predicted = torch.max(node_logits.data, 1)
        node_accuracy[node.name]['n_correct'] += (node_y == node_coarsest_predicted).sum().item()
        for child in node.children:
            node_accuracy[node.name]['children'][child.name]['n_examples'] += (node_y == node.children_to_labels[child.name]).sum().item()
            node_accuracy[node.name]['children'][child.name]['n_correct'] += (node_coarsest_predicted[node_y == node.children_to_labels[child.name]] == node.children_to_labels[child.name]).sum().item()
            node_accuracy[node.name]['preds'] = torch.cat((node_accuracy[node.name]['preds'], node_logits))
            node_accuracy[node.name]['gts'] = torch.cat((node_accuracy[node.name]['gts'], node_y))

    acc=0.
    # if not pretrain:
    #     ys_pred_max = torch.argmax(out, dim=1)
    #     correct = torch.sum(torch.eq(ys_pred_max, ys))
    #     acc = correct.item() / float(len(ys))
    if print: 
        with torch.no_grad():
            avg_a_loss_pf = np.mean([node_a_loss_pf.item() for node_name, node_a_loss_pf in a_loss_pf.items()])
            avg_tanh_loss = np.mean([node_tanh_loss.item() for node_name, node_tanh_loss in tanh_loss.items()])

            # optional loss, dict will be empty if not used, so setting the average to a placeholder vale
            if len(kernel_orth_loss) > 0:
                avg_kernel_orth_loss = np.mean([node_kernel_orth_loss.item() for node_name, node_kernel_orth_loss in kernel_orth_loss.items()])
            else:
                avg_kernel_orth_loss = -5 # placeholder value

            # optional loss, dict will be empty if not used, so setting the average to a placeholder vale
            if len(uni_loss) > 0:
                avg_uni_loss = np.mean([node_uni_loss.item() for node_name, node_uni_loss in uni_loss.items()])
            else:
                avg_uni_loss = -5
            
            avg_class_loss = None
            avg_OOD_loss = None
            if pretrain:
                # # optional loss, dict will be empty if not used, so setting the average to a placeholder vale
                # if len(uni_loss) > 0:
                #     avg_uni_loss = np.mean([node_uni_loss.item() for node_name, node_uni_loss in uni_loss.items()])
                # else:
                #     avg_uni_loss = -5
                train_iter.set_postfix_str(
                f'L: {loss.item():.3f}, LA:{avg_a_loss_pf.item():.2f}, LT:{avg_tanh_loss.item():.3f}, L_ORTH:{avg_kernel_orth_loss:.3f}, L_UNI:{avg_uni_loss:.3f}',refresh=False)
            else:
                avg_class_loss = np.mean([node_class_loss.item() for node_name, node_class_loss in class_loss.items()])
                avg_OOD_loss = np.mean([node_OOD_loss.item() for node_name, node_OOD_loss in OOD_loss.items()]) if OOD_loss_required else -5
                if finetune:
                    train_iter.set_postfix_str(
                    f'L:{loss.item():.3f},LC:{avg_class_loss.item():.3f}, LA:{avg_a_loss_pf.item():.2f}, LT:{avg_tanh_loss.item():.3f}, L_OOD:{avg_OOD_loss:.3f}',refresh=False)
                else:
                    train_iter.set_postfix_str(
                    f'L:{loss.item():.3f},LC:{avg_class_loss.item():.3f}, LA:{avg_a_loss_pf.item():.2f}, LT:{avg_tanh_loss.item():.3f}, L_OOD:{avg_OOD_loss:.3f}, L_ORTH:{avg_kernel_orth_loss:.3f}',refresh=False)            
    return loss, class_loss, a_loss_pf, tanh_loss, OOD_loss, kernel_orth_loss, uni_loss, avg_class_loss, avg_a_loss_pf, avg_tanh_loss, avg_OOD_loss, avg_kernel_orth_loss, avg_uni_loss, acc


# Extra uniform loss from https://www.tongzhouwang.info/hypersphere/. Currently not used but you could try adding it if you want. 
def uniform_loss(x, t=2):
    # print("sum elements: ", torch.sum(torch.pow(x,2), dim=1).shape, torch.sum(torch.pow(x,2), dim=1)) #--> should be ones
    loss = (torch.pdist(x, p=2).pow(2).mul(-t).exp().mean() + 1e-10).log()
    return loss

# from https://gitlab.com/mipl/carl/-/blob/main/losses.py
def align_loss(inputs, targets, EPS=1e-12):
    assert inputs.shape == targets.shape
    assert targets.requires_grad == False
    
    loss = torch.einsum("nc,nc->n", [inputs, targets])
    loss = -torch.log(loss + EPS).mean()
    return loss

# from https://github.com/samaonline/Orthogonal-Convolutional-Neural-Networks/blob/master/imagenet/utils.py
def orth_dist(mat, stride=None):
    mat = mat.reshape( (mat.shape[0], -1) )
    if mat.shape[0] < mat.shape[1]:
        mat = mat.permute(1,0)
    return torch.norm( torch.t(mat)@mat - torch.eye(mat.shape[1]).cuda())