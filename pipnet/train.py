from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import math
import numpy as np

from collections import defaultdict

def train_pipnet(net, train_loader, optimizer_net, optimizer_classifier, scheduler_net, scheduler_classifier, criterion, epoch, nr_epochs, device, pretrain=False, finetune=False, progress_prefix: str = 'Train Epoch'):

    root = net.module.root
    name2label = train_loader.dataset.dataset.dataset.class_to_idx
    label2name = {label:name for name, label in name2label.items()}
    node_accuracy = defaultdict(lambda: {'n_examples': 0, 'n_correct': 0, 'accuracy': None, 'children': defaultdict(lambda: {'n_examples': 0, 'n_correct': 0})})

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

    iters = len(train_loader)
    # Show progress on progress bar. 
    train_iter = tqdm(enumerate(train_loader),
                    total=len(train_loader),
                    desc=progress_prefix+'%s'%epoch,
                    mininterval=2.,
                    ncols=0)
    
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
    else:
        align_pf_weight = 5. 
        t_weight = 2.
        unif_weight = 0.
        cl_weight = 2.

    
    print("Align weight: ", align_pf_weight, ", U_tanh weight: ", t_weight, "Class weight:", cl_weight, flush=True)
    print("Pretrain?", pretrain, "Finetune?", finetune, flush=True)
    
    lrs_net = []
    lrs_class = []
    # Iterate through the data set to update leaves, prototypes and network
    for i, (xs1, xs2, ys) in train_iter:       
        
        xs1, xs2, ys = xs1.to(device), xs2.to(device), ys.to(device)
       
        # Reset the gradients
        optimizer_classifier.zero_grad(set_to_none=True)
        optimizer_net.zero_grad(set_to_none=True)
       
        # Perform a forward pass through the network
        proto_features, pooled, out = net(torch.cat([xs1, xs2]))
        
        loss, acc = calculate_loss(proto_features, pooled, out, ys, align_pf_weight, t_weight, unif_weight, cl_weight, \
                                   net.module._multiplier, pretrain, finetune, criterion, \
                                    train_iter, print=True, EPS=1e-8, root=root, label2name=label2name, node_accuracy=node_accuracy)
        
        # Compute the gradient
        loss.backward()

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
                # YTIR - adding this because this was done in the original code, but why this is required this parameter is supposed to be constant & non-trainable
                net.module._multiplier.copy_(torch.clamp(net.module._multiplier.data, min=1.0))
    train_info['train_accuracy'] = total_acc/float(i+1)
    train_info['loss'] = total_loss/float(i+1)
    train_info['lrs_net'] = lrs_net
    train_info['lrs_class'] = lrs_class

    for node_name in node_accuracy:
        node_accuracy[node_name]['accuracy'] = round((node_accuracy[node_name]['n_correct'] / node_accuracy[node_name]['n_examples']) * 100, 2)
    
    train_info['node_accuracy'] = node_accuracy

    for node_name in node_accuracy:
        acc = node_accuracy[node_name]["accuracy"]
        samples = node_accuracy[node_name]["n_examples"]
        log_string = f'\tNode name: {node_name}, acc: {acc}, samples: {samples}'
        for child in net.module.root.get_node(node_name).children:
            child_n_correct = node_accuracy[node_name]['children'][child.name]['n_correct']
            child_n_examples = node_accuracy[node_name]['children'][child.name]['n_examples']
            log_string += ", " + f'{child.name}={child_n_correct}/{child_n_examples}'
        print(log_string)
    
    return train_info

# def calculate_loss(proto_features, pooled, out, ys1, align_pf_weight, t_weight, unif_weight, cl_weight, net_normalization_multiplier, pretrain, finetune, criterion, train_iter, print=True, EPS=1e-10):
#     ys = torch.cat([ys1,ys1])
#     pooled1, pooled2 = pooled.chunk(2)
#     pf1, pf2 = proto_features.chunk(2)

#     embv2 = pf2.flatten(start_dim=2).permute(0,2,1).flatten(end_dim=1)
#     embv1 = pf1.flatten(start_dim=2).permute(0,2,1).flatten(end_dim=1)
    
#     a_loss_pf = (align_loss(embv1, embv2.detach())+ align_loss(embv2, embv1.detach()))/2.
#     tanh_loss = -(torch.log(torch.tanh(torch.sum(pooled1,dim=0))+EPS).mean() + torch.log(torch.tanh(torch.sum(pooled2,dim=0))+EPS).mean())/2.

#     if not finetune:
#         loss = align_pf_weight*a_loss_pf
#         loss += t_weight * tanh_loss
    
#     if not pretrain:
#         softmax_inputs = torch.log1p(out**net_normalization_multiplier)
#         class_loss = criterion(F.log_softmax((softmax_inputs),dim=1),ys)
        
#         if finetune:
#             loss= cl_weight * class_loss
#         else:
#             loss+= cl_weight * class_loss
#     # Our tanh-loss optimizes for uniformity and was sufficient for our experiments. However, if pretraining of the prototypes is not working well for your dataset, you may try to add another uniformity loss from https://www.tongzhouwang.info/hypersphere/ Just uncomment the following three lines
#     # else:
#     #     uni_loss = (uniform_loss(F.normalize(pooled1+EPS,dim=1)) + uniform_loss(F.normalize(pooled2+EPS,dim=1)))/2.
#     #     loss += unif_weight * uni_loss

#     acc=0.
#     if not pretrain:
#         ys_pred_max = torch.argmax(out, dim=1)
#         correct = torch.sum(torch.eq(ys_pred_max, ys))
#         acc = correct.item() / float(len(ys))
#     if print: 
#         with torch.no_grad():
#             if pretrain:
#                 train_iter.set_postfix_str(
#                 f'L: {loss.item():.3f}, LA:{a_loss_pf.item():.2f}, LT:{tanh_loss.item():.3f}, num_scores>0.1:{torch.count_nonzero(torch.relu(pooled-0.1),dim=1).float().mean().item():.1f}',refresh=False)
#             else:
#                 if finetune:
#                     train_iter.set_postfix_str(
#                     f'L:{loss.item():.3f},LC:{class_loss.item():.3f}, LA:{a_loss_pf.item():.2f}, LT:{tanh_loss.item():.3f}, num_scores>0.1:{torch.count_nonzero(torch.relu(pooled-0.1),dim=1).float().mean().item():.1f}, Ac:{acc:.3f}',refresh=False)
#                 else:
#                     train_iter.set_postfix_str(
#                     f'L:{loss.item():.3f},LC:{class_loss.item():.3f}, LA:{a_loss_pf.item():.2f}, LT:{tanh_loss.item():.3f}, num_scores>0.1:{torch.count_nonzero(torch.relu(pooled-0.1),dim=1).float().mean().item():.1f}, Ac:{acc:.3f}',refresh=False)            
#     return loss, acc


def calculate_loss(proto_features, pooled, out, ys1, align_pf_weight, t_weight, unif_weight, cl_weight, net_normalization_multiplier, pretrain, finetune, criterion, train_iter, print=True, EPS=1e-10, root=None, label2name=None, node_accuracy=None):
    batch_names = [label2name[y.item()] for y in ys1]
    loss = 0
    a_loss_pf = {}
    tanh_loss = {}
    for node in root.nodes_with_children():
        children_idx = torch.tensor([name in node.descendents for name in batch_names])
        batch_names_coarsest = [node.closest_descendent_for(name).name for name in batch_names if name in node.descendents]
        node_y = torch.tensor([node.children_to_labels[name] for name in batch_names_coarsest]).cuda()

        if len(node_y) == 0:
            continue

        ys = torch.cat([node_y,node_y])
        pooled1, pooled2 = pooled[node.name].chunk(2)
        pooled1 = pooled1[children_idx]
        pooled2 = pooled2[children_idx]
        pf1, pf2 = proto_features[node.name].chunk(2)
        pf1 = pf1[children_idx]
        pf2 = pf2[children_idx]
        out[node.name] = out[node.name][torch.cat([children_idx,children_idx])] # since out will have 2*batch_size samples

        embv2 = pf2.flatten(start_dim=2).permute(0,2,1).flatten(end_dim=1)
        embv1 = pf1.flatten(start_dim=2).permute(0,2,1).flatten(end_dim=1)
        
        a_loss_pf[node.name] = (align_loss(embv1, embv2.detach()) + align_loss(embv2, embv1.detach())) / 2.
        tanh_loss[node.name] = -(torch.log(torch.tanh(torch.sum(pooled1,dim=0))+EPS).mean() + torch.log(torch.tanh(torch.sum(pooled2,dim=0))+EPS).mean())/2.

        if not finetune:
            loss += align_pf_weight * a_loss_pf[node.name]
            loss += t_weight * tanh_loss[node.name]
        
        if not pretrain:
            softmax_inputs = torch.log1p(out[node.name]**net_normalization_multiplier)
            class_loss = criterion(F.log_softmax((softmax_inputs),dim=1),ys) * (len(node_y) / len(batch_names))
            
            if finetune:
                loss += cl_weight * class_loss
            else:
                loss += cl_weight * class_loss
        # Our tanh-loss optimizes for uniformity and was sufficient for our experiments. However, if pretraining of the prototypes is not working well for your dataset, you may try to add another uniformity loss from https://www.tongzhouwang.info/hypersphere/ Just uncomment the following three lines
        # else:
        #     uni_loss = (uniform_loss(F.normalize(pooled1+EPS,dim=1)) + uniform_loss(F.normalize(pooled2+EPS,dim=1)))/2.
        #     loss += unif_weight * uni_loss

        # For debugging purpose
        node_accuracy[node.name]['n_examples'] += ys.shape[0]
        _, node_coarsest_predicted = torch.max(out[node.name].data, 1)
        node_accuracy[node.name]['n_correct'] += (ys == node_coarsest_predicted).sum().item()
        for child in node.children:
            node_accuracy[node.name]['children'][child.name]['n_examples'] += (ys == node.children_to_labels[child.name]).sum().item()
            node_accuracy[node.name]['children'][child.name]['n_correct'] += (node_coarsest_predicted[ys == node.children_to_labels[child.name]] == node.children_to_labels[child.name]).sum().item()

    acc=0.
    # if not pretrain:
    #     ys_pred_max = torch.argmax(out, dim=1)
    #     correct = torch.sum(torch.eq(ys_pred_max, ys))
    #     acc = correct.item() / float(len(ys))
    if print: 
        with torch.no_grad():
            avg_a_loss_pf = np.mean([node_a_loss_pf.item() for node_name, node_a_loss_pf in a_loss_pf.items()])
            avg_tanh_loss = np.mean([node_tanh_loss.item() for node_name, node_tanh_loss in tanh_loss.items()])
            if pretrain:
                train_iter.set_postfix_str(
                f'L: {loss.item():.3f}, LA:{avg_a_loss_pf.item():.2f}, LT:{avg_tanh_loss.item():.3f}',refresh=False)
            else:
                if finetune:
                    train_iter.set_postfix_str(
                    f'L:{loss.item():.3f},LC:{class_loss.item():.3f}, LA:{avg_a_loss_pf.item():.2f}, LT:{avg_tanh_loss.item():.3f}, Ac:{acc:.3f}',refresh=False)
                else:
                    train_iter.set_postfix_str(
                    f'L:{loss.item():.3f},LC:{class_loss.item():.3f}, LA:{avg_a_loss_pf.item():.2f}, LT:{avg_tanh_loss.item():.3f}, Ac:{acc:.3f}',refresh=False)            
    return loss, acc


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