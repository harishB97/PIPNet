import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from features.resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet50_features_inat, resnet101_features, resnet152_features
from features.convnext_features import convnext_tiny_26_features, convnext_tiny_13_features, convnext_tiny_7_features, apply_gaussian_multiplier_to_convnext_stage
import torch
from torch import Tensor
from util.node import Node
import numpy as np
from collections import defaultdict, OrderedDict
from pipnet_byol.pipnet_byol import PIPNetBYOL
import pdb

# from https://gitlab.com/mipl/carl/-/blob/main/losses.py
def align_loss(inputs, targets, EPS=1e-12):
    assert inputs.shape == targets.shape
    assert targets.requires_grad == False
    
    loss = torch.einsum("nc,nc->n", [inputs, targets])
    loss = -torch.log(loss + EPS).mean()
    return loss

def findCorrespondingToMax(base, target):
    output, indices = F.max_pool2d(base, kernel_size=(base.shape[-1], base.shape[-2]), return_indices=True)# these are logits
    tensor_flattened = target.view(target.shape[0], target.shape[1], -1)
    indices_flattened = indices.view(target.shape[0], target.shape[1], -1)
    corresponding_values_in_target = torch.gather(tensor_flattened, 2, indices_flattened)
    corresponding_values_in_target = corresponding_values_in_target.view(target.shape[0],\
                                     target.shape[1], 1, 1)
    pooled_target = corresponding_values_in_target
    return pooled_target

def functional_UnitConv2D(in_features, weight, bias, stride = 1, padding=0):
    normalized_weight = F.normalize(weight.data, p=2, dim=(1, 2, 3)) # Normalize the kernels to unit vectors
    normalized_input = F.normalize(in_features, p=2, dim=1) # Normalize the input to unit vectors
    if bias is not None:
        normalized_bias = F.normalize(bias.data, p=2, dim=0) # Normalize the kernels to unit vectors
    else:
        normalized_bias = None
    return F.conv2d(normalized_input, normalized_weight, normalized_bias, stride=stride, padding=padding)

class GumbelSoftmax(nn.Module):
    def __init__(self, tau=1, hard=False, dim=-1):
        super(GumbelSoftmax, self).__init__()
        self.tau = tau
        self.hard = hard
        self.dim = dim

    def forward(self, logits):
        return F.gumbel_softmax(logits, tau=self.tau, hard=self.hard, dim=self.dim)


class PIPNet(nn.Module):
    def __init__(self,
                 num_classes: int,
                 num_prototypes: int,
                 feature_net: nn.Module,
                 args: argparse.Namespace,
                 add_on_layers: nn.Module,
                 pool_layer: nn.Module,
                 classification_layers: dict, #nn.Module,
                 num_parent_nodes: int,
                 root: Node
                 ):
        super().__init__()
        assert num_classes > 0
        # self._num_features = args.num_features
        self._num_classes = num_classes
        # self._num_prototypes = num_prototypes # this is only the minimum number of protos per node, might vary for each node
        self._net = feature_net
        # self._add_on = add_on_layers
        for node_name in add_on_layers:
            setattr(self, '_'+node_name+'_add_on', add_on_layers[node_name])
            setattr(self, '_'+node_name+'_num_protos', add_on_layers[node_name].weight.shape[0])
        self._pool = pool_layer
        self._avg_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1,1)), #outputs (bs, ps,1,1)
                nn.Flatten() #outputs (bs, ps)
                ) 
        for node_name in classification_layers:
            setattr(self, '_'+node_name+'_classification', classification_layers[node_name])
        # self._classification = classification_layers
        self._multiplier = nn.Parameter(torch.ones((1,),requires_grad=True)) # this can directly be set to 2.0 and requires_grad=False, not sure why its not done
        # self._multiplier = classification_layers.normalization_multiplier
        if args.softmax.split('|')[0] == 'y':
            self._softmax = nn.Softmax(dim=1)
        elif args.gumbel_softmax == 'y':
            self._gumbel_softmax = GumbelSoftmax(tau=args.gs_tau, hard=False, dim=1)
        self._num_parent_nodes = num_parent_nodes # I dont remember why this was added
        self.root = root

        for node_name in add_on_layers:
            num_prototypes = getattr(self, '_'+node_name+'_num_protos')
            proto_presence = torch.zeros(num_prototypes, 2)
            proto_presence = nn.Parameter(proto_presence, requires_grad=True)
            nn.init.xavier_normal_(proto_presence, gain=1.0)
            setattr(self, '_'+node_name+'_proto_presence', proto_presence)
            # getattr(self, '_'+'root'+'_proto_presence').requires_grad
            

        self.args = args

        if (args.multiply_cs_softmax == 'y') and not (args.softmax.split('|')[0] == 'y' or args.gumbel_softmax == 'y'):
            raise Exception('Use either softmax or gumbel softmax when using multiply_cs_softmax')
        
        self.conc_log_ip = ('y' in args.conc_log_ip)
        # self.conv_layer_in_add_on = type(self._add_on[0]) == nn.Conv2d

    
    def forward(self, xs,  inference=False, apply_overspecificity_mask=False):
        features = self._net(xs) 
        proto_features = {}
        proto_features_cs = {}
        proto_features_softmaxed = {}
        pooled = {}
        out = {}

        if self.args.sg_before_protos == 'y':
            proto_layer_input_features = features.clone().detach()
        else:
            proto_layer_input_features = features

        for node in self.root.nodes_with_children():
            proto_features[node.name] = getattr(self, '_'+node.name+'_add_on')(proto_layer_input_features)

            if isinstance(getattr(self, '_'+node.name+'_add_on'), UnitConv2D):
                proto_features[node.name] = torch.abs(proto_features[node.name])

            if self.args.softmax.split('|')[0] == 'y':
                if len(self.args.softmax.split('|')) > 1:
                    softmax_tau = int(self.args.softmax.split('|')[1])
                else:
                    if isinstance(getattr(self, '_'+node.name+'_add_on'), ProjectConv2D):
                        raise Exception('Do not use softmax temp 0.2 for project distance')
                    softmax_tau = 0.2
                
                if self.args.softmax_over_channel == 'y': #self.conc_log_ip:
                    # softmax over the channel instead of over the patch
                    B, C, H, W = proto_features[node.name].shape
                    proto_features[node.name] = proto_features[node.name].reshape(B, C, -1)
                    proto_features_softmaxed[node.name] = F.softmax(proto_features[node.name], dim=-1)
                    proto_features_softmaxed[node.name] = proto_features_softmaxed[node.name].reshape(B, C, H, W)
                    proto_features[node.name] = proto_features_softmaxed[node.name]
                else:
                    proto_features[node.name] = proto_features[node.name] / softmax_tau
                    proto_features_softmaxed[node.name] = self._softmax(proto_features[node.name])
                    proto_features[node.name] = proto_features_softmaxed[node.name] # will be overwritten if args.multiply_cs_softmax == 'y'
            
            elif self.args.gumbel_softmax == 'y':
                proto_features_softmaxed[node.name] = self._gumbel_softmax(proto_features[node.name])
                proto_features[node.name] = proto_features_softmaxed[node.name] # will be overwritten if args.multiply_cs_softmax == 'y'

            if self.args.multiply_cs_softmax == 'y':
                prototypes = getattr(self, '_'+node.name+'_add_on')
                cosine_similarity = functional_UnitConv2D(proto_layer_input_features, prototypes.weight, prototypes.bias)
                proto_features[node.name] = cosine_similarity * proto_features_softmaxed[node.name]

            pooled[node.name] = self._pool(proto_features[node.name])

            if self.args.focal == 'y':
                pooled[node.name] = pooled[node.name] - self._avg_pool(proto_features[node.name])
            
            if apply_overspecificity_mask:
                mask = F.gumbel_softmax(getattr(self, '_'+node.name+'_proto_presence'), tau=0.5, hard=True, dim=-1)[:, 1].unsqueeze(0)
                pooled[node.name] = mask * pooled[node.name]

            if inference:
                pooled[node.name] = torch.where(pooled[node.name] < 0.1, 0., pooled[node.name])  #during inference, ignore all prototypes that have 0.1 similarity or lower
            out[node.name] = getattr(self, '_'+node.name+'_classification')(pooled[node.name]) #shape (bs*2, num_classes) # these are logits
        return features, proto_features, pooled, out
    
    def get_joint_distribution(self, out, leave_out_classes=None, apply_overspecificity_mask=False, device='cuda', softmax_tau=1):
        batch_size = out['root'].size(0)
        #top_level = torch.nn.functional.softmax(self.root.logits,1)            
        top_level = out['root']
        bottom_level = self.root.distribution_over_furthest_descendents(net=self, batch_size=batch_size, out=out, leave_out_classes=leave_out_classes,\
                                                                        apply_overspecificity_mask=apply_overspecificity_mask, device='cuda', softmax_tau=softmax_tau)    
        names = self.root.unwrap_names_of_joint(self.root.names_of_joint_distribution())
        idx = np.argsort(names)
        bottom_level = bottom_level[:,idx]        

        # num_classes = max([idx for _, idx in class_to_idx]) + 1
        # torch.zeros((bottom_level.shape[0], num_classes))
        return top_level, bottom_level
    
    def get_classification_layers(self):
        return [getattr(self, attr) for attr in dir(self) if attr.endswith('_classification')] 

    def calculate_loss(self, epoch, net, additional_network_outputs, features, proto_features, pooled, out, ys, align_weight, align_pf_weight, t_weight, mm_weight, unif_weight, cl_weight, OOD_loss_weight, \
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
                            breakpoint()
                            pdb.set_trace()
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

            if ('y' in args.conc_log_ip):
                conc_log_ip_weight = 0.01
                TOPK = int(args.conc_log_ip.split('|')[1]) if (len(args.conc_log_ip.split('|')) > 1) else 1
                EPS=1e-12
                num_protos = pooled[node.name].shape[-1]
                classification_weights = getattr(net.module, '_'+node.name+'_classification').weight

                # if node.name == 'root':
                #     breakpoint()
                if (len(args.conc_log_ip.split('|')) > 2) and (epoch < int(args.conc_log_ip.split('|')[2])):
                    pass
                else:
                    if args.protopool == 'n':
                        # mask = torch.zeros_like(pooled[node.name][children_idx], dtype=torch.bool) # [batch_size, num_protos]
                        node_y_expanded = node_y.unsqueeze(-1).repeat(1, num_protos) # [batch_size] to [batch_size, num_protos]
                        conc_log_ip_loss[node.name] = 0.
                        for child_node in node.children:
                            child_class_idx = node.children_to_labels[child_node.name]
                            relevant_proto_idx = torch.nonzero(classification_weights[child_class_idx, :] > 1e-3).squeeze(-1)
                            # torch.nonzero(classification_weights[1, :] > 1e-3).squeeze(-1)
                            # mask[:, relevant_proto_idx] = node_y_expanded[:, relevant_proto_idx] == child_class_idx

                            relevant_data_idx = torch.nonzero(node_y == child_class_idx).squeeze(-1)

                            if len(relevant_data_idx) == 0:
                                continue # no data points with child_class_idx present so skip this child_node

                            _, topk_idx = torch.topk(pooled[node.name][children_idx][relevant_data_idx, :][:, relevant_proto_idx], dim=0, k=TOPK)
                            # proto_idx = relevant_proto_idx.unsqueeze(0).repeat(TOPK, 1) # [topk, num_protos]
                            proto_idx = torch.arange(0, len(relevant_proto_idx)).unsqueeze(0).repeat(TOPK, 1) # [topk, num_protos]
                            topk_idx = topk_idx.reshape(-1) # [topk*num_protos]
                            proto_idx = proto_idx.reshape(-1) # [topk*num_protos]
                            topk_activation_maps = proto_features[node.name][children_idx][relevant_data_idx, :][:, relevant_proto_idx][topk_idx, proto_idx]
                            
                            if args.conc_log_ip_peak_normalize == 'y':
                                max_vals, _ = torch.max(topk_activation_maps.reshape(topk_activation_maps.size(0), -1), dim=1, keepdim=True)
                                max_vals = max_vals.reshape(-1, 1, 1)
                                topk_activation_maps = topk_activation_maps / max_vals
                            
                            conc_log_ip_temp = torch.einsum("nhw,nhw->n", [topk_activation_maps, topk_activation_maps.clone().detach()])
                            conc_log_ip_loss_for_child = -torch.log(conc_log_ip_temp + EPS).mean()
                            conc_log_ip_loss[node.name] += conc_log_ip_loss_for_child

                            # if torch.isnan(conc_log_ip_loss_for_child):
                            #     breakpoint()

                            loss += conc_log_ip_weight * conc_log_ip_loss_for_child / (len(root.nodes_with_children()) if normalize_by_node_count else 1.)
                            if not 'CONC_LOG_IP' in losses_used:
                                losses_used.append('CONC_LOG_IP')

                    else:
                        _, topk_idx = torch.topk(pooled[node.name][children_idx], dim=0, k=TOPK) # [topk, num_protos]
                        proto_idx = torch.arange(0, num_protos).unsqueeze(0).repeat(TOPK, 1) # [topk, num_protos]
                        topk_idx = topk_idx.reshape(-1) # [topk*num_protos]
                        proto_idx = proto_idx.reshape(-1) # [topk*num_protos]
                        topk_activation_maps = proto_features[node.name][children_idx][topk_idx, proto_idx]
                        conc_log_ip_temp = torch.einsum("nhw,nhw->n", [topk_activation_maps, topk_activation_maps.clone().detach()])
                        conc_log_ip_loss[node.name] = -torch.log(conc_log_ip_temp + EPS).mean()

                        loss += conc_log_ip_weight * conc_log_ip_loss[node.name] / (len(root.nodes_with_children()) if normalize_by_node_count else 1.)
                        if not 'CONC_LOG_IP' in losses_used:
                            losses_used.append('CONC_LOG_IP')

            if ('y' in args.ant_conc_log_ip):
                ant_conc_log_ip_weight = 0.01
                TOPK = int(args.ant_conc_log_ip.split('|')[1]) if (len(args.ant_conc_log_ip.split('|')) > 1) else 1
                EPS=1e-12
                num_protos = pooled[node.name].shape[-1]
                classification_weights = getattr(net.module, '_'+node.name+'_classification').weight
                if args.protopool == 'n':
                    node_y_expanded = node_y.unsqueeze(-1).repeat(1, num_protos) # [batch_size] to [batch_size, num_protos]
                    ant_conc_log_ip_loss[node.name] = 0.
                    for child_node in node.children:
                        child_class_idx = node.children_to_labels[child_node.name]
                        relevant_proto_idx = torch.nonzero(classification_weights[child_class_idx, :] > 1e-3).squeeze(-1)
                        
                        # look at data points that do not belong to child_node
                        relevant_data_idx = torch.nonzero(node_y != child_class_idx).squeeze(-1)

                        if len(relevant_data_idx) == 0:
                            continue # no data points that is not equal to child_class_idx present so skip this child_node

                        _, topk_idx = torch.topk(pooled[node.name][children_idx][relevant_data_idx, :][:, relevant_proto_idx], dim=0, k=TOPK)
                        # proto_idx = relevant_proto_idx.unsqueeze(0).repeat(TOPK, 1) # [topk, num_protos]
                        proto_idx = torch.arange(0, len(relevant_proto_idx)).unsqueeze(0).repeat(TOPK, 1) # [topk, num_protos]
                        topk_idx = topk_idx.reshape(-1) # [topk*num_protos]
                        proto_idx = proto_idx.reshape(-1) # [topk*num_protos]
                        topk_activation_maps = proto_features[node.name][children_idx][relevant_data_idx, :][:, relevant_proto_idx][topk_idx, proto_idx]
                        ant_conc_log_ip_temp = torch.einsum("nhw,nhw->n", [topk_activation_maps, topk_activation_maps.clone().detach()])
                        ant_conc_log_ip_loss_for_child = torch.log(ant_conc_log_ip_temp + EPS).mean() # IMPORTANT: this should be POSITIVE
                        ant_conc_log_ip_loss[node.name] += ant_conc_log_ip_loss_for_child

                        # if torch.isnan(conc_log_ip_loss_for_child):
                        #     breakpoint()

                        loss += ant_conc_log_ip_weight * ant_conc_log_ip_loss_for_child / (len(root.nodes_with_children()) if normalize_by_node_count else 1.)
                        if not 'ANT_CONC_LOG_IP' in losses_used:
                            losses_used.append('ANT_CONC_LOG_IP')
                else:
                    raise Exception('Do not use ant_conc_log_ip loss when protopool is true')

            if ('y' in args.act_l1):
                act_l1_weight = 0.01
                TOPK = int(args.act_l1.split('|')[1]) if (len(args.act_l1.split('|')) > 1) else 1
                EPS=1e-12
                num_protos = pooled[node.name].shape[-1]
                classification_weights = getattr(net.module, '_'+node.name+'_classification').weight

                if args.protopool == 'n':
                    # mask = torch.zeros_like(pooled[node.name][children_idx], dtype=torch.bool) # [batch_size, num_protos]
                    # node_y_expanded = node_y.unsqueeze(-1).repeat(1, num_protos) # [batch_size] to [batch_size, num_protos]
                    act_l1_loss[node.name] = 0.
                    for child_node in node.children:
                        child_class_idx = node.children_to_labels[child_node.name]
                        relevant_proto_idx = torch.nonzero(classification_weights[child_class_idx, :] > 1e-3).squeeze(-1)
                        # torch.nonzero(classification_weights[1, :] > 1e-3).squeeze(-1)
                        # mask[:, relevant_proto_idx] = node_y_expanded[:, relevant_proto_idx] == child_class_idx

                        relevant_data_idx = torch.nonzero(node_y == child_class_idx).squeeze(-1)

                        if len(relevant_data_idx) == 0:
                            continue # no data points with child_class_idx present so skip this child_node

                        _, topk_idx = torch.topk(pooled[node.name][children_idx][relevant_data_idx, :][:, relevant_proto_idx], dim=0, k=min(TOPK, len(relevant_data_idx)))
                        # proto_idx = relevant_proto_idx.unsqueeze(0).repeat(TOPK, 1) # [topk, num_protos]
                        proto_idx = torch.arange(0, len(relevant_proto_idx)).unsqueeze(0).repeat(min(TOPK, len(relevant_data_idx)), 1) # [topk, num_protos]
                        topk_idx = topk_idx.reshape(-1) # [topk*num_protos]
                        proto_idx = proto_idx.reshape(-1) # [topk*num_protos]   
                        topk_activation_maps = proto_features[node.name][children_idx][relevant_data_idx, :][:, relevant_proto_idx][topk_idx, proto_idx]
                        # if topk_activation_maps.size(0) == 0:
                        #     breakpoint()
                        max_vals, _ = torch.max(topk_activation_maps.reshape(topk_activation_maps.size(0), -1), dim=1, keepdim=True)
                        max_vals = max_vals.reshape(-1, 1, 1)
                        mask = topk_activation_maps != max_vals
                        masked_tensor = topk_activation_maps * mask
                        act_l1_loss_for_child = masked_tensor.abs().mean()
                        act_l1_loss[node.name] += act_l1_loss_for_child

                        loss += act_l1_weight * act_l1_loss_for_child / (len(root.nodes_with_children()) if normalize_by_node_count else 1.)
                        if not 'ACT_L1' in losses_used:
                            losses_used.append('ACT_L1')

                else:
                    # classification_weights = getattr(net.module, '_'+node.name+'_classification').weight
                    # node_y.unsqueeze(-1).repeat(1, num_protos)
                    # torch.nonzero(classification_weights[0, :] > 1e-3).squeeze(-1)
                    _, topk_idx = torch.topk(pooled[node.name][children_idx], dim=0, k=min(TOPK, pooled[node.name][children_idx].shape[0])) # [topk, num_protos]
                    proto_idx = torch.arange(0, num_protos).unsqueeze(0).repeat(min(TOPK, pooled[node.name][children_idx].shape[0]), 1) # [topk, num_protos]
                    topk_idx = topk_idx.reshape(-1) # [topk*num_protos]
                    proto_idx = proto_idx.reshape(-1) # [topk*num_protos]
                    topk_activation_maps = proto_features[node.name][children_idx][topk_idx, proto_idx]
                    max_vals, _ = torch.max(topk_activation_maps.reshape(topk_activation_maps.size(0), -1), dim=1, keepdim=True)
                    max_vals = max_vals.reshape(-1, 1, 1)
                    mask = topk_activation_maps != max_vals
                    masked_tensor = topk_activation_maps * mask
                    act_l1_loss[node.name] = masked_tensor.abs().mean()

                    loss += act_l1_weight * act_l1_loss[node.name] / (len(root.nodes_with_children()) if normalize_by_node_count else 1.)
                    if not 'ACT_L1' in losses_used:
                        losses_used.append('ACT_L1')

            if (not pretrain) and (not finetune) and minmaximize:
                minmaximize_loss[node.name] = 0
                for child_node in node.children:
                    if child_node.name not in batch_names_coarsest:
                        continue
                    child_label = node.children_to_labels[child_node.name]
                    list_of_min_wrt_each_proto = []
                    if child_node.is_leaf():
                        descendant_idx = torch.tensor([name == child_node.name for name in batch_names])
                        if int(descendant_idx.sum().item()) == 0:
                            continue
                        idx_of_protos_relevant_to_child = getattr(net.module, '_'+node.name+'_classification').weight[child_label] > 1e-3
                        # pooled[node.name].shape => (B, node.num_protos)
                        # proto_activations_of_descendants.shape => (sum(descendant_idx), num_relevant_protos)
                        proto_activations_of_descendants = pooled[node.name][descendant_idx][:, idx_of_protos_relevant_to_child]
                        # min_wrt_each_proto.shape => (num_relevant_protos), since minimum taken over descendant images in the batch
                        min_wrt_each_proto, _ = torch.min(proto_activations_of_descendants, dim=0)
                        list_of_min_wrt_each_proto.append(min_wrt_each_proto)
                    else:
                        for descendant_name in child_node.leaf_descendents:
                            descendant_idx = torch.tensor([name == descendant_name for name in batch_names])
                            if int(descendant_idx.sum().item()) == 0:
                                continue
                            idx_of_protos_relevant_to_child = getattr(net.module, '_'+node.name+'_classification').weight[child_label] > 1e-3
                            # pooled[node.name].shape => (B, node.num_protos)
                            # proto_activations_of_descendants.shape => (sum(descendant_idx), num_relevant_protos)
                            proto_activations_of_descendants = pooled[node.name][descendant_idx][:, idx_of_protos_relevant_to_child]
                            # min_wrt_each_proto.shape => (num_relevant_protos), since minimum taken over descendant images in the batch
                            min_wrt_each_proto, _ = torch.min(proto_activations_of_descendants, dim=0)
                            list_of_min_wrt_each_proto.append(min_wrt_each_proto)
                    # stack_of_min_wrt_each_proto.shape => (len(child_node.leaf_descendents), num_relevant_protos)
                        
                    stack_of_min_wrt_each_proto = torch.stack(list_of_min_wrt_each_proto, dim=0)
                    minmaximize_loss[node.name] += -torch.sum(torch.mean(stack_of_min_wrt_each_proto, dim=0))

                mm_loss += mm_weight * minmaximize_loss[node.name] / (len(root.nodes_with_children()) if normalize_by_node_count else 1.)
                if not 'MM' in losses_used:
                    losses_used.append('MM')

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
                            breakpoint()
                            pdb.set_trace()
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
                                breakpoint()
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

            if (not pretrain) and cluster_desc:
                TOPK = 1 # float('inf') # 3
                # cluster loss corresponding to every descendant species
                cluster_loss_for_each_descendant = []
                for child_node in node.children:
                    if child_node.is_leaf(): # because leaf nodes do not have any descendants
                        descendant_idx = torch.tensor([name == child_node.name for name in batch_names])

                        prototypes = getattr(net.module, '_'+node.name+'_add_on') # getattr(net.module, '_'+node.name+'_classification')
                        cosine_similarity = torch.abs(functional_UnitConv2D(features, prototypes.weight, prototypes.bias))
                        pooled_cosine_similarity = findCorrespondingToMax(proto_features[node.name], cosine_similarity)
                        
                        descendant_pooled_cs_1, descendant_pooled_cs_2 = pooled_cosine_similarity[descendant_idx].chunk(2)

                        child_class_idx = node.children_to_labels[child_node.name]
                        classification_weights = getattr(net.module, '_'+node.name+'_classification').weight
                        relevant_protos_to_child_node = [proto_idx.item() for proto_idx in torch.nonzero(classification_weights[child_class_idx, :] > 1e-3)]

                        for proto_idx in relevant_protos_to_child_node:
                            num_of_descendants_in_batch = descendant_pooled_cs_1[:, proto_idx].shape[0]
                            if (num_of_descendants_in_batch == 0): continue
                            topk_nearest_patches_1, _ = torch.topk(descendant_pooled_cs_1[:, proto_idx].squeeze(), min(TOPK, num_of_descendants_in_batch))

                            num_of_descendants_in_batch = descendant_pooled_cs_2[:, proto_idx].shape[0]
                            if (num_of_descendants_in_batch == 0): continue
                            topk_nearest_patches_2, _ = torch.topk(descendant_pooled_cs_2[:, proto_idx].squeeze(), min(TOPK, num_of_descendants_in_batch))

                            cluster_loss_for_each_descendant.append((torch.mean(topk_nearest_patches_1) + torch.mean(topk_nearest_patches_2)) / 2.)

                    else:
                        for descendant_name in child_node.leaf_descendents:
                            descendant_idx = torch.tensor([name == descendant_name for name in batch_names])
                            prototypes = getattr(net.module, '_'+node.name+'_add_on') # getattr(net.module, '_'+node.name+'_classification')
                            cosine_similarity = torch.abs(functional_UnitConv2D(features, prototypes.weight, prototypes.bias))
                            pooled_cosine_similarity = findCorrespondingToMax(proto_features[node.name], cosine_similarity)
                            
                            descendant_pooled_cs_1, descendant_pooled_cs_2 = pooled_cosine_similarity[descendant_idx].chunk(2)

                            child_class_idx = node.children_to_labels[child_node.name]
                            classification_weights = getattr(net.module, '_'+node.name+'_classification').weight
                            relevant_protos_to_child_node = [proto_idx.item() for proto_idx in torch.nonzero(classification_weights[child_class_idx, :] > 1e-3)]

                            for proto_idx in relevant_protos_to_child_node:
                                num_of_descendants_in_batch = descendant_pooled_cs_1[:, proto_idx].shape[0]
                                if (num_of_descendants_in_batch == 0): continue
                                topk_nearest_patches_1, _ = torch.topk(descendant_pooled_cs_1[:, proto_idx].squeeze(), min(TOPK, num_of_descendants_in_batch))

                                num_of_descendants_in_batch = descendant_pooled_cs_2[:, proto_idx].shape[0]
                                if (num_of_descendants_in_batch == 0): continue
                                topk_nearest_patches_2, _ = torch.topk(descendant_pooled_cs_2[:, proto_idx].squeeze(), min(TOPK, num_of_descendants_in_batch))

                                cluster_loss_for_each_descendant.append(((torch.mean(topk_nearest_patches_1) + torch.mean(topk_nearest_patches_2)) / 2.) / len(child_node.leaf_descendents))
                
                if len(cluster_loss_for_each_descendant) > 0: # sometimes non of the descendants of the node could be in the batch
                    cluster_desc_loss[node.name] = -torch.sum(torch.stack(cluster_loss_for_each_descendant, dim=0)) / len(node.children)
                    # loss += t_weight * tanh_loss[node.name]
                    loss += cluster_desc_weight * cluster_desc_loss[node.name] / (len(root.nodes_with_children()) if normalize_by_node_count else 1.)
                    if not 'CLUS_DESC' in losses_used:
                        losses_used.append('CLUS_DESC')

            if (not pretrain) and sep_desc:
                TOPK = 1 # float('inf') # 3
                # cluster loss corresponding to every descendant species
                sep_loss_for_each_descendant = []
                for child_node in node.children:
                    for descendant_name in node.leaf_descendents:
                        if descendant_name in child_node.leaf_descendents:
                            continue
                        descendant_idx = torch.tensor([name == descendant_name for name in batch_names])
                        prototypes = getattr(net.module, '_'+node.name+'_add_on') # getattr(net.module, '_'+node.name+'_classification')
                        cosine_similarity = torch.abs(functional_UnitConv2D(features, prototypes.weight, prototypes.bias))
                        pooled_cosine_similarity = findCorrespondingToMax(proto_features[node.name], cosine_similarity)
                        
                        descendant_pooled_cs_1, descendant_pooled_cs_2 = pooled_cosine_similarity[descendant_idx].chunk(2)

                        child_class_idx = node.children_to_labels[child_node.name]
                        classification_weights = getattr(net.module, '_'+node.name+'_classification').weight
                        relevant_protos_to_child_node = [proto_idx.item() for proto_idx in torch.nonzero(classification_weights[child_class_idx, :] > 1e-3)]
                        
                        for proto_idx in relevant_protos_to_child_node:
                            num_of_descendants_in_batch = descendant_pooled_cs_1[:, proto_idx].shape[0]
                            if (num_of_descendants_in_batch == 0): continue
                            topk_nearest_patches_1, _ = torch.topk(descendant_pooled_cs_1[:, proto_idx].squeeze(), min(TOPK, num_of_descendants_in_batch))

                            num_of_descendants_in_batch = descendant_pooled_cs_2[:, proto_idx].shape[0]
                            if (num_of_descendants_in_batch == 0): continue
                            topk_nearest_patches_2, _ = torch.topk(descendant_pooled_cs_2[:, proto_idx].squeeze(), min(TOPK, num_of_descendants_in_batch))

                            sep_loss_for_each_descendant.append(((torch.mean(topk_nearest_patches_1) + torch.mean(topk_nearest_patches_2)) / 2.) / len(child_node.leaf_descendents))
                
                if len(sep_loss_for_each_descendant) > 0: # sometimes non of the descendants of the node could be in the batch
                    sep_desc_loss[node.name] = torch.sum(torch.stack(sep_loss_for_each_descendant, dim=0)) / len(node.children)
                    # loss += t_weight * tanh_loss[node.name]
                    loss += sep_desc_weight * sep_desc_loss[node.name] / (len(root.nodes_with_children()) if normalize_by_node_count else 1.)
                    if not 'SEP_DESC' in losses_used:
                        losses_used.append('SEP_DESC')


            if (not pretrain) and (not finetune) and subspace_sep:
                projection_operators = []
                subspace_sep_loss[node.name] = 0

                prototypes = getattr(net.module, '_'+node.name+'_add_on')
                unit_length_prototypes = F.normalize(prototypes.weight, p=2, dim=(1, 2, 3)) # F.normalize(prototypes.weight, p=2, dim=(1, 2, 3))

                for child_node in node.children:
                    # can also try the zero padding idea instead to handle varying subspace sizes
                    child_class_idx = node.children_to_labels[child_node.name]
                    classification_weights = getattr(net.module, '_'+node.name+'_classification').weight
                    relevant_protos_to_child_node = [proto_idx.item() for proto_idx in torch.nonzero(classification_weights[child_class_idx, :] > 0)]
                    child_node_relevant_prototypes = unit_length_prototypes[relevant_protos_to_child_node] # [num_protos, 768, 1, 1]
                    child_node_relevant_prototypes = child_node_relevant_prototypes.squeeze() # [num_protos, 768]
                    child_node_relevant_prototypes_T = child_node_relevant_prototypes.transpose(0, 1) # [768, num_protos]
                    child_node_projection_operator = torch.matmul(child_node_relevant_prototypes_T, child_node_relevant_prototypes, ) # [768, 768]
                    projection_operators.append(child_node_projection_operator)

                    if not child_node.is_leaf():
                        child_projection_operators = []

                        child_prototypes = getattr(net.module, '_'+child_node.name+'_add_on')
                        unit_length_child_prototypes = F.normalize(child_prototypes.data, p=2, dim=(1, 2, 3))

                        for grand_child_node in child_node.children:
                            # can also try the zero padding idea instead to handle varying subspace sizes
                            grand_child_class_idx = child_node.children_to_labels[grand_child_node.name]
                            child_classification_weights = getattr(net.module, '_'+child_node.name+'_classification').weight
                            relevant_protos_to_grand_child_node = [proto_idx.item() for proto_idx in torch.nonzero(child_classification_weights[grand_child_class_idx, :] > 0)]
                            grand_child_node_relevant_prototypes = unit_length_child_prototypes[relevant_protos_to_grand_child_node] # [num_protos, 768, 1, 1]
                            grand_child_node_relevant_prototypes = grand_child_node_relevant_prototypes.squeeze() # [num_protos, 768]
                            grand_child_node_relevant_prototypes_T = grand_child_node_relevant_prototypes.transpose(0, 1) # [768, num_protos]
                            grand_child_node_projection_operator = torch.matmul(grand_child_node_relevant_prototypes_T, grand_child_node_relevant_prototypes, ) # [768, 768]
                            child_projection_operators.append(grand_child_node_projection_operator)
                        child_projection_operator = torch.stack(child_projection_operators, dim=0) # [num_grand_children, 768, 768]
                        projection_operator_1 = torch.unsqueeze(child_node_projection_operator,dim=0)\
                                                        .unsqueeze(child_node_projection_operator,dim=0)#[1,1,768,768]
                        projection_operator_2 = torch.unsqueeze(child_projection_operator, dim=0)#[1,num_grand_children,768,768]
                        child_to_grand_child_distance = torch.norm(projection_operator_1-projection_operator_2+1e-10,p='fro',dim=[2,3]) # [1,num_grand_children,768,768] -> [1,num_grand_children]
                        subspace_sep_loss[node.name] += -(torch.norm(child_to_grand_child_distance,p=1,dim=[0,1],dtype=torch.double) / \
                                                                    torch.sqrt(torch.tensor(2,dtype=torch.double)).cuda()) / \
                                                                    len(node.children)

                projection_operator = torch.stack(projection_operators, dim=0) # [num_children, 768, 768]
                projection_operator_1 = torch.unsqueeze(projection_operator,dim=1)#[num_children,1,768,768]
                projection_operator_2 = torch.unsqueeze(projection_operator, dim=0)#[1,num_children,768,768]
                pairwise_distance =  torch.norm(projection_operator_1-projection_operator_2+1e-10,p='fro',dim=[2,3]) #[num_children,num_children,768,768]->[num_children,num_children]
                subspace_sep_loss[node.name] += -(0.5 * torch.norm(pairwise_distance,p=1,dim=[0,1],dtype=torch.double) / \
                                                torch.sqrt(torch.tensor(2,dtype=torch.double)).to(device)) / \
                                                len(node.children)

                loss += subspace_sep_weight * subspace_sep_loss[node.name] / (len(root.nodes_with_children()) if normalize_by_node_count else 1.)
                if not 'SS' in losses_used:
                    losses_used.append('SS')

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

            if (not pretrain) and ('OOD_ent' in args) and ('y' in args.OOD_ent):
                OOD_ent_loss_weight =  float(args.OOD_ent.split('|')[1])
                not_children_idx = torch.tensor([name not in node.leaf_descendents for name in batch_names]) # includes OOD images as well as images belonging to other nodes
                if not_children_idx.sum().item() > 0:
                    OOD_logits = out[node.name][not_children_idx] # [sum(not_children_idx), node.num_children()]
                    OOD_prob = F.softmax(torch.log1p(OOD_logits**2), dim=1)
                    OOD_ent_loss[node.name] = entropy_loss(OOD_prob)
                    # print(OOD_ent_loss[node.name])
                    # breakpoint()
                    loss += OOD_ent_loss_weight * OOD_ent_loss[node.name] / (len(root.nodes_with_children()) if normalize_by_node_count else 1.)
                    if not 'OOD_ENT' in losses_used:
                        losses_used.append('OOD_ENT')
        
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

            # For debugging purpose
            node_accuracy[node.name]['n_examples'] += node_y.shape[0]
            _, node_coarsest_predicted = torch.max(node_logits.data, 1)
            node_accuracy[node.name]['n_correct'] += (node_y == node_coarsest_predicted).sum().item()
            for child in node.children:
                node_accuracy[node.name]['children'][child.name]['n_examples'] += (node_y == node.children_to_labels[child.name]).sum().item()
                node_accuracy[node.name]['children'][child.name]['n_correct'] += (node_coarsest_predicted[node_y == node.children_to_labels[child.name]] == node.children_to_labels[child.name]).sum().item()
                node_accuracy[node.name]['preds'] = torch.cat((node_accuracy[node.name]['preds'], node_logits.detach().cpu())).detach().cpu()#.numpy()
                node_accuracy[node.name]['gts'] = torch.cat((node_accuracy[node.name]['gts'], node_y.detach().cpu())).detach().cpu()#.numpy()

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
                    
                if len(kernel_orth_loss) > 0:
                    avg_kernel_orth_loss = np.mean([node_kernel_orth_loss.item() for node_name, node_kernel_orth_loss in kernel_orth_loss.items()])
                else:
                    avg_kernel_orth_loss = -5 # placeholder value

                avg_class_loss = None
                avg_OOD_loss = None

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



base_architecture_to_features = {'resnet18': resnet18_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 'resnet50_inat': resnet50_features_inat,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features,
                                 'convnext_tiny_26': convnext_tiny_26_features,
                                 'convnext_tiny_13': convnext_tiny_13_features,
                                 'convnext_tiny_7': convnext_tiny_7_features}

# adapted from https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
class NonNegLinear(nn.Module):
    """Applies a linear transformation to the incoming data with non-negative weights`
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(NonNegLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        torch.nn.init.normal_(self.weight, mean=1.0,std=0.1)

        self.normalization_multiplier = nn.Parameter(torch.ones((1,),requires_grad=True))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
            torch.nn.init.constant_(self.bias, val=0.)
        else:
            self.register_parameter('bias', None)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input,torch.relu(self.weight), self.bias)
    
class Linear(nn.Module):
    """Applies a linear transformation to the incoming data with non-negative weights`
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        torch.nn.init.normal_(self.weight, mean=1.0,std=0.1)

        self.normalization_multiplier = nn.Parameter(torch.ones((1,),requires_grad=True))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
            torch.nn.init.constant_(self.bias, val=0.)
        else:
            self.register_parameter('bias', None)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input,self.weight, self.bias)

import torch
import torch.nn as nn
import torch.nn.functional as F

class UnitConv2D(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros'):
        super(UnitConv2D, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

    def forward(self, input):
        normalized_weight = F.normalize(self.weight.data, p=2, dim=(1, 2, 3)) # Normalize the kernels to unit vectors
        normalized_input = F.normalize(input, p=2, dim=1) # Normalize the input to unit vectors

        if self.bias is not None:
            normalized_bias = F.normalize(self.bias.data, p=2, dim=0) # Normalize the kernels to unit vectors
        else:
            normalized_bias = None
        return self._conv_forward(normalized_input, normalized_weight, normalized_bias)
    
class L2Conv2D(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros'):
        if bias:
            raise Exception('Do not use bias for l2conv2d')
        super(L2Conv2D, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.ones = nn.Parameter(torch.ones(self.weight.data.shape), requires_grad=False)

    def distance_2_similarity(self, distances):
        # if self.prototype_activation_function == 'log':
        return torch.log((distances + 1) / (distances + 1e-4))
        # elif self.prototype_activation_function == 'linear':
        #     return -distances
        # else:
        #     return self.prototype_activation_function(distances)

    def forward(self, input):
        x = input
        x2 = x ** 2
        x2_patch_sum = F.conv2d(input=x2, weight=self.ones)
        p2 = self.weight.data ** 2
        p2 = torch.sum(p2, dim=(1, 2, 3))
        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1, 1)
        p2_reshape = p2.unsqueeze(-1).unsqueeze(-1)
        # xp = F.conv2d(input=x, weight=self.prototype_vectors) 
        xp = self._conv_forward(x, self.weight.data, None) # bias directly set to None
        intermediate_result = - 2 * xp + p2_reshape  # use broadcast
        distances = F.relu(x2_patch_sum + intermediate_result)
        return self.distance_2_similarity(distances)

class ProjectConv2D(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros'):
        super(ProjectConv2D, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

    def forward(self, input):
        normalized_weight = F.normalize(self.weight.data, p=2, dim=(1, 2, 3)) # Normalize the kernels to unit vectors

        if self.bias is not None:
            normalized_bias = F.normalize(self.bias.data, p=2, dim=0) # Normalize the kernels to unit vectors
        else:
            normalized_bias = None
        return self._conv_forward(input, normalized_weight, normalized_bias)

class DinoV2(torch.nn.Module):

    def __init__(self, model_name, latent_shape):
        super(DinoV2, self).__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', model_name).cuda()
        self.latent_shape = latent_shape

    def forward(self, x):
        out = self.model(x, is_training=True)['x_norm_patchtokens']
        out = out.view(out.shape[0], self.latent_shape[0], self.latent_shape[1], out.shape[2])
        out = out.permute(0, 3, 1, 2)
        return out

def get_network(num_classes: int, args: argparse.Namespace, root=None): 

    if 'dinov2_vits14' in args.net:
        features = DinoV2(model_name=args.net, latent_shape=(int(args.image_size/14), int(args.image_size/14)))
        first_add_on_layer_in_channels = 384
        # if args.num_features == 0:
        #     raise Exception('Do not set num_features to 0 for dinov2')

        if args.basic_cnext_gaussian_multiplier != '':
            raise NotImplementedError

        if args.stage4_reducer_net != '':
            raise NotImplementedError

    else:
        features = base_architecture_to_features[args.net](pretrained=not args.disable_pretrained)
        features_name = str(features).upper()
        if 'next' in args.net:
            features_name = str(args.net).upper()
        if features_name.startswith('RES') or features_name.startswith('CONVNEXT'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
        else:
            raise Exception('other base architecture NOT implemented')

    if args.basic_cnext_gaussian_multiplier != '':
        stages = args.basic_cnext_gaussian_multiplier.split('|')[0]
        stages = [int(s) for s in stages.split(',')]
        sigma = float(args.basic_cnext_gaussian_multiplier.split('|')[1])
        factor = int(args.basic_cnext_gaussian_multiplier.split('|')[2])
        for stage in stages:
            apply_gaussian_multiplier_to_convnext_stage(features, stage=stage, sigma=sigma, factor=factor, device='cuda')
    
    if args.stage4_reducer_net != '':
        layer_infos = args.stage4_reducer_net.split('|')
        reducer_layers = [('backbone', features)]
        for i, layer_info in enumerate(layer_infos):
            in_channels = int(layer_info.split(',')[0])
            out_channels = int(layer_info.split(',')[1])
            reducer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, \
                        kernel_size=1, stride = 1, padding=0, bias=True)
            reducer_layers.append(('stage4_reducer_'+str(i)+'_conv', reducer))
            print('Added reducer', 'stage4_reducer_'+str(i)+'_conv', in_channels, 'to', out_channels)
            if (len(layer_info.split(',')) > 2):
                if layer_info.split(',')[2] == 'gelu':
                    reducer_layers.append(('stage4_reducer_'+str(i)+'_gelu', nn.GELU()))
                    print('Added reducer', 'stage4_reducer_'+str(i)+'_gelu')
        
        features = nn.Sequential(OrderedDict(reducer_layers))
        first_add_on_layer_in_channels = out_channels

    if args.num_features == 0:
        num_prototypes = first_add_on_layer_in_channels
    else:
        num_prototypes = args.num_features
    print("Number of prototypes: ", num_prototypes, flush=True)
        
    parent_nodes = root.nodes_with_children()
    add_on_layers = {}
    print((10*'-')+f'Prototypes per descendant: {args.num_protos_per_descendant}'+(10*'-'))
    for node in parent_nodes:
        if args.unitconv2d == 'y' and args.projectconv2d == 'y':
            raise Exception('Do not set both unitconv2d and projectconv2d to y')
        if args.unitconv2d == 'y':
            add_on_layers[node.name] = UnitConv2D(in_channels=first_add_on_layer_in_channels, out_channels=node.num_protos, \
                                                kernel_size=1, stride = 1, padding=0, bias=True if args.add_on_bias else False) # is bias required ??, prev True now set to False for unit length conv2d
        elif args.projectconv2d == 'y':
            add_on_layers[node.name] = ProjectConv2D(in_channels=first_add_on_layer_in_channels, out_channels=node.num_protos, \
                                                kernel_size=1, stride = 1, padding=0, bias=True if args.add_on_bias else False) # is bias required ??, prev True now set to False for unit length conv2d
        elif args.l2conv2d == 'y':
            add_on_layers[node.name] = L2Conv2D(in_channels=first_add_on_layer_in_channels, out_channels=node.num_protos, \
                                                kernel_size=1, stride = 1, padding=0, bias=True if args.add_on_bias else False)
        else:
            add_on_layers[node.name] = nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=node.num_protos, \
                                                kernel_size=1, stride = 1, padding=0, bias=True if args.add_on_bias else False) # is bias required ??, prev True now set to False for unit length conv2d
        # print(f'Assigned {node.num_protos} protos to node {node.name}')

        # for child_node in node.children:
        #     add_on_layers[node.name+'_'+child_node.name] = UnitConv2D(in_channels=first_add_on_layer_in_channels, \
        #                                                                 out_channels=node.num_protos_per_child[child_node.name], \
        #                                                                 kernel_size=1, stride = 1, padding=0, bias=False) # is bias required ??, prev True now set to False for unit length conv2d
        #     print(f'Assigned {node.num_protos_per_child[child_node.name]} protos to child {child_node.name} of node {node.name}')


    # add_on_layers = nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=num_prototypes * len(parent_nodes), kernel_size=1, stride = 1, padding=0, bias=True)

    pool_layer = nn.Sequential(
                nn.AdaptiveMaxPool2d(output_size=(1,1)), #outputs (bs, ps,1,1)
                nn.Flatten() #outputs (bs, ps)
                ) 

    if args.bias:
        print('--------- Using bias in Classification layer ---------')

    classification_layers = {}
    for node in parent_nodes:
        if ('classifier' in args) and (args.classifier == 'Linear'):
            classification_layers[node.name] = Linear(node.num_protos, node.num_children(), bias=True if args.bias else False)
        else:
            classification_layers[node.name] = NonNegLinear(node.num_protos, node.num_children(), bias=True if args.bias else False)
        
        if args.protopool == 'n':
            classification_layers[node.name].weight.requires_grad = False
            start_idx = 0
            # element-wise disabling gradient is not possible
            # so setting the values to negative so during training they will become zero because of relu
            # and once relu becomes zero its gradients also become zero and they wont get trained
            for child_node in node.children:
                child_label = node.children_to_labels[child_node.name]
                end_idx = start_idx + node.num_protos_per_child[child_node.name]
                classification_layers[node.name].weight[child_label, :start_idx] = -0.5
                classification_layers[node.name].weight[child_label, end_idx:] = -0.5
                start_idx = end_idx

            classification_layers[node.name].weight.requires_grad = True

            # if node.name == '144+147':
            #     breakpoint()

        # for child_node in node.children:
        #     classification_layers[node.name+'_'+child_node.name] = NonNegLinear(node.num_protos_per_child[child_node.name], \
        #                                                                         1, bias=True if args.bias else False)
        
        
    return features, add_on_layers, pool_layer, classification_layers, num_prototypes


    