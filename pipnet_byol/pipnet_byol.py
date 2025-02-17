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
from copy import deepcopy
# from pipnet.pipnet import UnitConv2D, ProjectConv2D

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


class PIPNetBYOL(nn.Module):
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

        MLP_HIDDEN_SIZE = 3072

        assert num_classes > 0
        self._num_features = args.num_features
        self._num_classes = num_classes
        # self._num_prototypes = num_prototypes # this is only the minimum number of protos per node, might vary for each node
        self._net = feature_net

        self._projector = nn.Sequential(
                            nn.Conv2d(in_channels=add_on_layers['root'].in_channels, out_channels=MLP_HIDDEN_SIZE, kernel_size=1, stride = 1, padding=0, bias=True),
                            nn.BatchNorm2d(MLP_HIDDEN_SIZE, eps=1e-5, momentum=0.1),
                            nn.ReLU(),
                            nn.Conv2d(in_channels=MLP_HIDDEN_SIZE, out_channels=add_on_layers['root'].in_channels, kernel_size=1, stride = 1, padding=0, bias=True),
                        )
        self._predictor = nn.Sequential(
                            nn.Conv2d(in_channels=add_on_layers['root'].in_channels, out_channels=MLP_HIDDEN_SIZE, kernel_size=1, stride = 1, padding=0, bias=True),
                            nn.BatchNorm2d(MLP_HIDDEN_SIZE, eps=1e-5, momentum=0.1),
                            nn.ReLU(),
                            nn.Conv2d(in_channels=MLP_HIDDEN_SIZE, out_channels=add_on_layers['root'].in_channels, kernel_size=1, stride = 1, padding=0, bias=True),
                        )
        # self._target_net = nn.Sequential(
        #                     feature_net.clone(),
        #                     self._projector.clone()
        #                 )
        self._target_feature_net = deepcopy(self._net)#.clone()
        self._target_projector = deepcopy(self._projector)#.clone()
        self._target_feature_net.requires_grad = False
        self._target_projector.requires_grad = False

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

        self.args = args

        if (args.multiply_cs_softmax == 'y') and not (args.softmax.split('|')[0] == 'y' or args.gumbel_softmax == 'y'):
            raise Exception('Use either softmax or gumbel softmax when using multiply_cs_softmax')

    
    def forward(self, xs, inference=False):
        features = self._net(xs) 

        if not inference:
            online_network_out = self._predictor(self._projector(features))
            target_network_out = self._target_projector(self._target_feature_net(xs))

        proto_features = {}
        proto_features_cs = {}
        proto_features_softmaxed = {}
        pooled = {}
        out = {}

        # if self.args.sg_before_protos == 'y':
        #     proto_layer_input_features = features.clone().detach()
        # else:
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

            if inference:
                pooled[node.name] = torch.where(pooled[node.name] < 0.1, 0., pooled[node.name])  #during inference, ignore all prototypes that have 0.1 similarity or lower
                
            out[node.name] = getattr(self, '_'+node.name+'_classification')(pooled[node.name]) #shape (bs*2, num_classes) # these are logits

        # self._root._classification.weight.requires_grad 
        # getattr(self, '_'+'root'+'_classification').weight.requires_grad
        # pooled['root'].requires_grad
        # proto_features['root'].requires_grad
        # getattr(self, '_'+'root'+'_add_on').weight.requires_grad
        # features.requires_grad

        if inference:
            return features, proto_features, pooled, out
        else:
            return online_network_out, target_network_out, features, proto_features, pooled, out
    
    def get_joint_distribution(self, out, device='cuda'):
        batch_size = out['root'].size(0)
        #top_level = torch.nn.functional.softmax(self.root.logits,1)            
        top_level = out['root']
        bottom_level = self.root.distribution_over_furthest_descendents(batch_size, out)    
        names = self.root.unwrap_names_of_joint(self.root.names_of_joint_distribution())
        idx = np.argsort(names)
        bottom_level = bottom_level[:,idx]        
        return top_level, bottom_level
    
    def get_classification_layers(self):
        return [getattr(self, attr) for attr in dir(self) if attr.endswith('_classification')] 


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

def get_network(num_classes: int, args: argparse.Namespace, root=None): 
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
        if args.unitconv2d == 'y':
            add_on_layers[node.name] = UnitConv2D(in_channels=first_add_on_layer_in_channels, out_channels=node.num_protos, \
                                                kernel_size=1, stride = 1, padding=0, bias=True if args.add_on_bias else False) # is bias required ??, prev True now set to False for unit length conv2d
        else:
            add_on_layers[node.name] = nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=node.num_protos, \
                                                kernel_size=1, stride = 1, padding=0, bias=True if args.add_on_bias else False) # is bias required ??, prev True now set to False for unit length conv2d
        print(f'Assigned {node.num_protos} protos to node {node.name}')

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

    classification_layers = {}
    for node in parent_nodes:
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
                classification_layers[node.name].weight[child_label, :start_idx] = -1.0
                classification_layers[node.name].weight[child_label, end_idx:] = -1.0
                start_idx = end_idx

            classification_layers[node.name].weight.requires_grad = True

        # for child_node in node.children:
        #     classification_layers[node.name+'_'+child_node.name] = NonNegLinear(node.num_protos_per_child[child_node.name], \
        #                                                                         1, bias=True if args.bias else False)
        
        
    return features, add_on_layers, pool_layer, classification_layers, num_prototypes


    