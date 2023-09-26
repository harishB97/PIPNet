import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from features.resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet50_features_inat, resnet101_features, resnet152_features
from features.convnext_features import convnext_tiny_26_features, convnext_tiny_13_features 
import torch
from torch import Tensor
from util.node import Node
import numpy as np
from collections import defaultdict

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
        self._num_features = args.num_features
        self._num_classes = num_classes
        # self._num_prototypes = num_prototypes # this is only the minimum number of protos per node, might vary for each node
        self._net = feature_net
        # self._add_on = add_on_layers
        for node_and_child_name in add_on_layers:
            setattr(self, '_'+node_and_child_name+'_add_on', add_on_layers[node_and_child_name])
            setattr(self, '_'+node_and_child_name+'_num_protos', add_on_layers[node_and_child_name].weight.shape[0])
        self._pool = pool_layer
        for node_and_child_name in classification_layers:
            setattr(self, '_'+node_and_child_name+'_classification', classification_layers[node_and_child_name])
        # self._classification = classification_layers
        self._multiplier = nn.Parameter(torch.ones((1,),requires_grad=True)) # this can directly be set to 2.0 and requires_grad=False, not sure why its not done
        # self._multiplier = classification_layers.normalization_multiplier 
        self._softmax = nn.Softmax(dim=1)
        self._num_parent_nodes = num_parent_nodes # I dont remember why this was added
        self.root = root

    
    def forward(self, xs,  inference=False):
        # features = self._net(xs) 
        # proto_features = {}
        # pooled = {}
        # out = {}
        # for node in self.root.nodes_with_children():
        #     proto_features[node.name] = getattr(self, '_'+node.name+'_add_on')(features)
        #     proto_features[node.name] = self._softmax(proto_features[node.name])
        #     pooled[node.name] = self._pool(proto_features[node.name])
        #     if inference:
        #         pooled[node.name] = torch.where(pooled[node.name] < 0.1, 0., pooled[node.name])  #during inference, ignore all prototypes that have 0.1 similarity or lower
        #     out[node.name] = getattr(self, '_'+node.name+'_classification')(pooled[node.name]) #shape (bs*2, num_classes) # these are logits

        features = self._net(xs) 
        proto_features = defaultdict(dict)
        proto_features_softmaxed = dict()
        pooled = defaultdict(dict)
        out = dict() #defaultdict(dict)
        for node in self.root.nodes_with_children():
            for child_node in node.children:
                proto_features[node.name][child_node.name] = getattr(self, '_'+node.name+'_'+child_node.name+'_add_on')(features)
            proto_features_concatenated = torch.cat([proto_features[node.name][child_node.name] for child_node in node.children], 1) # 1 is the channel dimension
            proto_features_softmaxed[node.name] = self._softmax(proto_features_concatenated)
            proto_features_split = torch.split(proto_features_softmaxed[node.name], [node.num_protos_per_child[child_node.name] for child_node in node.children], dim=1)

            for idx, child_node in enumerate(node.children):
                proto_features[node.name][child_node.name] = proto_features_split[idx]

            for child_node in node.children:
                pooled[node.name][child_node.name] = self._pool(proto_features[node.name][child_node.name])
                if inference:
                    #during inference, ignore all prototypes that have 0.1 similarity or lower
                    pooled[node.name][child_node.name] = torch.where(pooled[node.name][child_node.name] < 0.1, 0., pooled[node.name][child_node.name])  

            each_class_logit = []
            for child_node in node.children:
                each_class_logit.append(getattr(self, '_'+node.name+'_'+child_node.name+'_classification')(pooled[node.name][child_node.name])) #shape (bs*2, 1)
            out[node.name] = torch.cat(each_class_logit, 1) #shape (bs*2, num_classes)

        # only pooled here is dict of dict because for applying tanh loss the "pooled" vector for each child should be seperate
        return proto_features_softmaxed, pooled, out
    
    def get_joint_distribution(self, out, device='cuda'):
        
        batch_size = out['root'].size(0)

        #top_level = torch.nn.functional.softmax(self.root.logits,1)            
        top_level = out['root']
        bottom_level = self.root.distribution_over_furthest_descendents(batch_size, out)    

        names = self.root.unwrap_names_of_joint(self.root.names_of_joint_distribution())
        idx = np.argsort(names)

        bottom_level = bottom_level[:,idx]        
        
        return top_level, bottom_level
    
    # def forward_backbone(self, xs,  inference=False):
    #     features = self._net(xs) 
    #     proto_features_all_nodes = self._add_on(features)
    #     proto_features = {}
    #     pooled = {}
    #     out = {}
    #     for i, node in enumerate(self.root.nodes_with_children()):
    #         proto_features_i = proto_features_all_nodes[:, i*self._num_prototypes:(i+1)*self._num_prototypes, :, :]
    #         proto_features_i = self._softmax(proto_features_i)
    #         pooled_i = self._pool(proto_features_i)
    #         if inference:
    #             pooled_i = torch.where(pooled_i < 0.1, 0., pooled_i)  #during inference, ignore all prototypes that have 0.1 similarity or lower
    #         out_i = getattr(self, '_'+node.name+'_classification')(pooled_i) #shape (bs*2, num_classes) 
    #         proto_features[node.name] = proto_features_i
    #         pooled[node.name] = pooled_i
    #         out[node.name] = out_i

    #     return proto_features, pooled, out
    
    def get_classification_layers(self):
        return [getattr(self, attr) for attr in dir(self) if attr.endswith('_classification')] 


base_architecture_to_features = {'resnet18': resnet18_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 'resnet50_inat': resnet50_features_inat,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features,
                                 'convnext_tiny_26': convnext_tiny_26_features,
                                 'convnext_tiny_13': convnext_tiny_13_features}

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
        self.normalization_multiplier = nn.Parameter(torch.ones((1,),requires_grad=True))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input,torch.relu(self.weight), self.bias)


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
    
    # original
    # if args.num_features == 0:
    #     num_prototypes = first_add_on_layer_in_channels
    #     print("Number of prototypes: ", num_prototypes, flush=True)
    #     add_on_layers = nn.Sequential(
    #         nn.Softmax(dim=1), #softmax over every prototype for each patch, such that for every location in image, sum over prototypes is 1                
    # )
    # else:
    #     num_prototypes = args.num_features
    #     print("Number of prototypes set from", first_add_on_layer_in_channels, "to", num_prototypes,". Extra 1x1 conv layer added. Not recommended.", flush=True)
    #     add_on_layers = nn.Sequential(
    #         nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=num_prototypes, kernel_size=1, stride = 1, padding=0, bias=True), 
    #         nn.Softmax(dim=1), #softmax over every prototype for each patch, such that for every location in image, sum over prototypes is 1                
    # )
        
    if args.num_features == 0:
        num_prototypes = first_add_on_layer_in_channels
    else:
        num_prototypes = args.num_features
    print("Number of prototypes: ", num_prototypes, flush=True)

    # parent_nodes = root.nodes_with_children()
    # add_on_layers = {}
    # for node in parent_nodes:
    #     add_on_layers[node.name] = nn.Sequential(
    #                                             nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=num_prototypes, kernel_size=1, stride = 1, padding=0, bias=True), 
    #                                             nn.Softmax(dim=1), #softmax over every prototype for each patch, such that for every location in image, sum over prototypes is 1                
    #                                         )
        
    parent_nodes = root.nodes_with_children()
    add_on_layers = {}
    # prototypes_per_descendant = args.num_protos_per_descendant
    # change 0 to num_prototypes for having minimum num of protos at any node
    print((10*'-')+f'Prototypes per descendant: {args.num_protos_per_descendant}'+(10*'-'))
    for node in parent_nodes:
        # add_on_layers[node.name] = nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=max(20, (node.num_descendents() * prototypes_per_descendant)), \
        #                                      kernel_size=1, stride = 1, padding=0, bias=True)
        # proto_count = max(0, (node.num_descendents() * prototypes_per_descendant))
        # print(f'Assigned {proto_count} protos to node {node.name}')

        for child_node in node.children:
            add_on_layers[node.name+'_'+child_node.name] = nn.Conv2d(in_channels=first_add_on_layer_in_channels, \
                                                                        out_channels=node.num_protos_per_child[child_node.name], \
                                                                        kernel_size=1, stride = 1, padding=0, bias=True) # is bias required ??
            print(f'Assigned {node.num_protos_per_child[child_node.name]} protos to child {child_node.name} of node {node.name}')


    # add_on_layers = nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=num_prototypes * len(parent_nodes), kernel_size=1, stride = 1, padding=0, bias=True)

    pool_layer = nn.Sequential(
                nn.AdaptiveMaxPool2d(output_size=(1,1)), #outputs (bs, ps,1,1)
                nn.Flatten() #outputs (bs, ps)
                ) 
    
    # if args.bias:
    #     classification_layer = NonNegLinear(num_prototypes, num_classes, bias=True)
    # else:
    #     classification_layer = NonNegLinear(num_prototypes, num_classes, bias=False)

    classification_layers = {}
    for node in parent_nodes:
        # classification_layers[node.name] = NonNegLinear(max(20, (node.num_descendents() * prototypes_per_descendant)), \
        #                                                 node.num_children(), bias=True if args.bias else False)
        for child_node in node.children:
            classification_layers[node.name+'_'+child_node.name] = NonNegLinear(node.num_protos_per_child[child_node.name], \
                                                                                1, bias=True if args.bias else False)
        
        
    return features, add_on_layers, pool_layer, classification_layers, num_prototypes


    