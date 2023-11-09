import torch.nn as nn
import torch

import sys, os
import random
import numpy as np
from shutil import copy
import matplotlib.pyplot as plt
from copy import deepcopy

from omegaconf import OmegaConf
import shutil
import pickle
import random
from tqdm import tqdm
from util.data import ModifiedLabelLoader
from collections import defaultdict
import heapq
import pdb
from util.vis_pipnet import get_img_coordinates
import torchvision.transforms as transforms
from PIL import ImageFont, Image, ImageDraw as D
import torchvision
import torch.nn.functional as F

from pipnet.pipnet import PIPNet, get_network
from util.log import Log
from util.args import get_args, save_args, get_optimizer_nn
from util.data import get_dataloaders
from util.func import init_weights_xavier
from pipnet.train import train_pipnet, test_pipnet
# from pipnet.test import eval_pipnet, get_thresholds, eval_ood
from util.eval_cub_csv import eval_prototypes_cub_parts_csv, get_topk_cub, get_proto_patches_cub
from util.vis_pipnet import visualize, visualize_topk
from util.visualize_prediction import vis_pred, vis_pred_experiments
from util.node import Node
from util.phylo_utils import construct_phylo_tree, construct_discretized_phylo_tree
from util.func import get_patch_size
from util.data import ModifiedLabelLoader

def functional_UnitConv2D(in_features, weight, bias, stride = 1, padding=0):
    normalized_weight = F.normalize(weight.data, p=2, dim=(1, 2, 3)) # Normalize the kernels to unit vectors
    normalized_input = F.normalize(in_features, p=2, dim=1) # Normalize the input to unit vectors
    if bias is not None:
        normalized_bias = F.normalize(bias.data, p=2, dim=0) # Normalize the kernels to unit vectors
    else:
        normalized_bias = None
    return F.conv2d(normalized_input, normalized_weight, normalized_bias, stride=stride, padding=padding)

def findCorrespondingToMax(base, target):
    output, indices = F.max_pool2d(base, kernel_size=(26, 26), return_indices=True)# these are logits
    tensor_flattened = target.view(target.shape[0], target.shape[1], -1)
    indices_flattened = indices.view(target.shape[0], target.shape[1], -1)
    corresponding_values_in_target = torch.gather(tensor_flattened, 2, indices_flattened)
    corresponding_values_in_target = corresponding_values_in_target.view(target.shape[0],\
                                     target.shape[1], 1, 1)
    pooled_target = corresponding_values_in_target
    return pooled_target

def customForwardWithCSandSoftmax(net, xs,  inference=False):
    features = net.module._net(xs) 
    proto_features = {}
    proto_features_cs = {}
    proto_features_softmaxed = {}
    pooled = {}
    pooled_cs = {}
    pooled_softmaxed = {}
    out = {}
    for node in net.module.root.nodes_with_children():
        # this may or may not be cosine similarity based on UniConv2D or Conv2d
        proto_features[node.name] = getattr(net.module, '_'+node.name+'_add_on')(features)
        
        #calculating cosine similarity
        prototypes = getattr(net.module, '_'+node.name+'_add_on')
        proto_features_cs[node.name] = functional_UnitConv2D(features, prototypes.weight, prototypes.bias)

        if net.module.args.softmax == 'y':
            proto_features_softmaxed[node.name] = net.module._softmax(proto_features[node.name])
            proto_features[node.name] = proto_features_softmaxed[node.name] # will be overwritten if args.multiply_cs_softmax == 'y'
        elif net.module.args.gumbel_softmax == 'y':
            proto_features_softmaxed[node.name] = net.module._gumbel_softmax(proto_features[node.name])
            proto_features[node.name] = proto_features_softmaxed[node.name] # will be overwritten if args.multiply_cs_softmax == 'y'

        if net.module.args.multiply_cs_softmax == 'y':
            proto_features[node.name] = proto_features_cs[node.name] * proto_features_softmaxed[node.name]
        pooled[node.name] = net.module._pool(proto_features[node.name])
        
        # this could be softmax or cosine similarity
        pooled_cs[node.name] = findCorrespondingToMax(base=proto_features[node.name], \
                                                     target=proto_features_cs[node.name])
        
        pooled_softmaxed[node.name] = findCorrespondingToMax(base=proto_features[node.name], \
                                                     target=proto_features_softmaxed[node.name])

        if inference:
            pooled[node.name] = torch.where(pooled[node.name] < 0.1, 0., pooled[node.name])  #during inference, ignore all prototypes that have 0.1 similarity or lower
        out[node.name] = getattr(net.module, '_'+node.name+'_classification')(pooled[node.name]) #shape (bs*2, num_classes) # these are logits

    return features, proto_features, pooled, pooled_cs, pooled_softmaxed, out


from PIL import Image
import numpy as np
import pdb

def get_heatmap(latent_activation, input_image):
    image_a = latent_activation.cpu().numpy()
    image_a = (image_a - image_a.min()) / (image_a.max() - image_a.min())

    input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())
    image_b = input_image.permute(1, 2, 0).cpu().numpy()
    
    reshaped_image_a = np.array(Image.fromarray((image_a[0] * 255).astype('uint8')).resize((input_image.shape[-1], input_image.shape[-1])))
    normalized_heatmap = (reshaped_image_a - np.min(reshaped_image_a)) / (np.max(reshaped_image_a) - np.min(reshaped_image_a))
    
    heatmap_colormap = plt.get_cmap('jet')
    heatmap_colored = heatmap_colormap(normalized_heatmap)
    
    heatmap_colored_uint8 = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
    image_a_heatmap_pillow = Image.fromarray(heatmap_colored_uint8)
    image_b_pillow = Image.fromarray((image_b * 255).astype('uint8'))
    
    result_image = Image.blend(image_b_pillow, image_a_heatmap_pillow, alpha=0.3)
    
    return np.array(result_image)


def get_heatmap_uninterpolated(latent_activation, input_image):
    image_a = latent_activation.cpu().numpy()
    image_a = (image_a - image_a.min()) / (image_a.max() - image_a.min())

    input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())
    image_b = input_image.permute(1, 2, 0).cpu().numpy()
    
    reshaped_image_a = np.array(Image.fromarray((image_a[0] * 255).astype('uint8')).resize((input_image.shape[-1], input_image.shape[-1]), \
                                                                                          resample=Image.NEAREST ))
    normalized_heatmap = (reshaped_image_a - np.min(reshaped_image_a)) / (np.max(reshaped_image_a) - np.min(reshaped_image_a))
    
    heatmap_colormap = plt.get_cmap('jet')
    heatmap_colored = heatmap_colormap(normalized_heatmap)
    
    heatmap_colored_uint8 = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
    image_a_heatmap_pillow = Image.fromarray(heatmap_colored_uint8)
    image_b_pillow = Image.fromarray((image_b * 255).astype('uint8'))
    
    result_image = Image.blend(image_b_pillow, image_a_heatmap_pillow, alpha=0.3)
    ## Load Model
    return np.array(result_image)


def get_heap():
    list_ = []
    heapq.heapify(list_)
    return list_

def save_images_topk(args, dataloader, net, root, save_path, topk=10, find_non_descendants=False, device='cpu'):

    save_images = True
    font = ImageFont.truetype("arial.ttf", 50)
    font2 = ImageFont.truetype("arial.ttf", 20)
    patchsize, skip = get_patch_size(args)

    for node in root.nodes_with_children():
    #     if node.name == 'root':
    #         continue
        non_leaf_children_names = [child.name for child in node.children if not child.is_leaf()]
        if len(non_leaf_children_names) == 0: # if all the children are leaf nodes then skip this node
            continue

        name2label = dataloader.dataset.class_to_idx # param
        label2name = {label:name for name, label in name2label.items()}
        modifiedLabelLoader = ModifiedLabelLoader(dataloader, node)
        coarse_label2name = modifiedLabelLoader.modifiedlabel2name
        node_label_to_children = {label: name for name, label in node.children_to_labels.items()}
        
        imgs = modifiedLabelLoader.filtered_imgs

        img_iter = tqdm(enumerate(modifiedLabelLoader),
                        total=len(modifiedLabelLoader),
                        mininterval=50.,
                        desc='Collecting topk',
                        ncols=0)

        classification_weights = getattr(net.module, '_'+node.name+'_classification').weight
        
        # maps proto_number -> grand_child_name (or descendant leaf name) -> list of top-k activations
        proto_mean_activations = defaultdict(lambda: defaultdict(get_heap))

        # maps class names to the prototypes that belong to that
        class_and_prototypes = defaultdict(set)

        for i, (xs, orig_y, ys) in img_iter:
            if not find_non_descendants: 
                # do only when finding descendants
                if coarse_label2name[ys.item()] not in non_leaf_children_names:
                    continue

            xs, ys = xs.to(device), ys.to(device)

            with torch.no_grad():
                model_output = customForwardWithCSandSoftmax(net, xs, inference=False)
                _, softmaxes, pooled, pooled_ip, pooled_softmax, _ = model_output
    #             model_output = net(xs, inference=False)
    #             if len(model_output) == 3:
    #                 softmaxes, pooled, _ = model_output
    #             elif len(model_output) == 4:
    #                 _, softmaxes, pooled, _ = model_output
                pooled = pooled[node.name].squeeze(0) 
                pooled_ip = pooled_ip[node.name].squeeze(0) 
                softmaxes = softmaxes[node.name]#.squeeze(0)

                for p in range(pooled.shape[0]): # pooled.shape -> [768] (== num of prototypes)
                    c_weight = torch.max(classification_weights[:,p]) # classification_weights[:,p].shape -> [200] (== num of classes)
                    relevant_proto_classes = torch.nonzero(classification_weights[:, p] > 1e-3)
                    relevant_proto_class_names = [node_label_to_children[class_idx.item()] for class_idx in relevant_proto_classes]
                    
                    # Take the max per prototype.                             
                    max_per_prototype, max_idx_per_prototype = torch.max(softmaxes, dim=0)
                    max_per_prototype_h, max_idx_per_prototype_h = torch.max(max_per_prototype, dim=1)
                    max_per_prototype_w, max_idx_per_prototype_w = torch.max(max_per_prototype_h, dim=1) #shape (num_prototypes)
                    
                    h_idx = max_idx_per_prototype_h[p, max_idx_per_prototype_w[p]]
                    w_idx = max_idx_per_prototype_w[p]

                    if len(relevant_proto_class_names) == 0:
                        continue
                    
                    if (len(relevant_proto_class_names) == 1) and (relevant_proto_class_names[0] not in non_leaf_children_names):
                        continue
                    
                    h_coor_min, h_coor_max, w_coor_min, w_coor_max = get_img_coordinates(args.image_size, softmaxes.shape, patchsize, skip, h_idx, w_idx)
                    latent_activation = softmaxes[:, p, :, :]
                    if not find_non_descendants:
                        if (coarse_label2name[ys.item()] in relevant_proto_class_names):
                            child_node = root.get_node(coarse_label2name[ys.item()])
                            leaf_descendent = label2name[orig_y.item()][4:7]
                            img_to_open = imgs[i][0] # it is a tuple of (path to image, lable)
                            if topk and (len(proto_mean_activations[p][leaf_descendent]) >= topk):
                                heapq.heappushpop(proto_mean_activations[p][leaf_descendent],\
                                                (pooled[p].item(), pooled_ip[p].item(), img_to_open,\
                                                (h_coor_min, h_coor_max, w_coor_min, w_coor_max), latent_activation))
                            else:
                                heapq.heappush(proto_mean_activations[p][leaf_descendent],\
                                            (pooled[p].item(), pooled_ip[p].item(), img_to_open,\
                                                (h_coor_min, h_coor_max, w_coor_min, w_coor_max), latent_activation))
                    else:
                        if (coarse_label2name[ys.item()] not in relevant_proto_class_names):
                            child_node = root.get_node(coarse_label2name[ys.item()])
                            leaf_descendent = label2name[orig_y.item()][4:7]
                            img_to_open = imgs[i][0] # it is a tuple of (path to image, lable)
                            if topk and (len(proto_mean_activations[p][leaf_descendent]) >= topk):
                                heapq.heappushpop(proto_mean_activations[p][leaf_descendent],\
                                                (pooled[p].item(), pooled_ip[p].item(), img_to_open,\
                                                (h_coor_min, h_coor_max, w_coor_min, w_coor_max), latent_activation))
                            else:
                                heapq.heappush(proto_mean_activations[p][leaf_descendent],\
                                            (pooled[p].item(), pooled_ip[p].item(), img_to_open,\
                                                (h_coor_min, h_coor_max, w_coor_min, w_coor_max), latent_activation))
                    class_and_prototypes[', '.join(relevant_proto_class_names)].add(p)

        
        print('Node', node.name)
        for child_classname in class_and_prototypes:
            
            print('\t'*1, 'Child:', child_classname)
            for p in class_and_prototypes[child_classname]:
                
                logstr = '\t'*2 + f'Proto:{p} '
                for leaf_descendent in proto_mean_activations[p]:
                    mean_activation = round(np.mean([activation for activation, *_ in proto_mean_activations[p][leaf_descendent]]), 4)
                    num_images = len(proto_mean_activations[p][leaf_descendent])
                    logstr += f'{leaf_descendent}:({mean_activation}) '
                print(logstr)
                
                # have this for NON descendants
                if len(proto_mean_activations[p]) == 0:
                    continue
                
                if save_images:
                    patches = []
                    right_descriptions = []
                    text_region_width = 3 # 3x the width of a patch
                    for leaf_descendent, heap in proto_mean_activations[p].items():
                        heap = sorted(heap)[::-1]
                        mean_activation = round(np.mean([activation for activation, *_ in proto_mean_activations[p][leaf_descendent]]), 2)
                        least_activation = min([round(activation, 2) for activation, *_ in proto_mean_activations[p][leaf_descendent]])
                        most_activation = max([round(activation, 2) for activation, *_ in proto_mean_activations[p][leaf_descendent]])
                        mean_cosine_similarity = round(np.mean([activation_inner_product for _, activation_inner_product, *_ in proto_mean_activations[p][leaf_descendent]]), 2)
                        for ele in heap:
                            activation, activation_inner_product, img_to_open, (h_coor_min, h_coor_max, w_coor_min, w_coor_max), latent_activation = ele
                            image = transforms.Resize(size=(args.image_size, args.image_size))(Image.open(img_to_open))
                            img_tensor = transforms.ToTensor()(image)#.unsqueeze_(0) #shape (1, 3, h, w)
                            img_tensor_patch = img_tensor[:, h_coor_min:h_coor_max, w_coor_min:w_coor_max]
    #                         overlayed_image_np = get_heatmap(latent_activation, img_tensor)
    #                         overlayed_image = torch.tensor(overlayed_image_np).permute(2, 0, 1).float() / 255.
    #                         patches.append(overlayed_image)
                            
                            overlayed_image_np = get_heatmap(latent_activation, img_tensor)
                            
                            overlayed_image_pil = Image.fromarray(overlayed_image_np)
                            draw = D.Draw(overlayed_image_pil)
                            text = f"{round(activation, 2), round(activation_inner_product, 2)}"
    #                         text_width, text_height = draw.textsize(text, font2)
                            bbox = draw.textbbox((0, 0), text, font2)
                            text_width = bbox[2] - bbox[0]
                            text_height = bbox[3] - bbox[1]
                            x, y = 224 - text_width - 5, 5  # 10 pixels padding from right
                            draw.text((x, y), text, font=font2, fill=(255, 255, 255))
                            overlayed_image_np = np.array(overlayed_image_pil)
                            
                            overlayed_image = torch.tensor(overlayed_image_np).permute(2, 0, 1).float() / 255.
                            patches.append(overlayed_image)

                        # description on the right hand side
                        text = f'{mean_activation}, {leaf_descendent}'
                        txtimage = Image.new("RGB", (patches[0].shape[-2]*text_region_width,patches[0].shape[-1]), (0, 0, 0))
                        draw = D.Draw(txtimage)
                        draw.text((200, patches[0].shape[1]//2), text, anchor='mm', fill="white", font=font)
                        txttensor = transforms.ToTensor()(txtimage)#.unsqueeze_(0)
                        right_descriptions.append(txttensor)
                    
                    # weird thing padding should be zero for non descendants else it raises some error
                    if find_non_descendants:
                        padding = 0
                    else:
                        padding = 1

                    grid = torchvision.utils.make_grid(patches, nrow=topk, padding=padding)
                    grid_right_descriptions = torchvision.utils.make_grid(right_descriptions, nrow=1, padding=padding)

                    # merging right description with the grid of images
                    grid = torch.cat([grid, grid_right_descriptions], dim=-1)

                    # description on the top
                    text = f'Node:{node.name}, p{p}, Child:{child_classname}'
                    txtimage = Image.new("RGB", (grid.shape[-1], 224), (0, 0, 0))
                    draw = D.Draw(txtimage)
                    draw.text((350, patches[0].shape[1]//2), text, anchor='mm', fill="white", font=font)
                    txttensor = transforms.ToTensor()(txtimage)#.unsqueeze_(0)

                    # merging top description with the grid of images
                    grid = torch.cat([grid, txttensor], dim=1)
                    
                    prefix = 'non_' if find_non_descendants else ''
                    os.makedirs(os.path.join(save_path, prefix + f'descendent_specific_topk_heatmap_ep=last', node.name), exist_ok=True)
                    torchvision.utils.save_image(grid, os.path.join(save_path, prefix + f'descendent_specific_topk_heatmap_ep=last', node.name, f'{child_classname}-p{p}.png'))

    print('Done !!!')