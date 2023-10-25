from pipnet.pipnet import PIPNet, get_network
from util.log import Log
import torch.nn as nn
from util.args import get_args, save_args, get_optimizer_nn
from util.data import get_dataloaders
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
import pickle
from util.func import get_patch_size
import random
from util.data import ModifiedLabelLoader
from tqdm import tqdm

# %% -------------------------------------load the phylogeny tree----------------------------------------

# run_path = '/home/harishbabu/projects/PIPNet/runs/010-CUB-27-imgnet_OOD_cnext26_img=224_nprotos=20'
# run_path = '/home/harishbabu/projects/PIPNet/runs/031-CUB-18-imgnet_cnext26_img=224_nprotos=20_orth-on-rel'
# run_path = '/home/harishbabu/projects/PIPNet/runs/032-CUB-18-imgnet_cnext26_img=224_nprotos=20_orth-on-rel'
# run_path = '/home/harishbabu/projects/PIPNet/runs/035-CUB-18-imgnet_OOD_cnext26_img=224_nprotos=20_orth-on-rel'
# run_path = '/home/harishbabu/projects/PIPNet/runs/043-035_clone-CUB-18-imgnet_OOD_cnext26_img=224_nprotos=20_orth-on-rel'
# run_path = "/home/harishbabu/projects/PIPNet/runs/036-CUB-18-imgnet_OOD_cnext26_img=224_nprotos=20_orth-on-rel_uniformity"
# run_path = "/home/harishbabu/projects/PIPNet/runs/041-035_clone-CUB-18-imgnet_OOD_cnext26_img=224_nprotos=20_orth-on-rel"
# run_path = "/home/harishbabu/projects/PIPNet/runs/042-035_clone-CUB-18-imgnet_OOD_cnext26_img=224_nprotos=20_orth-on-rel"
# run_path = "/home/harishbabu/projects/PIPNet/runs/044-CUB-18-imgnet_OOD_cnext26_img=224_nprotos=20-or-4per-desc_orth-on-rel"
# run_path = "/home/harishbabu/projects/PIPNet/runs/046-CUB-18-imgnet_OOD_cnext26_img=224_nprotos=10per-desc_orth-on-rel"
# run_path = "/home/harishbabu/projects/PIPNet/runs/047-CUB-18-imgnet_OOD_cnext26_img=224_nprotos=5per-desc_tanh-desc"
# run_path = "/home/harishbabu/projects/PIPNet/runs/048-CUB-18-imgnet_OOD_cnext26_img=224_nprotos=5per-desc_tanh-desc_unit-sphere"
# run_path = "/home/harishbabu/projects/PIPNet/runs/051-CUB-18-imgnet_cnext26_img=224_nprotos=4per-desc_tanh-desc_unit-sphere_AW=5-TW=2-UW=2-CW=2"
# run_path = "/home/harishbabu/projects/PIPNet/runs/052-CUB-18-imgnet_OOD_cnext26_img=224_nprotos=4per-desc_tanh-desc_unit-sphere_AW=5-TW=2-UW=2-CW=2"
# run_path = "/home/harishbabu/projects/PIPNet/runs/055-CUB-18_cnext26_img=224_nprotos=4per-desc_unit-sphere_no-softmax_AW=3-TW=2-UW=3-CW=2"
# run_path = "/home/harishbabu/projects/PIPNet/runs/056-CUB-18-imgnet_cnext26_img=224_nprotos=4per-desc_unit-sphere_no-softmax_AW=3-TW=2-UW=3-CW=2"
# run_path = "/home/harishbabu/projects/PIPNet/runs/057-CUB-18-imgnet_cnext26_img=224_nprotos=4per-desc_unit-sphere_no-meanpool_no-softmax_AW=3-TW=2-UW=3-CW=2"
# run_path = "/home/harishbabu/projects/PIPNet/runs/058-CUB-18-imgnet_with-equalize-aug_cnext26_img=224_nprotos=4per-desc_unit-sphere_no-meanpool_no-softmax_AW=3-TW=2-UW=3-CW=2"
run_path = "/home/harishbabu/projects/PIPNet/runs/059-CUB-18-imgnet_with-equalize-aug_cnext26_img=224_nprotos=4per-desc_unit-sphere_finetune=5_no-meanpool_no-softmax_AW=3-TW=2-UW=3-CW=2_batch=20"
args_file = open(os.path.join(run_path, 'metadata', 'args.pickle'), 'rb')
args = pickle.load(args_file)

if args.phylo_config:
    phylo_config = OmegaConf.load(args.phylo_config)

if args.phylo_config:
    # construct the phylo tree
    if phylo_config.phyloDistances_string == 'None':
        if '031' in run_path: # this run uses a different phylogeny file that had an extra root node which is a mistake
            root = construct_phylo_tree('/home/harishbabu/data/phlyogenyCUB/18Species-with-extra-root-node/1_tree-consensus-Hacket-18Species-modified_cub-names_v1.phy')
        else:
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

# if 'tanh-desc' in run_path:
for node in root.nodes_with_children():
    node.set_num_protos(args.num_protos_per_descendant)

# %% -------------------------------------load the model ----------------------------------------


if torch.cuda.is_available():
    device = torch.device('cuda')
    device_ids = [torch.cuda.current_device()]
else:
    device = torch.device('cpu')
    device_ids = []

args_file = open(os.path.join(run_path, 'metadata', 'args.pickle'), 'rb')
args = pickle.load(args_file)

# ckpt_file_name = 'net_overspecific_pruned_replaced_thresh=0.5_last'
ckpt_file_name = 'net_trained_last'
# ckpt_file_name = 'net_trained_10'
# ckpt_file_name = 'net_pretrained'
epoch = ckpt_file_name.split('_')[-1]

ckpt_path = os.path.join(run_path, 'checkpoints', ckpt_file_name)
checkpoint = torch.load(ckpt_path, map_location=device)

if ckpt_file_name != 'net_trained_last':
    print('\n', (10*'-')+'WARNING: Not using the final trained model'+(10*'-'), '\n')

# Obtain the dataset and dataloaders
trainloader, trainloader_pretraining, trainloader_normal, trainloader_normal_augment, projectloader, testloader, test_projectloader, classes = get_dataloaders(args, device)
if len(classes)<=20:
    if args.validation_size == 0.:
        print("Classes: ", testloader.dataset.class_to_idx, flush=True)
    else:
        print("Classes: ", str(classes), flush=True)

# Create a convolutional network based on arguments and add 1x1 conv layer
feature_net, add_on_layers, pool_layer, classification_layers, num_prototypes = get_network(len(classes), args, root=root)
   
# Create a PIP-Net
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
net.load_state_dict(checkpoint['model_state_dict'],strict=True)
net.eval()
criterion = nn.NLLLoss(reduction='mean').to(device)

# Forward one batch through the backbone to get the latent output size
# with torch.no_grad():
#     xs1, _, _ = next(iter(trainloader))
#     xs1 = xs1.to(device)
#     proto_features, _, _ = net(xs1)
#     wshape = proto_features['root'].shape[-1]
#     args.wshape = wshape #needed for calculating image patch size
#     print("Output shape: ", proto_features['root'].shape, flush=True)
    
args.wshape = 26

# %% ------------------------------------- define the integrated gradients functions ----------------------------------------

def integrated_gradients(model, input_image, output, node_name, child_name, p, num_steps=100, device='cuda'):
    model.eval()

    baseline = torch.zeros((1, 3, 224, 224)).to(device)  # Assuming input size is 224x224 and 3 channels (RGB)

    gradients = torch.autograd.grad(outputs=output, inputs=input_image, retain_graph=True)[0]

    integrated_gradients = torch.zeros_like(input_image)

    scaling_factor = (input_image - baseline) / num_steps

    for i in range(1, num_steps + 1):
        step_input = baseline + i * scaling_factor
        _, softmaxes, pooled, _ = model(step_input, inference=False)
        output = pooled[node_name][child_name].squeeze(0)[p]
        step_gradients = torch.autograd.grad(outputs=output, inputs=step_input, retain_graph=True)[0]
        integrated_gradients += step_gradients

    integrated_gradients /= num_steps

    return integrated_gradients


def get_img_coordinates_using_gradients(model, input_image, output, node_name, child_name, p, patch_size=32, num_steps=100, device='cuda'):

    attributions = integrated_gradients(model, input_image, output, node_name, child_name, p, num_steps)

    grayscale_attributions = torch.sum(attributions, dim=1, keepdim=True)

    best_patch_coords = None
    best_score = None

    for i in range(grayscale_attributions.size(2) - patch_size + 1):
        for j in range(grayscale_attributions.size(3) - patch_size + 1):
            h_coord_min, h_coord_max, w_coord_min, w_coord_max = i, i+patch_size, j, j+patch_size
            patch = grayscale_attributions[:, :, h_coord_min:h_coord_max, w_coord_min:w_coord_max]
            
            score = torch.sum(patch)
            
            if best_score is None or score > best_score:
                best_score = score
                best_patch_coords = (h_coord_min, h_coord_max, w_coord_min, w_coord_max)

    return best_patch_coords

# %% ------------------------------------- Calculate and plot the activations ----------------------------------------

# Proto activations on leaf descendents - topk images

from util.data import ModifiedLabelLoader
from collections import defaultdict
import heapq
import pdb
from util.vis_pipnet import get_img_coordinates
import torchvision.transforms as transforms
from PIL import Image, ImageDraw as D
import torchvision

topk = 10
save_images = True

def get_heap():
    list_ = []
    heapq.heapify(list_)
    return list_

patchsize, skip = get_patch_size(args)

for node in root.nodes_with_children():
#     if node.name == 'root':
#         continue
    non_leaf_children_names = [child.name for child in node.children if not child.is_leaf()]
    if len(non_leaf_children_names) == 0: # if all the children are leaf nodes then skip this node
        continue

    name2label = projectloader.dataset.class_to_idx
    label2name = {label:name for name, label in name2label.items()}
    modifiedLabelLoader = ModifiedLabelLoader(projectloader, node)
    coarse_label2name = modifiedLabelLoader.modifiedlabel2name
    node_label_to_children = {label: name for name, label in node.children_to_labels.items()}
    
    imgs = modifiedLabelLoader.filtered_imgs

#     img_iter = tqdm(enumerate(modifiedLabelLoader),
#                     total=len(modifiedLabelLoader),
#                     mininterval=50.,
#                     desc='Collecting topk',
#                     ncols=0)
    
    # maps class names to the prototypes that belong to that
    class_and_prototypes = defaultdict(set)
    
    # maps class names to the prototypes that DON'T have strong connection to classification
    class_and_non_relevant_prototypes = defaultdict(set)
    
    # maps child_class_name -> proto_number -> grand_child_name (or descendant leaf name) -> list of top-k activations
    proto_mean_activations = defaultdict(lambda: defaultdict(lambda: defaultdict(get_heap)))
    
    for child_node in node.children:
        classification_weights = getattr(net.module, '_'+node.name+'_'+child_node.name+'_classification').weight
        
#         if all([grand_child.is_leaf() for grand_child in child_node]):
#             continue

        img_iter = tqdm(enumerate(modifiedLabelLoader),
                    total=len(modifiedLabelLoader),
                    mininterval=50.,
                    desc='Collecting topk',
                    ncols=0)
        
        for i, (xs, orig_y, ys) in img_iter:
            
            if coarse_label2name[ys.item()] not in non_leaf_children_names:
                continue
                
            xs, ys = xs.to(device), ys.to(device)
            
            if coarse_label2name[ys.item()] != child_node.name:
                continue
            
            with torch.no_grad():
                # pooled is dict of dict, mapping [node.name][child_node.name] to correspoding pooled tensor
                # softmaxes is dict, mapping [node.name] to tensor that is produced after concatenating the output
                # of add_ons of each child node and doing softmax on them
                _, softmaxes, pooled, _ = net(xs, inference=False)
                pooled = pooled[node.name][child_node.name].squeeze(0)
                softmaxes = softmaxes[node.name]#.squeeze(0)
                softmaxes_split = torch.split(softmaxes, [node.num_protos_per_child[child_node.name] for child_node in node.children], dim=1)
                idx = [temp_child_node.name for temp_child_node in node.children].index(child_node.name)
                softmaxes_of_child_node = softmaxes_split[idx]
                
                for p in range(pooled.shape[0]): # pooled.shape -> [768] (== num of prototypes)
                    c_weight = torch.max(classification_weights[:,p]) # classification_weights[:,p].shape -> [200] (== num of classes)
#                     relevant_proto_class_names = child_node.descendents # names of all descendants of the child_node
                    if c_weight < 1e-3:
                        class_and_non_relevant_prototypes[child_node.name].add(p)
                        continue
                    relevant_proto_class_names = [child_node.name]
                    
                    # Take the max per prototype.                             
                    max_per_prototype, max_idx_per_prototype = torch.max(softmaxes_of_child_node, dim=0)
                    max_per_prototype_h, max_idx_per_prototype_h = torch.max(max_per_prototype, dim=1)
                    max_per_prototype_w, max_idx_per_prototype_w = torch.max(max_per_prototype_h, dim=1) #shape (num_prototypes)
                    
                    h_idx = max_idx_per_prototype_h[p, max_idx_per_prototype_w[p]]
                    w_idx = max_idx_per_prototype_w[p]

                    # if prototype not relevant # never happens
                    if len(relevant_proto_class_names) == 0:
                        continue
                        
                    # might happen but can be better written
                    if (len(relevant_proto_class_names) == 1) and (relevant_proto_class_names[0] not in non_leaf_children_names):
                        continue
                        
                    h_coor_min, h_coor_max, w_coor_min, w_coor_max = get_img_coordinates(args.image_size, softmaxes_of_child_node.shape, patchsize, skip, h_idx, w_idx)
                    
                    if (coarse_label2name[ys.item()] in relevant_proto_class_names):
#                         child_node = root.get_node(coarse_label2name[ys.item()])
                        leaf_descendent = label2name[orig_y.item()][4:7]
                        img_to_open = imgs[i][0] # it is a tuple of (path to image, lable)
                        if topk and (len(proto_mean_activations[child_node.name][p][leaf_descendent]) > topk):
                            heapq.heappushpop(proto_mean_activations[child_node.name][p][leaf_descendent], (pooled[p].item(), img_to_open, (h_coor_min, h_coor_max, w_coor_min, w_coor_max)))
                        else:
                            heapq.heappush(proto_mean_activations[child_node.name][p][leaf_descendent], (pooled[p].item(), img_to_open, (h_coor_min, h_coor_max, w_coor_min, w_coor_max)))
#                     pdb.set_trace()
                    class_and_prototypes[child_node.name].add(p)
    
    for child_classname in class_and_prototypes:
        print('-'*20, child_classname, class_and_prototypes[child_classname], '-'*20)
    for child_classname in class_and_non_relevant_prototypes:
        print('-'*20, child_classname, class_and_non_relevant_prototypes[child_classname], '-'*20)
    
    print('Node', node.name)
    for child_classname in class_and_prototypes:
        
        print('\t'*1, 'Child:', child_classname)
        for p in class_and_prototypes[child_classname]:
            
            logstr = '\t'*2 + f'Proto:{p} '
            for leaf_descendent in proto_mean_activations[child_classname][p]:
                mean_activation = round(np.mean([activation for activation, *_ in proto_mean_activations[child_classname][p][leaf_descendent]]), 4)
                num_images = len(proto_mean_activations[child_classname][p][leaf_descendent])
                logstr += f'{leaf_descendent}:({mean_activation}) '
            print(logstr)
            
            if save_images:
                patches = []
                right_descriptions = []
                text_region_width = 7 # 7x the width of a patch
                for leaf_descendent, heap in proto_mean_activations[child_classname][p].items():
                    heap = sorted(heap)[::-1]
                    mean_activation = round(np.mean([activation for activation, *_ in proto_mean_activations[child_classname][p][leaf_descendent]]), 4)
                    for ele in heap:
                        activation, img_to_open, (h_coor_min, h_coor_max, w_coor_min, w_coor_max) = ele
                        image = transforms.Resize(size=(args.image_size, args.image_size))(Image.open(img_to_open))
                        img_tensor = transforms.ToTensor()(image)#.unsqueeze_(0) #shape (1, 3, h, w)
                        bbox_coords = torch.tensor([[w_coor_min, h_coor_min, w_coor_max, h_coor_max]])
                        bb_image = torchvision.utils.draw_bounding_boxes((img_tensor * 255).type(torch.uint8), bbox_coords, colors='red') / 255
                        patches.append(bb_image)

                    # description on the right hand side
                    text = f'{mean_activation}, {leaf_descendent}'
                    txtimage = Image.new("RGB", (patches[0].shape[-2]*text_region_width,patches[0].shape[-1]), (0, 0, 0))
                    draw = D.Draw(txtimage)
                    draw.text((5, patches[0].shape[1]//2), text, anchor='mm', fill="white")
                    txttensor = transforms.ToTensor()(txtimage)#.unsqueeze_(0)
                    right_descriptions.append(txttensor)

                grid = torchvision.utils.make_grid(patches, nrow=topk+1, padding=1)
                grid_right_descriptions = torchvision.utils.make_grid(right_descriptions, nrow=1, padding=1)

                # merging right description with the grid of images
#                 pdb.set_trace()
                grid = torch.cat([grid, grid_right_descriptions], dim=-1)

                # description on the top
                text = f'Node:{node.name}, p{p}, Child:{child_classname}'
                txtimage = Image.new("RGB", (grid.shape[-1], args.wshape), (0, 0, 0))
                draw = D.Draw(txtimage)
                draw.text((5, patches[0].shape[1]//2), text, anchor='mm', fill="white")
                txttensor = transforms.ToTensor()(txtimage)#.unsqueeze_(0)

                # merging top description with the grid of images
                grid = torch.cat([grid, txttensor], dim=1)

                os.makedirs(os.path.join(run_path, f'descendent_specific_topk_bb_ep={epoch}', node.name), exist_ok=True)
                torchvision.utils.save_image(grid, os.path.join(run_path, f'descendent_specific_topk_bb_ep={epoch}', node.name, f'{child_classname}-p{p}.png'))
            
print('Done !!!')
