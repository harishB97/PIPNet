import sys, os
from collections import defaultdict
import heapq
from adversarial.utils import unshuffle_dataloader, get_heap, undo_preprocess, get_less_activation_locations_mask, get_less_activation_locations, get_normalize_transform

# 118 Project dist with BYOL, HPIPNetBYOLOpt2ProtopoolProjDist
run_path = "/home/harishbabu/projects/PIPNet/runs/118-HPIPNetBYOLOpt2ProtopoolProjDist_CUB-18-imgnet-224_with-equalize-aug_cnext26_img=224_nprotos=20_BYOL_no-KO_no-OOD_no-AL_no-TANH"


try:
    sys.path.remove('/home/harishbabu/projects/PIPNet')
except:
    pass
sys.path.insert(0, os.path.join(run_path, 'source_clone'))

print(run_path)

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

from torchvision.datasets.folder import ImageFolder
from torch.utils.data import DataLoader
from skimage.filters import threshold_local, gaussian
import ntpath

print(sys.path)

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

import torch
import torch.optim as optim

#%% Load model

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

exp_no = int(os.path.basename(run_path)[:3])

if exp_no < 77:
    if ('num_protos_per_descendant' in args) and (args.num_protos_per_descendant > 0):
        for node in root.nodes_with_children():
            node.set_num_protos(args.num_protos_per_descendant)
if exp_no == 77:
    # update num of protos per node based on num_protos_per_descendant
    if args.num_features == 0 and args.num_protos_per_descendant == 0:
        raise Exception('Either of num_features or num_protos_per_descendant must be greater than zero')
    for node in root.nodes_with_children():
        node.set_num_protos(num_protos_per_descendant=args.num_protos_per_descendant,\
                                                            min_protos=args.num_features)
else:
    if ('num_protos_per_descendant' in args):
        # update num of protos per node based on num_protos_per_descendant
        if args.num_features == 0 and args.num_protos_per_descendant == 0:
            raise Exception('Either of num_features or num_protos_per_descendant must be greater than zero')
        for node in root.nodes_with_children():
            node.set_num_protos(num_protos_per_descendant=args.num_protos_per_descendant,\
                                min_protos=args.num_features,\
                                split_protos=('protopool' in args) and (args.protopool == 'n'))

if torch.cuda.is_available():
    device = torch.device('cuda')
    device_ids = [torch.cuda.current_device()]
else:
    device = torch.device('cpu')
    device_ids = []

# args_file = open(os.path.join(run_path, 'metadata', 'args.pickle'), 'rb')
# args = pickle.load(args_file)

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

print(args.batch_size, trainloader.batch_size)

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

# Create a PIP-Net
if ('byol' in args) and (args.byol == 'y'):
    from pipnet.pipnet import PIPNetBYOL
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
net.load_state_dict(checkpoint['model_state_dict'],strict=True)
print(net.eval())
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

#%%

def adversarial_attack_GPT(net, node_name, proto_idx, image, locs, num_steps=50, epsilon=0.01, alpha=0.001, random_init=True):
    # Ensure model is in evaluation mode and image requires gradient
    net.eval()
    image = image.clone().detach().requires_grad_(True)

    # Original image to compare with adversarial image
    original_image = image.clone().detach()

    # Define an optimizer for the adversarial image
    optimizer = optim.SGD([image], lr=alpha)

    for step in range(num_steps):
        optimizer.zero_grad()
        output = net(image)

        # Extract the activation map for the m-th prototype
        activation_map = output[0, 0]  # Assuming output is [B, 1, H', W']
        
        # Calculate the loss based on the distance of peak activation from target locations
        loss = calculate_custom_loss(activation_map, locs)

        # Perform backpropagation and update the image
        loss.backward()
        optimizer.step()

        # Apply constraints to keep the pixel values valid and within epsilon-ball of the original image
        with torch.no_grad():
            perturbed_image = image + (image - original_image).clamp(-epsilon, epsilon)
            image.data = torch.clamp(perturbed_image, 0, 1)

        # Check if the peak activation has moved to one of the target locations
        if check_peak_activation(activation_map, locs):
            return True

    return False


def adversarial_attack(net, node_name, proto_idx, image, locs, preprocess_transform, num_steps=40, epsilon=8/255, alpha=2/255, random_init=True):
    clip_min, clip_max = 0.0, 1.0

    # Ensure model is in evaluation mode and image requires gradient
    net.eval()

    # Get mask for the locations to be considered for adversarial attack
    if 'byol' in args and (args.byol.split('|')[0] == 'y'):
        _, _, features, proto_features, pooled, out = net(image)
    else:
        features, proto_features, pooled, out = net(image)
    activation_map = proto_features[node_name][:, proto_idx, :, :]
    adversarial_locs_mask = get_less_activation_locations_mask(activation_map.squeeze(0), threshold=0.4, window_size=[5, 5])

    # Undo the preprocessing std and mean
    image = undo_preprocess(image)

    # Original image to compare with adversarial image
    original_image = image.clone().detach()

    # Clone and detach the image
    image = image.clone().detach().requires_grad_(True)

    # Add a random noise to the image initially
    if random_init:
        image = torch.clamp(
            image + torch.empty_like(image).uniform_(-epsilon, epsilon),
            clip_min,
            clip_max,
        )

    # Define an optimizer for the adversarial image
    optimizer = optim.SGD([image], lr=alpha)

    for step in range(num_steps):
        image = image.requires_grad_(True)

        optimizer.zero_grad()

        if 'byol' in args and (args.byol.split('|')[0] == 'y'):
            preprocessed_image = preprocess_transform(image)
            _, _, features, proto_features, pooled, out = net(preprocessed_image)
        else:
            preprocessed_image = preprocess_transform(image)
            features, proto_features, pooled, out = net(preprocessed_image)

        # Extract the activation map for the m-th prototype
        adversarial_activation_map = proto_features[node_name][:, proto_idx, :, :]

        assert adversarial_activation_map.shape[0] == 1

        # Calculate the loss based on the distance of peak activation from target locations
        loss = calculate_custom_loss(adversarial_activation_map, adversarial_locs_mask)

        # Perform backpropagation and update the image
        loss.backward()
        optimizer.step()

        # Apply constraints to keep the pixel values valid and within epsilon-ball of the original image
        with torch.no_grad():
            perturbed_image = image + (image - original_image).clamp(-epsilon, epsilon)
            image = torch.clamp(perturbed_image, 0, 1)

    if 'byol' in args and (args.byol.split('|')[0] == 'y'):
        preprocessed_image = preprocess_transform(image)
        _, _, features, proto_features, pooled, out = net(preprocessed_image)
    else:
        preprocessed_image = preprocess_transform(image)
        features, proto_features, pooled, out = net(preprocessed_image)
    adversarial_activation_map = proto_features[node_name][:, proto_idx, :, :]

    # Check if the peak activation has moved to one of the target locations
    if check_peak_activation(adversarial_activation_map, adversarial_locs_mask):
        return True

    return False


# Helper function to calculate your custom loss
def calculate_custom_loss(activation_map, adversarial_locs_mask):
    """
    activation_map.shape => [1, 1, H, W]
    adversarial_locs_mask.shape => [H, W]
    """
    # Implement your custom loss calculation here
    adversarial_locs_mask = adversarial_locs_mask.unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
    high_act_locs_mask = torch.logical_not(adversarial_locs_mask)

    return torch.mean(activation_map * adversarial_locs_mask) - torch.mean(activation_map * high_act_locs_mask)


# Helper function to check if peak activation is at target location
def check_peak_activation(adversarial_activation_map, adversarial_locs_mask):
    # Implement your check here
    max_idx = torch.argmax(adversarial_activation_map)
    H, W = adversarial_activation_map.shape
    max_coord = (max_idx // W, max_idx % W)

    # Access the corresponding value in tensor2
    return adversarial_locs_mask[max_coord].item()

#%%

vizloader_name = 'projectloader'
topk = 5
minimum_peak_act_threshold = 0.8
vizloader_dict = {'trainloader': trainloader,
                 'projectloader': projectloader,
                 'testloader': testloader,
                 'test_projectloader': test_projectloader}
vizloader_dict[vizloader_name] = unshuffle_dataloader(vizloader_dict[vizloader_name])

if type(vizloader_dict[vizloader_name].dataset) == ImageFolder:
    name2label = vizloader_dict[vizloader_name].dataset.class_to_idx
    label2name = {label:name for name, label in name2label.items()}
else:
    name2label = vizloader_dict[vizloader_name].dataset.dataset.dataset.class_to_idx
    label2name = {label:name for name, label in name2label.items()}

# maps node.name -> proto_num -> data related to it
node_proto_data = defaultdict(lambda: defaultdict(list))

for node in root.nodes_with_children():

    modifiedLabelLoader = ModifiedLabelLoader(vizloader_dict[vizloader_name], node)
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
    proto_mean_activations = defaultdict(get_heap)

    # maps class names to the prototypes that belong to that
    class_and_prototypes = defaultdict(set)

    for i, (xs, orig_y, ys) in img_iter:
        xs, ys = xs.to(device), ys.to(device)

        with torch.no_grad():
            features, proto_features, pooled, out = net(xs, inference=True)

            proto_features = proto_features[node.name]
            pooled = pooled[node.name].squeeze(0)

            for p in range(pooled.shape[0]): # pooled.shape -> [768] (== num of prototypes)
                c_weight = torch.max(classification_weights[:,p]) # classification_weights[:,p].shape -> [200] (== num of classes)
                relevant_proto_classes = torch.nonzero(classification_weights[:, p] > 1e-3)
                relevant_proto_class_names = [node_label_to_children[class_idx.item()] for class_idx in relevant_proto_classes]
                if len(relevant_proto_class_names) == 0:
                    continue

                # Take the max per prototype.                             
                max_per_prototype, max_idx_per_prototype = torch.max(proto_features, dim=0)
                max_per_prototype_h, max_idx_per_prototype_h = torch.max(max_per_prototype, dim=1)
                max_per_prototype_w, max_idx_per_prototype_w = torch.max(max_per_prototype_h, dim=1) #shape (num_prototypes)
                h_idx = max_idx_per_prototype_h[p, max_idx_per_prototype_w[p]]
                w_idx = max_idx_per_prototype_w[p]

                latent_activation = proto_features[:, p, :, :]

                if (coarse_label2name[ys.item()] in relevant_proto_class_names):
                    img_to_open = imgs[i][0] # it is a tuple of (path to image, lable)

                    if topk and (len(proto_mean_activations[p]) >= topk):
                        heapq.heappushpop(proto_mean_activations[p], (pooled[p].item(), xs, img_to_open, latent_activation))
                    else:
                        heapq.heappush(proto_mean_activations[p], (pooled[p].item(), xs, img_to_open, latent_activation))

                class_and_prototypes[', '.join(relevant_proto_class_names)].add(p)

    print('Node', node.name)
    for class_label in range(classification_weights.shape[0]):
        child_name = (coarse_label2name[class_label])
        print('Num protos for', child_name, torch.nonzero(classification_weights[class_label, :] > 1e-3).shape[0])

    for child_classname in class_and_prototypes:
        
        print('\t'*1, 'Child:', child_classname)
        for p in class_and_prototypes[child_classname]:

            if any([peak_act < minimum_peak_act_threshold for peak_act, *_ in proto_mean_activations[p]]):
                continue

            node_proto_data[node.name][p] = proto_mean_activations[p]



            



