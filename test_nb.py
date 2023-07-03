#!/usr/bin/env python
# coding: utf-8

# In[6]:


import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import os
import torchvision.transforms as transforms
from pipnet.pipnet import PIPNet, get_network
from util.data import get_dataloaders
from util.vis_pipnet import get_img_coordinates
from util.func import get_patch_size
from util.eval_cub_csv import get_topk_cub
from util.node import Node
from PIL import ImageFont, Image, ImageDraw as D
from pipnet.train import test_pipnet
from omegaconf import OmegaConf
from util.phylo_utils import construct_phylo_tree, construct_discretized_phylo_tree
import wandb

print(torch.cuda.is_available())

# get_ipython().system('which python')


# In[7]:


# run_path = '/home/harishbabu/projects/PIPNet/runs/004-CUB-27-imgnet_cnext26_img=224_nprotos=200'
run_path = '/home/harishbabu/projects/PIPNet/runs/005-CUB-27-imgnet_cnext26_img=224_nprotos=50'

device = torch.device('cuda')
device_ids = [torch.cuda.current_device()]

# device = torch.device('cpu')
# device_ids = []

args_file = open(os.path.join(run_path, 'metadata', 'args.pickle'), 'rb')
args = pickle.load(args_file)

ckpt_path = os.path.join(run_path, 'checkpoints', 'net_trained_last')
checkpoint = torch.load(ckpt_path, map_location=device)


# In[8]:


if args.phylo_config:
    phylo_config = OmegaConf.load(args.phylo_config)

if args.phylo_config:
    # construct the phylo tree
    if phylo_config.phyloDistances_string == 'None':
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


# In[9]:


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
# net.load_state_dict(checkpoint['model_state_dict'],strict=True)
criterion = nn.NLLLoss(reduction='mean').to(device)


# In[10]:


epoch = 0
run = wandb.init(project="dump", name=os.path.basename(args.log_dir), config=vars(args), reinit=False)
info = test_pipnet(net, testloader, criterion, epoch, device, progress_prefix= 'Test Epoch', wandb_logging=True, wandb_log_subdir = 'test')
print('test', info['fine_accuracy'])
info = test_pipnet(net, trainloader, criterion, epoch, device, progress_prefix= 'Train Epoch', wandb_logging=False, wandb_log_subdir = 'train')
print('train', info['fine_accuracy'])

# In[ ]:




