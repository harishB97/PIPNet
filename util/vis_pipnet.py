from tqdm import tqdm
import argparse
import torch
import torch.nn.functional as F
import torch.utils.data
import os
from PIL import Image, ImageDraw as D
import torchvision.transforms as transforms
import torchvision
from util.func import get_patch_size
import random
from util.data import ModifiedLabelLoader
import numpy as np

import wandb
import textwrap

@torch.no_grad()                    
def visualize_topk(net, projectloader, num_classes, device, foldername, args: argparse.Namespace, k=10, node=None, wandb_logging=True):
    print(f"Visualizing prototypes for topk of node {node.name} ...", flush=True)

    # if projectloader.shuffle:
    #     raise('Disable shuffle of projection dataloader')

    name2label = projectloader.dataset.class_to_idx
    label2name = {label:name for name, label in name2label.items()}
    modifiedLabelLoader = ModifiedLabelLoader(projectloader, node)
    projectloader = modifiedLabelLoader
    coarse_label2name = modifiedLabelLoader.modifiedlabel2name

    dir = os.path.join(args.log_dir, foldername)
    if not os.path.exists(dir):
        os.makedirs(dir)

    near_imgs_dirs = dict()
    seen_max = dict()
    saved = dict()
    saved_ys = dict()
    tensors_per_prototype = dict()
    
    for p in range(net.module._num_prototypes):
        near_imgs_dir = os.path.join(dir, str(p))
        near_imgs_dirs[p]=near_imgs_dir
        seen_max[p]=0.
        saved[p]=0
        saved_ys[p]=[]
        tensors_per_prototype[p]=[]
    
    patchsize, skip = get_patch_size(args)

    # imgs = projectloader.dataset.imgs
    imgs = projectloader.filtered_imgs
    
    # Make sure the model is in evaluation mode
    net.eval()
    # classification_weights = net.module._classification.weight
    classification_weights = getattr(net.module, '_'+node.name+'_classification').weight

    # Show progress on progress bar
    img_iter = tqdm(enumerate(projectloader),
                    total=len(projectloader),
                    mininterval=50.,
                    desc='Collecting topk',
                    ncols=0)

    # Iterate through the data
    images_seen = 0
    topks = dict()
    # Iterate through the training set
    for i, (xs, orig_y, ys) in img_iter:
        images_seen+=1
        xs, ys = xs.to(device), ys.to(device)

        with torch.no_grad():
            # Use the model to classify this batch of input data
            pfs, pooled, _ = net(xs, inference=True)
            pooled = pooled[node.name].squeeze(0) 
            pfs = pfs[node.name].squeeze(0) # pfs.shape -> [768, 26, 26] (after squeeze)
            
            for p in range(pooled.shape[0]): # pooled.shape -> [768] (== num of prototypes)
                c_weight = torch.max(classification_weights[:,p]) # classification_weights[:,p].shape -> [200] (== num of classes)
                if c_weight > 1e-3:#ignore prototypes that are not relevant to any class
                    if p not in topks.keys():
                        topks[p] = []
                        
                    if len(topks[p]) < k:
                        topks[p].append((i, pooled[p].item()))
                    else:
                        topks[p] = sorted(topks[p], key=lambda tup: tup[1], reverse=True)
                        if topks[p][-1][1] < pooled[p].item():
                            topks[p][-1] = (i, pooled[p].item())
                        if topks[p][-1][1] == pooled[p].item():
                            # equal scores. randomly chose one (since dataset is not shuffled so latter images with same scores can now also get in topk).
                            replace_choice = random.choice([0, 1])
                            if replace_choice > 0:
                                topks[p][-1] = (i, pooled[p].item())

    alli = []
    prototypes_not_used = []
    for p in topks.keys():
        found = False
        for idx, score in topks[p]:
            alli.append(idx)
            if score > 0.1:  #in case prototypes have fewer than k well-related patches
                found = True
        if not found:
            prototypes_not_used.append(p) # meaning of the topk scores none are above 0.1, they'll still be in alli 

    print(len(prototypes_not_used), "prototypes do not have any similarity score > 0.1. Will be ignored in visualisation.")
    abstained = 0
    # Show progress on progress bar
    img_iter = tqdm(enumerate(projectloader),
                    total=len(projectloader),
                    mininterval=50.,
                    desc='Visualizing topk',
                    ncols=0)
    for i, (xs, orig_y, ys) in img_iter: #shuffle is false so should lead to same order as in imgs
        if i in alli:
            xs, orig_y, ys = xs.to(device), orig_y.to(device), ys.to(device)
            for p in topks.keys():
                if p not in prototypes_not_used:
                    for idx, score in topks[p]:
                        if idx == i:
                            # Use the model to classify this batch of input data
                            with torch.no_grad():
                                softmaxes, pooled, out = net(xs, inference=True) #softmaxes has shape (1, num_prototypes, W, H)
                                softmaxes = softmaxes[node.name]
                                pooled = pooled[node.name]
                                out = out[node.name]
                                outmax = torch.amax(out,dim=1)[0] #shape ([1]) because batch size of projectloader is 1
                                if outmax.item() == 0.:
                                    abstained+=1
                            
                            # Take the max per prototype.                             
                            max_per_prototype, max_idx_per_prototype = torch.max(softmaxes, dim=0)
                            max_per_prototype_h, max_idx_per_prototype_h = torch.max(max_per_prototype, dim=1)
                            max_per_prototype_w, max_idx_per_prototype_w = torch.max(max_per_prototype_h, dim=1) #shape (num_prototypes)
                            
                            c_weight = torch.max(classification_weights[:,p]) #ignore prototypes that are not relevant to any class
                            if (c_weight > 1e-10) or ('pretrain' in foldername):
                                
                                h_idx = max_idx_per_prototype_h[p, max_idx_per_prototype_w[p]]
                                w_idx = max_idx_per_prototype_w[p]
                                
                                img_to_open = imgs[i]
                                if isinstance(img_to_open, tuple) or isinstance(img_to_open, list): #dataset contains tuples of (img,label)
                                    img_to_open = img_to_open[0]
                                
                                image = transforms.Resize(size=(args.image_size, args.image_size))(Image.open(img_to_open))
                                img_tensor = transforms.ToTensor()(image).unsqueeze_(0) #shape (1, 3, h, w)
                                h_coor_min, h_coor_max, w_coor_min, w_coor_max = get_img_coordinates(args.image_size, softmaxes.shape, patchsize, skip, h_idx, w_idx)
                                img_tensor_patch = img_tensor[0, :, h_coor_min:h_coor_max, w_coor_min:w_coor_max]
                                        
                                saved[p]+=1
                                tensors_per_prototype[p].append({'img_tensor': img_tensor[0], 'img_tensor_patch': img_tensor_patch, \
                                                                'coords': (h_coor_min, h_coor_max, w_coor_min, w_coor_max), \
                                                                'fine_label': orig_y, 'coarse_label': ys})
                                # tensors_per_prototype[p].append(img_tensor_patch)

    print("Abstained: ", abstained, flush=True)
    all_tensors = []
    # if wandb_logging:
    #     run = wandb.init(dir=f"/Media/{node.name}", reinit=True)
    for p in range(net.module._num_prototypes):
        if saved[p]>0:
            # add text next to each topk-grid, to easily see which prototype it is

            # create a grid of patches, add text next to each topk-grid, to easily see which prototype it is
            patches = [x['img_tensor_patch'] for x in tensors_per_prototype[p]]
            text = "P "+str(p)
            txtimage = Image.new("RGB", (patches[0].shape[1],patches[0].shape[2]), (0, 0, 0))
            draw = D.Draw(txtimage)
            draw.text((patches[0].shape[0]//2, patches[0].shape[1]//2), text, anchor='mm', fill="white")
            txttensor = transforms.ToTensor()(txtimage)
            patches.append(txttensor)
            grid = torchvision.utils.make_grid(patches, nrow=k+1, padding=1)
            torchvision.utils.save_image(grid,os.path.join(dir,"grid_topk_%s.png"%(str(p))))
            if saved[p]>=k:
                # all_tensors+=tensors_per_prototype[p]
                all_tensors+=patches
            # if wandb_logging:
            #     wandb.log({f"{node.name}_topk_{p}.png": wandb.Image(grid)})
                

            # create a grid of images with bounding box, add text next to each topk-grid, to easily see which prototype it is
            bb_img_tensors = []
            for x in tensors_per_prototype[p]:
                # add bounding box
                h_coor_min, h_coor_max, w_coor_min, w_coor_max = x['coords']
                bb_img_tensor = torchvision.utils.draw_bounding_boxes((x['img_tensor'] * 255).type(torch.uint8), boxes=torch.tensor([[w_coor_min, h_coor_min, w_coor_max, h_coor_max]]), colors=(0, 255, 255))
                # add coarse and fine label to each of the topk image
                if coarse_label2name[x['coarse_label'].item()].startswith('cub'):
                    coarse_name = coarse_label2name[x['coarse_label'].item()][4:7]
                else:
                    coarse_name = coarse_label2name[x['coarse_label'].item()] 
                text = f"Coarse={coarse_name}, Fine={label2name[x['fine_label'].item()][4:7]}" # fine label assumes cub name in the format cub_122_Harris_Sparrow
                bb_img = torchvision.transforms.functional.to_pil_image(bb_img_tensor)
                padding_top = 40
                padding = (0, padding_top, 0, 0)  # Padding (left, top, right, bottom)
                bb_img_padded = Image.new("RGB", (bb_img.width, bb_img.height + padding_top), color=(0, 0, 0))
                bb_img_padded.paste(bb_img, (0, padding_top))
                draw = D.Draw(bb_img_padded)
                draw.text((patches[0].shape[0]//2, patches[0].shape[1]//2), text, anchor='mm', fill="white")
                bb_img_tensors.append(transforms.ToTensor()(bb_img_padded))

            # add prototype number and coarse labels to know the classes this prototype belongs to
            relevant_proto_classes = torch.nonzero(classification_weights[:, p] > 1e-3)
            relevant_proto_class_names = []
            node_label_to_children = {label: name for name, label in node.children_to_labels.items()}
            for class_idx in relevant_proto_classes:
                if node_label_to_children[class_idx.item()].startswith('cub'):
                    relevant_proto_class_names.append(node_label_to_children[class_idx.item()][4:7])
                else:
                    relevant_proto_class_names.append(node_label_to_children[class_idx.item()])
            text = "P "+str(p) + f" belongs to: {','.join(relevant_proto_class_names)}"
            textwrap.wrap(text, width = bb_img_tensors[0].shape[1])
            txtimage = Image.new("RGB", (bb_img_tensors[0].shape[2],bb_img_tensors[0].shape[1]), (0, 0, 0))
            draw = D.Draw(txtimage)
            for lc, line in enumerate(textwrap.wrap(text, width = 30)):
                draw.text((10, 10 + lc*20), line, anchor='mm', fill="white")
            txttensor = transforms.ToTensor()(txtimage)
            bb_img_tensors.append(txttensor)
            # try:
            grid = torchvision.utils.make_grid(bb_img_tensors, nrow=k+1, padding=1)
            torchvision.utils.save_image(grid,os.path.join(dir,"grid_bb_topk_%s.png"%(str(p))))
            # except:
            #     pass
            if wandb_logging:
                wandb.log({f"{node.name}_bb_topk_{p}.png": wandb.Image(grid)})


    if len(all_tensors)>0:
        grid = torchvision.utils.make_grid(all_tensors, nrow=k+1, padding=1)
        torchvision.utils.save_image(grid,os.path.join(dir,"grid_topk_all.png"))
    else:
        print("Pretrained prototypes not visualized. Try to pretrain longer.", flush=True)
    return topks
        

def visualize(net, projectloader, num_classes, device, foldername, args: argparse.Namespace, node=None):
    print("Visualizing prototypes...", flush=True)

    # if projectloader.shuffle:
    #     raise('Disable shuffle of projection dataloader')

    modifiedLabelLoader = ModifiedLabelLoader(projectloader, node)
    projectloader = modifiedLabelLoader

    dir = os.path.join(args.log_dir, foldername)
    if not os.path.exists(dir):
        os.makedirs(dir)

    near_imgs_dirs = dict()
    seen_max = dict()
    saved = dict()
    saved_ys = dict()
    tensors_per_prototype = dict()
    abstainedimgs = set()
    notabstainedimgs = set()
    
    for p in range(net.module._num_prototypes):
        near_imgs_dir = os.path.join(dir, str(p))
        near_imgs_dirs[p]=near_imgs_dir
        seen_max[p]=0.
        saved[p]=0
        saved_ys[p]=[]
        tensors_per_prototype[p]=[]
    
    patchsize, skip = get_patch_size(args)

    # imgs = projectloader.dataset.imgs
    imgs = projectloader.filtered_imgs
    
    # skip some images for visualisation to speed up the process
    if len(imgs)/num_classes <10:
        skip_img=10
    elif len(imgs)/num_classes < 50:
        skip_img=5
    else:
        skip_img = 2
    print("Every", skip_img, "is skipped in order to speed up the visualisation process", flush=True)

    # Make sure the model is in evaluation mode
    net.eval()

    # classification_weights = net.module._classification.weight
    classification_weights = getattr(net.module, '_'+node.name+'_classification').weight

    # Show progress on progress bar
    img_iter = tqdm(enumerate(projectloader),
                    total=len(projectloader),
                    mininterval=100.,
                    desc='Visualizing',
                    ncols=0)

    # Iterate through the data
    images_seen_before = 0
    for i, (xs, ys) in img_iter: #shuffle is false so should lead to same order as in imgs
        if i % skip_img == 0:
            images_seen_before+=xs.shape[0]
            continue
        
        xs, ys = xs.to(device), ys.to(device)
        # Use the model to classify this batch of input data
        with torch.no_grad():
            softmaxes, _, out = net(xs, inference=True) 
            softmaxes = softmaxes[node.name]
            out = out[node.name]

        max_per_prototype, max_idx_per_prototype = torch.max(softmaxes, dim=0)
        # In PyTorch, images are represented as [channels, height, width]
        max_per_prototype_h, max_idx_per_prototype_h = torch.max(max_per_prototype, dim=1)
        max_per_prototype_w, max_idx_per_prototype_w = torch.max(max_per_prototype_h, dim=1)
        for p in range(0, net.module._num_prototypes):
            c_weight = torch.max(classification_weights[:,p]) #ignore prototypes that are not relevant to any class
            if c_weight>0:
                h_idx = max_idx_per_prototype_h[p, max_idx_per_prototype_w[p]]
                w_idx = max_idx_per_prototype_w[p]
                idx_to_select = max_idx_per_prototype[p,h_idx, w_idx].item()
                found_max = max_per_prototype[p,h_idx, w_idx].item()

                imgname = imgs[images_seen_before+idx_to_select]
                if out.max() < 1e-8:
                    abstainedimgs.add(imgname)
                else:
                    notabstainedimgs.add(imgname)
                
                if found_max > seen_max[p]:
                    seen_max[p]=found_max
               
                if found_max > 0.5:
                    img_to_open = imgs[images_seen_before+idx_to_select]
                    if isinstance(img_to_open, tuple) or isinstance(img_to_open, list): #dataset contains tuples of (img,label)
                        imglabel = img_to_open[1]
                        img_to_open = img_to_open[0]

                    image = transforms.Resize(size=(args.image_size, args.image_size))(Image.open(img_to_open).convert("RGB"))
                    img_tensor = transforms.ToTensor()(image).unsqueeze_(0) #shape (1, 3, h, w)
                    h_coor_min, h_coor_max, w_coor_min, w_coor_max = get_img_coordinates(args.image_size, softmaxes.shape, patchsize, skip, h_idx, w_idx)
                    img_tensor_patch = img_tensor[0, :, h_coor_min:h_coor_max, w_coor_min:w_coor_max]
                    saved[p]+=1
                    tensors_per_prototype[p].append((img_tensor_patch, found_max))
                    
                    save_path = os.path.join(dir, "prototype_%s")%str(p)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    draw = D.Draw(image)
                    draw.rectangle([(w_coor_min,h_coor_min), (w_coor_max, h_coor_max)], outline='yellow', width=2)
                    image.save(os.path.join(save_path, 'p%s_%s_%s_%s_rect.png'%(str(p),str(imglabel),str(round(found_max, 2)),str(img_to_open.split('/')[-1].split('.jpg')[0]))))
                    
        
        images_seen_before+=len(ys)

    print("num images abstained: ", len(abstainedimgs), flush=True)
    print("num images not abstained: ", len(notabstainedimgs), flush=True)
    for p in range(net.module._num_prototypes):
        if saved[p]>0:
            try:
                sorted_by_second = sorted(tensors_per_prototype[p], key=lambda tup: tup[1], reverse=True)
                sorted_ps = [i[0] for i in sorted_by_second]
                grid = torchvision.utils.make_grid(sorted_ps, nrow=16, padding=1)
                torchvision.utils.save_image(grid,os.path.join(dir,"grid_%s.png"%(str(p))))
            except RuntimeError:
                pass

# convert latent location to coordinates of image patch
def get_img_coordinates(img_size, softmaxes_shape, patchsize, skip, h_idx, w_idx):

    w_idx = w_idx.item() if torch.is_tensor(w_idx) else w_idx
    h_idx = h_idx.item() if torch.is_tensor(h_idx) else h_idx
    
    # in case latent output size is 26x26. For convnext with smaller strides. 
    if softmaxes_shape[1] == 26 and softmaxes_shape[2] == 26:
        #Since the outer latent patches have a smaller receptive field, skip size is set to 4 for the first and last patch. 8 for rest.

        h_coor_min = max(0,(h_idx-1)*skip+4)
        if h_idx < softmaxes_shape[-1]-1:
            h_coor_max = h_coor_min + patchsize
        else:
            h_coor_min -= 4
            h_coor_max = h_coor_min + patchsize
            
        w_coor_min = max(0,(w_idx-1)*skip+4)
        if w_idx < softmaxes_shape[-1]-1:
            w_coor_max = w_coor_min + patchsize
        else:
            w_coor_min -= 4
            w_coor_max = w_coor_min + patchsize
        
    else:
        h_coor_min = h_idx*skip
        h_coor_max = min(img_size, h_idx*skip+patchsize)
        w_coor_min = w_idx*skip
        w_coor_max = min(img_size, w_idx*skip+patchsize)  
    
    if h_idx == softmaxes_shape[1]-1:
        h_coor_max = img_size
    if w_idx == softmaxes_shape[2] -1:
        w_coor_max = img_size
    if h_coor_max == img_size:
        h_coor_min = img_size-patchsize
    if w_coor_max == img_size:
        w_coor_min = img_size-patchsize

    return h_coor_min, h_coor_max, w_coor_min, w_coor_max
    