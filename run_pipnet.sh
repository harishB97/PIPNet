#!/bin/bash

#SBATCH --account=ml4science
#SBATCH --partition=dgx_normal_q
#SBATCH --time=1-00:00:00 
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH -o ./SLURM/slurm-%j.out


echo start load env and run python

module reset
module load Anaconda3/2020.11
source activate hpnet1
module reset
source activate hpnet1
which python

# 026-CUB-27-imgnet_OOD_cnext26_img=224_nprotos=20_orth
python main.py --log_dir './runs/035-CUB-18-imgnet_OOD_cnext26_img=224_nprotos=20_orth-on-rel' \
               --dataset CUB-18-imgnet-224 \
               --validation_size 0.0 \
               --net convnext_tiny_26 \
               --batch_size 64 \
               --batch_size_pretrain 128 \
               --epochs 60 \
               --epochs_pretrain 10 \
               --optimizer 'Adam' \
               --lr 0.05 \
               --lr_block 0.0005 \
               --lr_net 0.0005 \
               --weight_decay 0.0 \
               --num_features 20 \
               --image_size 224 \
               --state_dict_dir_net '' \
               --freeze_epochs 10 \
               --dir_for_saving_images 'Visualization_results' \
               --seed 1 \
               --gpu_ids '' \
               --num_workers 8 \
               --phylo_config ./configs/cub18_phylogeny.yaml \
               --experiment_note "Added OOD for 18species subset. Added kernel orthogonality on only relevant prototype kernels with loss-weight 0.5. Filtered imgs in vis_pipnet and fixed the previous issue. Separate add_on for each node. Using cropped images for projection. Removed scaling -> (len(node_y) / len(ys[ys != OOD_LABEL])). Set finetune to 0 and Set freeze_epochs to 30. Added OOD loss, removed pretrained backbone. 005 had incorrect data.py. Fixed it again. Reducing protos to 50 from 200 since there is a lot of meaningless prototypes in 004. Not Using backbone thats already trained with all 190 species. Limited protos to 200 bcoz of memory issue. Added wandb logging" \
               --kernel_orth "y" \
               --OOD_dataset 'CUB-172-OOD-imgnet-224' \
               # --state_dict_dir_backbone '/home/harishbabu/projects/PIPNet/runs/CUB-190-imgnet_cnext26_img=224/checkpoints/net_trained_last' \
               # --bias False \
               # --disable_cuda False \
               # --disable_pretrained False \
               # --weighted_loss False \

#-------------------DEBUGGING PURPOSE ONLY------------------------#

# python main.py --log_dir './runs/checking6' \
#                --dataset CUB-27-imgnet-224 \
#                --validation_size 0.0 \
#                --net convnext_tiny_26 \
#                --batch_size 64 \
#                --batch_size_pretrain 128 \
#                --epochs 8 \
#                --epochs_pretrain 1 \
#                --optimizer 'Adam' \
#                --lr 0.05 \
#                --lr_block 0.0005 \
#                --lr_net 0.0005 \
#                --weight_decay 0.0 \
#                --num_features 20 \
#                --image_size 224 \
#                --state_dict_dir_net '' \
#                --freeze_epochs 10 \
#                --dir_for_saving_images 'Visualization_results' \
#                --seed 1 \
#                --gpu_ids '' \
#                --num_workers 8 \
#                --phylo_config ./configs/cub27_phylogeny.yaml \
#                --experiment_note "Added OOD loss. Reducing protos to 50 from 200 since there is a lot of meaningless prototypes in 004. Using backbone thats already trained with all 190 species. Limited protos to 200 bcoz of memory issue. Added wandb logging" \
#                --OOD_dataset 'CUB-163-OOD-imgnet-224' \
#                --state_dict_dir_backbone '/home/harishbabu/projects/PIPNet/runs/CUB-190-imgnet_cnext26_img=224/checkpoints/net_trained_last' \
#                # --bias False \
#                # --disable_cuda False \
#                # --disable_pretrained False \
#                # --weighted_loss False \


# python main.py --log_dir ./runs/checking --dataset CUB-27-imgnet-224 --validation_size 0.0 --net convnext_tiny_26 --batch_size 64 --batch_size_pretrain 128 --epochs 2 --epochs_pretrain 2 --optimizer 'Adam' --lr 0.05 --lr_block 0.0005 --lr_net 0.0005 --weight_decay 0.0 --num_features 200 --image_size 224 --state_dict_dir_net '' --freeze_epochs 10 --dir_for_saving_images 'Visualization_results' --seed 1 --gpu_ids '' --num_workers 8 --phylo_config ./configs/cub27_phylogeny.yaml --state_dict_dir_backbone '/home/harishbabu/projects/PIPNet/runs/CUB-190-imgnet_cnext26_img=224/checkpoints/net_trained_last'

exit;
# [print(xs1.shape) for i, (xs1, xs2, ys) in train_loader]

# [print(xs1.shape) for xs1, xs2, ys in train_loader]