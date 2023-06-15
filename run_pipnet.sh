#!/bin/bash

#SBATCH --account=mabrownlab
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


python main.py --log_dir './runs/001-CUB-190-imgnet_cnext26_img=224' \
               --dataset CUB-190-imgnet \
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
               --num_features 0 \
               --image_size 224 \
               --state_dict_dir_net '' \
               --freeze_epochs 10 \
               --dir_for_saving_images 'Visualization_results' \
               --seed 1 \
               --gpu_ids '' \
               --num_workers 8
               # --bias False \
               # --disable_cuda False \
               # --disable_pretrained False \
               # --weighted_loss False \

# python main.py --log_dir ./runs/checking --dataset CUB-200-2011 --validation_size 0.0 --net convnext_tiny_26 --batch_size 64 --batch_size_pretrain 128 --epochs 0 --epochs_pretrain 0 --optimizer 'Adam' --lr 0.05 --lr_block 0.0005 --lr_net 0.0005 --weight_decay 0.0 --num_features 0 --image_size 224 --state_dict_dir_net '' --freeze_epochs 10 --dir_for_saving_images 'Visualization_results' --seed 1 --gpu_ids '' --num_workers 8 --phylo_config ./configs/cub27_phylogeny.yaml

exit;
