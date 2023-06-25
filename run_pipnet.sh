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


python main.py --log_dir './runs/004-CUB-27-imgnet_cnext26_img=224_nprotos=200' \
               --dataset CUB-27-imgnet-224 \
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
               --num_features 200 \
               --image_size 224 \
               --state_dict_dir_net '' \
               --freeze_epochs 10 \
               --dir_for_saving_images 'Visualization_results' \
               --seed 1 \
               --gpu_ids '' \
               --num_workers 8 \
               --phylo_config ./configs/cub27_phylogeny.yaml \
               --state_dict_dir_backbone '/home/harishbabu/projects/PIPNet/runs/CUB-190-imgnet_cnext26_img=224/checkpoints/net_trained_last' \
               --experiment_note "Using backbone thats already trained with all 190 species. Limited protos to 200 bcoz of memory issue. Added wandb logging"
               # --bias False \
               # --disable_cuda False \
               # --disable_pretrained False \
               # --weighted_loss False \

# python main.py --log_dir ./runs/checking --dataset CUB-27-imgnet-224 --validation_size 0.0 --net convnext_tiny_26 --batch_size 64 --batch_size_pretrain 128 --epochs 2 --epochs_pretrain 2 --optimizer 'Adam' --lr 0.05 --lr_block 0.0005 --lr_net 0.0005 --weight_decay 0.0 --num_features 200 --image_size 224 --state_dict_dir_net '' --freeze_epochs 10 --dir_for_saving_images 'Visualization_results' --seed 1 --gpu_ids '' --num_workers 8 --phylo_config ./configs/cub27_phylogeny.yaml --state_dict_dir_backbone '/home/harishbabu/projects/PIPNet/runs/CUB-190-imgnet_cnext26_img=224/checkpoints/net_trained_last'

exit;
# [print(xs1.shape) for i, (xs1, xs2, ys) in train_loader]

# [print(xs1.shape) for xs1, xs2, ys in train_loader]