# #!/bin/bash

# #SBATCH --account=imageomicswithanuj
# #SBATCH --partition=dgx_normal_q
# #SBATCH --time=12:00:00 
# #SBATCH --gres=gpu:2
# #SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
# #SBATCH -o ./SLURM/slurm-%x.%j.out

# echo start load env and run python

# module reset
# module load Anaconda3/2020.11
# source activate hpnet4
# module reset
# source activate hpnet4
# which python

# # --net options
# # # dinov2_vits14_reg
# # # convnext_tiny_26
# # # convnext_tiny_13

# # --dataset options
# # FISH-38-224
# # CUB-190-imgnet-224
# # CUB-190-224
# # FISH-38-224

# DO THIS AFTER TRAINING WHEELS -|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
# set finetune back to 5, epochs_pretrain=30, epochs=60, freeze_epochs=10
python main.py --log_dir './runs/Testing-cleanedupcode' \
               --copy_files "y" \
               --wandb "y" \
               --dataset CUB-190-imgnet-224 \
               --net convnext_tiny_26 \
               --batch_size 256 \
               --batch_size_pretrain 256 \
               --epochs 75 \
               --epochs_pretrain 10 \
               --epochs_finetune_classifier 3 \
               --epochs_finetune_mask 60 \
               --freeze_epochs 10 \
               --gpu_ids '0,1' \
               --num_workers 8 \
               --phylo_config ./configs/cub190_phylogeny.yaml \
               --num_protos_per_child 10 \
               --weighted_ce_loss "y" \


# python main.py --log_dir './runs/Testing-cleanedupcode' \
#                --copy_files "n" \
#                --wandb "n" \
#                --dataset CUB-190-imgnet-224 \
#                --net convnext_tiny_26 \
#                --batch_size 256 \
#                --batch_size_pretrain 256 \
#                --epochs 75 \
#                --epochs_pretrain 0 \
#                --epochs_finetune_classifier 0 \
#                --epochs_finetune_mask 60 \
#                --freeze_epochs 0 \
#                --gpu_ids '0,1' \
#                --num_workers 8 \
#                --phylo_config ./configs/cub190_phylogeny.yaml \
#                --num_protos_per_child 10 \
#                --weighted_ce_loss "y" \

exit;