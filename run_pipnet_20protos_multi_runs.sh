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

# --epochs 75 \
# --epochs_pretrain 10 \
# --epochs_finetune 0 \
# --epochs_finetune_classifier 3 \
# --epochs_finetune_mask_prune 60 \
# --freeze_epochs 10 \
# --mask_prune_overspecific 'y|0|1.1' \

# DO THIS AFTER TRAINING WHEELS -|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
# set finetune back to 5, epochs_pretrain=30, epochs=60, freeze_epochs=10

sh run_pipnet_20protos_multi_runs_seed42.sh &

sh run_pipnet_20protos_multi_runs_seed102.sh &

sh run_pipnet_20protos_multi_runs_seed214.sh &

sh run_pipnet_20protos_multi_runs_seed777.sh &


exit;