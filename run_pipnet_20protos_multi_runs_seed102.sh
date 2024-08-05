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
export MASTER_PORT=29502
export OMP_NUM_THREADS=4
CUDA_VISIBLE_DEVICES=2,3 MASTER_PORT=29502 TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nproc_per_node=2 --master_port=29502 main_dist.py --log_dir './runs/229-178like_repeatation_seed=102_nprotos=20pc-cnext26_PruningBF=1.1NaiveHPIPNetMaskL1=0.5MaskTrainExtra=05epsEps=60Cl=2.0TanhDesc=0.05MinCont=0.1_CUB-190-imgnet-224_WeightedCE_with-equalize-aug_img=224' \
               --training_wheels "n" \
               --copy_files "n" \
               --wandb "n" \
               --dataset CUB-190-imgnet-224 \
               --net convnext_tiny_26 \
               --batch_size 128 \
               --batch_size_pretrain 128 \
               --epochs 75 \
               --epochs_pretrain 10 \
               --epochs_finetune 0 \
               --epochs_finetune_classifier 3 \
               --epochs_finetune_mask_prune 60 \
               --freeze_epochs 10 \
               --optimizer 'Adam' \
               --lr 0.05 \
               --lr_block 0.0005 \
               --lr_net 0.0005 \
               --weight_decay 0.0 \
               --image_size 224 \
               --state_dict_dir_net '' \
               --dir_for_saving_images 'Visualization_results' \
               --seed 102 \
               --gpu_ids '0,1,2' \
               --num_workers 8 \
               --phylo_config ./configs/cub190_phylogeny.yaml \
               --experiment_note "" \
               --kernel_orth "y" \
               --num_features 0 \
               --num_protos_per_descendant 0 \
               --num_protos_per_child 20 \
               --align "n" \
               --uni "n" \
               --align_pf "y" \
               --tanh "y" \
               --tanh_desc "y|0.05" \
               --tanh_during_second_phase 'y' \
               --sg_before_masking 'y' \
               --softmax "y|1" \
               --weighted_ce_loss "y" \
               --focal_loss "n" \
               --focal_loss_gamma 2.0 \
               --protopool "n" \
               --state_dict_dir_backbone "" \
               --viz_loader 'testloader,projectloader' \
               --classifier 'NonNegative' \
               --pipnet_sparsity 'y' \
               --mask_prune_overspecific 'n' \
               --geometric_mean_overspecificity_score 'n' \
               --minimize_contrasting_set 'y' \
               --leave_out_classes "" \
               --cl_weight 2.0 \
               --OOD_ent 'n' \
               # --bias \
               # --leave_out_classes "./configs/leave_out_classes_CUB-190_10_set1.txt" \
               # --state_dict_dir_backbone "/home/harishbabu/projects/PIPNet/runs/082-CUB-18-imgnet_with-equalize-aug_cnext26_img=224_nprotos=4per-leaf-desc_unit-sphere_finetune=5_no-meanpool_with-softmax_no-addon-bias_AW=3-TW=2-MMW=2-UW=3-CW=2_mm-loss_batch=48/checkpoints/net_pretrained" \
               # --state_dict_dir_net '/home/harishbabu/projects/PIPNet/runs/068-CUB-18-imgnet_with-equalize-aug_cnext26_img=224_nprotos=4per-desc_unit-sphere-protopool_finetune=5_no-meanpool_no-softmax_AW=3-TW=2-UW=3-CW=2_batch=20/checkpoints/net_pretrained' \
            #    --add_on_bias \
               # --OOD_dataset 'CUB-172-OOD-imgnet-224' \
               # --state_dict_dir_backbone '/home/harishbabu/projects/PIPNet/runs/CUB-190-imgnet_cnext26_img=224/checkpoints/net_trained_last' \
               # --bias False \
               # --disable_cuda False \
               # --disable_pretrained False \
               # --weighted_loss False \


exit;