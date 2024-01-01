#!/bin/bash

#SBATCH --account=ml4science
#SBATCH --partition=dgx_normal_q
#SBATCH --time=06:00:00 
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH -o ./SLURM/slurm-%j.out


echo start load env and run python

# module reset
# module load Anaconda3/2020.11
# source activate hpnet1
# module reset
# source activate hpnet1
# which python

module reset
module load Anaconda3/2020.11
source activate hpnet3
module reset
source activate hpnet3
which python

# module reset
# module load Anaconda3/2020.11
# source activate maskclip2
# module reset
# source activate maskclip2
# which python

# module reset
# module load Anaconda3/2020.11
# source activate maskclip1
# module reset
# source activate maskclip1
# which python

# when coming out of training wheels
# set copy files to "y"
# comment memory usage logging
# set the epochs right

# 080-CUB-18-imgnet_with-equalize-aug_cnext7_img=224_nprotos=20_unit-sphere-protopool_finetune=5_align-pf-during-training_no-meanpool_with-softmax_no-addon-bias_AW=3-TW=2-UW=3-CW=2_weighted-ce_batch=20
# epoch to 60, pretrain to 60, freeze_epochs 10, finetune to 5, viz topk commented at all places, print weights commented, prototype purity commented
# pretraining-check-001-AL=3_UW=6

# DO THIS AFTER TRAINING WHEELS -|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
# set finetune back to 5, epochs_pretrain=30, epochs=60, freeze_epochs=10
python main.py --log_dir './runs/110-FlatStructure180protosNoALUNI_CUB-18-imgnet-bg_with-equalize-aug_cnext26_BGM=4|1.0|50_img=224_nprotos=180_unit-sphere-protopool_no-meanpool_with-softmax-tau=0.2_no-addon-bias_AW=3-TW=2-MMW=2-UW=3-CW=2_batch=48' \
               --training_wheels "n" \
               --dataset CUB-18-imgnet-224 \
               --validation_size 0.0 \
               --net convnext_tiny_26 \
               --batch_size 48 \
               --batch_size_pretrain 48 \
               --epochs 60 \
               --epochs_pretrain 30 \
               --epochs_finetune 5 \
               --optimizer 'Adam' \
               --lr 0.05 \
               --lr_block 0.0005 \
               --lr_net 0.0005 \
               --weight_decay 0.0 \
               --image_size 224 \
               --state_dict_dir_net '' \
               --freeze_epochs 10 \
               --dir_for_saving_images 'Visualization_results' \
               --seed 1 \
               --gpu_ids '' \
               --num_workers 8 \
               --phylo_config ./configs/cub18_flat.yaml \
               --experiment_note "Flat structure noAL+UNI. Removed focal loss. Fixed descendant count problem. Added minmaximize loss, limiting it to cl and add on layers. Added softmax after cs. Base unit sphere model with 20 protos per node. Loading the pretrained backbone so setting epochs_pretrain 0. With bias in the addon layer. Protopool, no seperate classifiction layer for each child node. Not using softmax. Added finetune back this time it trains add-on along with classification. Using equalize aug as well, but keeping augment parameters to the new one. No meanpool. With 60 epochs of unit-sphere pretraining. Set meanpool kernel size to 2. Class loss doesnt affect convnext only AL+UNI does. Removed OOD again. first run after fixing all the memory issue. Pretrain->AL+UNI, finetune->CL, general training->AL+UNI+TANH_DESC+CL. fixed UW=0 now UW=2. unit sphere latent space. 4 per descendant. Saving every 30 epochs. Added csv logging for node wise losses. Added wandb for logging nodewise losses. Added OOD for 18species subset. Added kernel orthogonality on only relevant prototype kernels with loss-weight 0.5. Filtered imgs in vis_pipnet and fixed the previous issue. Separate add_on for each node. Using cropped images for projection. Removed scaling -> (len(node_y) / len(ys[ys != OOD_LABEL])). Set finetune to 0 and Set freeze_epochs to 30. Added OOD loss, removed pretrained backbone. 005 had incorrect data.py. Fixed it again. Reducing protos to 50 from 200 since there is a lot of meaningless prototypes in 004. Not Using backbone thats already trained with all 190 species. Limited protos to 200 bcoz of memory issue. Added wandb logging" \
               --kernel_orth "n" \
               --num_features 180 \
               --num_protos_per_descendant 0 \
               --copy_files "y" \
               --tanh_desc "n" \
               --align "n" \
               --uni "n" \
               --align_pf "n" \
               --minmaximize "n" \
               --unitconv2d "y" \
               --softmax "y" \
               --gumbel_softmax "n" \
               --gs_tau 1.0 \
               --multiply_cs_softmax "n" \
               --focal "n" \
               --wandb "y" \
               --weighted_ce_loss "n" \
               --focal_loss "n" \
               --focal_loss_gamma 2.0 \
               --protopool "y" \
               --stage4_reducer_net "" \
               --state_dict_dir_backbone "" \
               --basic_cnext_gaussian_multiplier "4|1.0|50" \
               # --leave_out_classes "cub_003_Sooty_Albatross,cub_051_Horned_Grebe,cub_032_Mangrove_Cuckoo" \
               # --state_dict_dir_backbone "/home/harishbabu/projects/PIPNet/runs/082-CUB-18-imgnet_with-equalize-aug_cnext26_img=224_nprotos=4per-leaf-desc_unit-sphere_finetune=5_no-meanpool_with-softmax_no-addon-bias_AW=3-TW=2-MMW=2-UW=3-CW=2_mm-loss_batch=48/checkpoints/net_pretrained" \
               # --state_dict_dir_net '/home/harishbabu/projects/PIPNet/runs/068-CUB-18-imgnet_with-equalize-aug_cnext26_img=224_nprotos=4per-desc_unit-sphere-protopool_finetune=5_no-meanpool_no-softmax_AW=3-TW=2-UW=3-CW=2_batch=20/checkpoints/net_pretrained' \
            #    --add_on_bias \
               # --OOD_dataset 'CUB-172-OOD-imgnet-224' \
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