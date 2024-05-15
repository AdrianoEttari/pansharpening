#!/bin/bash

# Activate Conda environment
conda activate pansharpening

model_name='UNet_Faces_superres_TO_REMOVE'
python train_diffusion_superres_COMPLETE.py --epochs=1 --noise_schedule='cosine' --batch_size=8 --image_size=256 --lr=2e-4 --snapshot_name=snapshot.pt --model_name="$model_name" --noise_steps=2 --dataset_path='sentinel_data_s2' --inp_out_channels=3 --loss=MSE --magnification_factor=2 --UNet_type='Residual Attention UNet' --Degradation_type='DownBlurNoise' --multiple_gpus='False' --ema_smoothing='True'

# model_name='UNet_Faces_superres_EMA_MSE_CelebA_magnification4'
# python train_diffusion_superres_EMA.py --epochs=1 --noise_schedule='cosine' --batch_size=2 --image_size=172 --lr=5e-4 --snapshot_name=snapshot.pt --model_name="$model_name" --noise_steps=1500 --device='cpu' --dataset_path='celebA_25k' --inp_out_channels=3 --loss=MSE --magnification_factor=4

# model_name='UNet_SIRS_TO_REMOVE'
# python train_UNet.py --epochs=1 --batch_size=2 --image_size=224 --lr=3e-4 --snapshot_name=snapshot.pt --model_name="$model_name" --device='cpu' --dataset_path='anime_data_50k' --inp_out_channels=3 --loss=Perceptual --magnification_factor=4 --UNet_type='Attention UNet'

# model_name='UNet_Faces_superres_MSE_Perc3-7'
# python train_diffusion_superres_EMA.py --epochs=21 --noise_schedule='cosine' --batch_size=10 --image_size=224 --lr=3e-4 --snapshot_name=snapshot.pt --model_name="$model_name" --noise_steps=1500 --device='mps' --dataset_path='Humans_test' --inp_out_channels=3 --loss=MSE+Perceptual_imgs --magnification_factor=4

# model_name='UNet_animals_cosine'
# python train_diffusion.py --epochs=1 --batch_size=5 --image_size=224 --lr=3e-4 --noise_schedule='cosine' --snapshot_name=snapshot.pt --model_name="$model_name" --noise_steps=20 --device='mps' --dataset_path=animal_imgs --inp_out_channels=3 --num_classes=10

# model_name='UNet_MNIST_generation'
# python train_diffusion.py --epochs=1 --batch_size=64 --image_size=28 --lr=3e-4 --noise_schedule='cosine' --snapshot_name=snapshot.pt --model_name="$model_name" --noise_steps=20 --device='mps' --inp_out_channels=1 --num_classes=10

# model_name='DDP_Residual_Attention_UNet_superres_magnification4_celebA_DownBlur'
# python Aggregation_Sampling.py --noise_schedule='cosine' --snapshot_name=snapshot.pt --noise_steps=1500 --model_input_size=224 --model_name="$model_name" --Degradation_type='DownBlur' --device='mps' --magnification_factor=4 --inp_out_channels=3 --patch_size=56 --stride=32 --destination_path='./Abraham_Lincoln_sr.jpg' --img_lr_path='./Abraham_Lincoln_lr.jpg' --UNet_type='Residual Attention UNet' 