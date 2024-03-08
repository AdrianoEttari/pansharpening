#!/bin/bash

conda activate pansharpening
model_name='UNet_Faces_superres'
python train_diffusion_superres.py --epochs=1 --noise_schedule='cosine' --batch_size=128 --image_size=224 --lr=3e-4 --snapshot_name=snapshot.pt --model_name="$model_name" --noise_steps=2 --device='cpu' --dataset_path='Human_Action_Recognition' --inp_out_channels=3 --magnification_factor=4

# model_name='UNet_MNIST_superres'
# python train_diffusion_superres.py --epochs=1 --noise_schedule='cosine' --batch_size=128 --image_size=28 --lr=3e-4 --snapshot_name=snapshot.pt --model_name="$model_name" --noise_steps=2 --device='cpu' --dataset_path='MNIST_data_superres' --inp_out_channels=1 --magnification_factor=4

# model_name='UNet_animals_cosine'
# python train_diffusion.py --epochs=1 --batch_size=5 --image_size=224 --lr=3e-4 --noise_schedule='cosine' --snapshot_name=snapshot.pt --model_name="$model_name" --noise_steps=20 --device='mps' --dataset_path=animal_imgs --inp_out_channels=3 --num_classes=10

# model_name='UNet_MNIST_generation'
# python train_diffusion.py --epochs=1 --batch_size=64 --image_size=28 --lr=3e-4 --noise_schedule='cosine' --snapshot_name=snapshot.pt --model_name="$model_name" --noise_steps=20 --device='mps' --inp_out_channels=1 --num_classes=10

