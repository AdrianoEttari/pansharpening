#%% IMPORT DATA
import torch
from train_diffusion_superres_COMPLETE import Diffusion
from torchvision import transforms
from utils import get_data_superres
from torch.utils.data import DataLoader
import os
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np

image_size = 512
input_channels = output_channels = 3
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'mps'
noise_schedule='cosine'
noise_steps = 1500
dataset_path = os.path.join('anime_data_50k')
magnification_factor = 4
Degradation_type = 'DownBlur'

# transform = transforms.Compose([
#     transforms.Resize((image_size, image_size),interpolation=transforms.InterpolationMode.BICUBIC),
#     ]) 

# test_path = f'{dataset_path}/test_original'

# test_dataset = get_data_superres(root_dir=test_path, magnification_factor=magnification_factor,blur_radius=0.5, data_format='PIL', transform=transform)

# test_img_lr = test_dataset[110][0]
# test_img_hr = test_dataset[110][1]

data_path = 'Haykyu_anime_test.jpg'
blur_radius = 0.5

img = Image.open(data_path)
img_lr = img.resize((128,128), Image.BICUBIC)
img_lr = img_lr.filter(ImageFilter.GaussianBlur(blur_radius))

to_tensor = transforms.ToTensor()
test_img_lr = to_tensor(img_lr)
test_img_hr = to_tensor(img.resize((512,512), Image.BICUBIC))

#%% 
from UNet_model_superres import Attention_UNet_superres,Residual_Attention_UNet_superres,Residual_Attention_UNet_superres_2,Residual_MultiHeadAttention_UNet_superres,Residual_Visual_MultiHeadAttention_UNet_superres
from torchvision import transforms

def model_tester(model_name_list, UNet_type_list, snapshot_name_list, test_img_lr, device, test_img_hr=None, destination_folder_path=None):
    '''
    This function allows to compare the results of different models on the super resolution task.
    '''
    super_lr_imgs = []
    for model_name, UNet_type, snapshot_name in zip(model_name_list,UNet_type_list, snapshot_name_list):
        snapshot_folder_path = os.path.join('models_run', model_name, 'weights')

        if UNet_type.lower() == 'attention unet':
            print('Using Attention UNet')
            model = Attention_UNet_superres(input_channels, output_channels, device).to(device)
        elif UNet_type.lower() == 'residual attention unet':
            print('Using Residual Attention UNet')
            model = Residual_Attention_UNet_superres(input_channels, output_channels, device).to(device)
        elif UNet_type.lower() == 'residual attention unet 2':
            print('Using Residual Attention UNet 2')
            model = Residual_Attention_UNet_superres_2(input_channels, output_channels, device).to(device)
        elif UNet_type.lower() == 'residual multihead attention unet':
            print('Using Residual MultiHead Attention UNet')
            model = Residual_MultiHeadAttention_UNet_superres(input_channels, output_channels, device).to(device)
        elif UNet_type.lower() == 'residual visual multihead attention unet':
            print('Using Residual Visual MultiHead Attention UNet')
            model = Residual_Visual_MultiHeadAttention_UNet_superres(input_channels, image_size ,output_channels, device).to(device)
        else:
            raise ValueError('The UNet type must be either Attention UNet or Residual Attention UNet or Residual Attention UNet 2 or Residual MultiHead Attention UNet or Residual Visual MultiHeadAttention UNet superres')

        snapshot_path = os.path.join(snapshot_folder_path, snapshot_name)

        image_size = test_img_lr.shape[-1] * magnification_factor

        diffusion = Diffusion(
                noise_schedule=noise_schedule, model=model,
                snapshot_path=snapshot_path,
                noise_steps=noise_steps, beta_start=1e-4, beta_end=0.02, 
                magnification_factor=magnification_factor,device=device,
                image_size=image_size, model_name=model_name, Degradation_type=Degradation_type,
                multiple_gpus=False, ema_smoothing=False)

        super_lr_img = diffusion.sample(1, model, test_img_lr, input_channels=3, plot_gif_bool=False)
        super_lr_imgs.append(super_lr_img)

    if destination_folder_path is not None:
        os.makedirs(destination_folder_path, exist_ok=True)
        for i,super_lr_img in enumerate(super_lr_imgs):
            transforms.ToPILImage()(super_lr_img[0].detach().cpu()).save(f'{destination_folder_path}/{model_name_list[i]}_superres.jpg')

    fig, axs = plt.subplots(len(super_lr_imgs)+1,2, figsize=(10 * len(super_lr_imgs), 10))
    
    axs[0,0].imshow(test_img_lr.permute(1,2,0).detach().cpu())
    axs[0,0].set_title('Low Resolution Image')
    if test_img_hr is not None:
        axs[0,1].imshow(test_img_hr.permute(1,2,0).detach().cpu())
        axs[0,1].set_title('High Resolution Image')
    for i,super_lr_img in enumerate(super_lr_imgs):
        axs[i+1,0].imshow(super_lr_img[0].permute(1,2,0).detach().cpu())
        axs[i+1,0].set_title(model_name_list[i])
        residual_img = test_img_hr.to(device) - super_lr_img[0].to(device)
        non_zero_values = residual_img.permute(1,2,0).detach().cpu().ravel()[residual_img.permute(1,2,0).detach().cpu().ravel() != 0]
        bins = np.linspace(non_zero_values.min(), non_zero_values.max(), num=50)
        axs[i+1,1].hist(non_zero_values, bins=bins, alpha=0.7, color='blue', edgecolor='black')
        axs[i+1,1].set_title('Residual Histogram (Non-Zero Values)')

    plt.tight_layout()
    plt.show()

model_tester(model_name_list=['DDP_Residual_Attention_UNet_superres_magnification4_ANIME50k_DownBlur','DDP_Residual_Attention_UNet_superres_EMA_magnification4_ANIME50k_DownBlur'],
             UNet_type_list=['residual attention unet', 'residual attention unet'],
              snapshot_name_list=['snapshot.pt','snapshot.pt'], 
                test_img_lr=test_img_lr,
                 device=device,
                   test_img_hr=test_img_hr,
                    destination_folder_path='ANIME_superres_results')


# %%