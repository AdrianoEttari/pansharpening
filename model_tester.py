#%% IMPORT DATA
import torch
from train_diffusion_superres_EMA import Diffusion
from torchvision import transforms
from utils import get_data_superres
from torch.utils.data import DataLoader
import os
from PIL import Image

image_size = 224
input_channels = output_channels = 3
device = 'mps'
noise_schedule='cosine'
noise_steps = 1500
dataset_path = os.path.join('anime_data')
magnification_factor = 2

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    ]) # The transforms.ToTensor() is in the get_data_superres function (in there
    # first is applied this transform to y, then the resize according to the magnification_factor
    # in order to get the x which is the lr_img and finally the to_tensor for both x
    # and y is applied)

train_path = f'{dataset_path}/train_original'
valid_path = f'{dataset_path}/val_original'
test_path = f'{dataset_path}/test_original'

train_dataset = get_data_superres(train_path, magnification_factor, 'PIL', transform)
val_dataset = get_data_superres(valid_path, magnification_factor, 'PIL', transform)
test_dataset = get_data_superres(test_path, magnification_factor, 'PIL', transform)

test_img_path = os.path.join('anime_data','test_original','Dr_Stone_6.jpg')
test_img_hr = Image.open(test_img_path)
test_img_hr = transform(test_img_hr)
downsample = transforms.Resize((test_img_hr.size[0] // magnification_factor, test_img_hr.size[1] // magnification_factor),
                                       interpolation=transforms.InterpolationMode.BICUBIC)
test_img_hr = transforms.ToTensor()(test_img_hr)
test_img_lr = downsample(test_img_hr)
test_img_lr = transforms.ToTensor()(test_img_lr)


#%% PLOT FUNCTION
import matplotlib.pyplot as plt

def plot_super_resolution_comparison(lr_img, hr_img, *super_lr_imgs):
    num_models = len(super_lr_imgs)
    if (num_models % 2 != 0) and (num_models != 1):
        num_cols = num_models//2
        num_rows = num_models//2 + 2
    elif num_models == 1:
        num_cols = 1
        num_rows = 3
    else:
        num_cols = num_models//2
        num_rows = num_models//2 + 1

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 5 * num_rows))
    axs = axs.ravel()
    # Plot low-resolution image
    axs[0].imshow(lr_img.permute(1,2,0).detach().cpu())
    axs[0].set_title('Low Resolution Image')
    axs[0].axis('off')

    # Plot high-resolution image
    axs[1].imshow(hr_img.permute(1,2,0).detach().cpu())
    axs[1].set_title('High Resolution Image')
    axs[1].axis('off')

    # Plot super resolution images from different models
    for i, super_lr_img in enumerate(super_lr_imgs):
        row = i + 2
        axs[row].imshow(super_lr_img[0].permute(1,2,0).detach().cpu())
        axs[row].set_title(f'Super Resolution Image {i+1}')
        axs[row].axis('off')

    # Hide empty subplots if there are less than 4 models
    for i in range(num_models + 1, num_rows):
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()
#%% SINGLE MODEL
from UNet_model_superres_new import Attention_UNet_superres

model_name = 'Attention_UNet_superres_magnification2_ANIME'
snapshot_folder_path = os.path.join('models_run', model_name, 'weights')
snapshot_name = 'snapshot.pt'

model = Attention_UNet_superres(input_channels, output_channels, device).to(device)
snapshot_path = os.path.join(snapshot_folder_path, snapshot_name)

diffusion = Diffusion(
        noise_schedule=noise_schedule, model=model,
        snapshot_path=snapshot_path,
        noise_steps=noise_steps, beta_start=1e-4, beta_end=0.02, 
        magnification_factor=magnification_factor,device=device,
        image_size=image_size, model_name=model_name)

# lr_img = test_dataset[10][0].to(device)
# hr_img = test_dataset[10][1].to(device)

super_lr_img = diffusion.sample(1, model, test_img_lr, input_channels=3, plot_gif_bool=False)

plot_super_resolution_comparison(test_img_lr, test_img_hr, super_lr_img)

#%% MULTI MODELS
width = train_dataset[0][1].shape[1]

from old_staff.UNet_model_superres import SimpleUNet_superres

model = SimpleUNet_superres(width, input_channels, output_channels, device).to(device)
model_name_1 = 'UNet_Faces_superres_EMA_PercLoss3-7'
snapshot_path_1 = '/Users/adrianoettari/Desktop/ASSEGNO_DI_RICERCA/pansharpening/models_run/UNet_Faces_superres_EMA_PercLoss3-7/weights/snapshot.pt'
diffusion_1 = Diffusion(
    noise_schedule=noise_schedule, model=model,
    snapshot_path=snapshot_path_1,
    noise_steps=noise_steps, beta_start=1e-4, beta_end=0.02, 
    magnification_factor=magnification_factor,device=device,
    image_size=image_size, model_name=model_name_1)

model_name_2 = 'UNet_Faces_superres_EMA_PercLoss2-8'
snapshot_path_2 = '/Users/adrianoettari/Desktop/ASSEGNO_DI_RICERCA/pansharpening/models_run/UNet_Faces_superres_EMA_PercLoss2-8/weights/snapshot.pt'
diffusion_2 = Diffusion(
    noise_schedule=noise_schedule, model=model,
    snapshot_path=snapshot_path_2,
    noise_steps=noise_steps, beta_start=1e-4, beta_end=0.02, 
    magnification_factor=magnification_factor,device=device,
    image_size=image_size, model_name=model_name_2)

model_name_3 = 'UNet_Faces_superres_EMA_MSE_2'
snapshot_path_3 = '/Users/adrianoettari/Desktop/ASSEGNO_DI_RICERCA/pansharpening/models_run/UNet_Faces_superres_EMA_MSE_2/weights/snapshot.pt'
diffusion_3 = Diffusion(
    noise_schedule=noise_schedule, model=model,
    snapshot_path=snapshot_path_3,
    noise_steps=noise_steps, beta_start=1e-4, beta_end=0.02, 
    magnification_factor=magnification_factor,device=device,
    image_size=image_size, model_name=model_name_3)

from UNet_model_superres_concat import SimpleUNet_superres as SimpleUNet_superres_concat
model_name_4 = 'UNet_Faces_superres_EMA_MSE_concatenation'
model_concat = SimpleUNet_superres_concat(width, input_channels, output_channels, device).to(device)
snapshot_path_4 = '/Users/adrianoettari/Desktop/ASSEGNO_DI_RICERCA/pansharpening/models_run/UNet_Faces_superres_EMA_MSE_concatenation/weights/snapshot.pt' 
diffusion_4 = Diffusion(
    noise_schedule=noise_schedule, model=model_concat,
    snapshot_path=snapshot_path_4,
    noise_steps=noise_steps, beta_start=1e-4, beta_end=0.02, 
    magnification_factor=magnification_factor,device=device,
    image_size=image_size, model_name=model_name_4)

lr_img = test_dataset[10][0].to(device)
hr_img = test_dataset[10][1].to(device)

super_lr_img_1 = diffusion_1.sample(1, model, lr_img, input_channels=3, plot_gif_bool=False)
super_lr_img_2 = diffusion_2.sample(1, model, lr_img, input_channels=3, plot_gif_bool=False)
super_lr_img_3 = diffusion_3.sample(1, model, lr_img, input_channels=3, plot_gif_bool=False)
super_lr_img_4 = diffusion_4.sample(1, model_concat, lr_img, input_channels=3, plot_gif_bool=False)

lr_img = lr_img.permute(1,2,0).detach().cpu().numpy()
hr_img = hr_img.permute(1,2,0).detach().cpu().numpy()
super_lr_img_1 = super_lr_img_1[0].permute(1,2,0).detach().cpu().numpy()
super_lr_img_2 = super_lr_img_2[0].permute(1,2,0).detach().cpu().numpy()
super_lr_img_3 = super_lr_img_3[0].permute(1,2,0).detach().cpu().numpy()
super_lr_img_4 = super_lr_img_4[0].permute(1,2,0).detach().cpu().numpy()


plot_super_resolution_comparison(lr_img, hr_img, super_lr_img_1, super_lr_img_2, super_lr_img_3, super_lr_img_4)

# %% GAUSSIAN BLUR, BICUBIC DOWNSAMPLIONG
from PIL import Image, ImageFilter
import numpy as np


img_path = '/Users/adrianoettari/Desktop/ASSEGNO_DI_RICERCA/pansharpening/Humans_test/train_original/1 (1).jpeg'

original_img = Image.open(img_path)
original_img = original_img.resize((224, 224), Image.BICUBIC)

radius = 2
blurred_image = original_img.filter(ImageFilter.GaussianBlur(radius))

downsampled_img = original_img.resize((224//4, 224//4), Image.BICUBIC).resize((224, 224), Image.BICUBIC)
downsampled_blurred_img = blurred_image.resize((224//4, 224//4), Image.BICUBIC).resize((224, 224), Image.BICUBIC)
