# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% super resolution sampling %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import torch
import os
from utils import get_data_superres
from torchvision import transforms
from UNet_model_superres import Residual_Attention_UNet_superres
from train_diffusion_superres_COMPLETE import Diffusion
import matplotlib.pyplot as plt
from PIL import Image,ImageFilter
import random

input_channels = output_channels = 3
device = 'mps'
image_size = 256
magnification_factor = 2
noise_schedule = 'cosine'
noise_steps = 1500
model_name = 'DDP_Residual_Attention_UNet_superres_magnification2_up42_sentinel2_patches_downblurnoise'
Degradation_type = 'DownBlurNoise'
blur_radius = 0.5

model = Residual_Attention_UNet_superres(input_channels, output_channels, device).to(device)

snapshot_path = os.path.join('models_run', model_name, 'weights', 'snapshot.pt')

diffusion = Diffusion(
    noise_schedule=noise_schedule, model=model,
    snapshot_path=snapshot_path,
    noise_steps=noise_steps, beta_start=1e-4, beta_end=0.02, 
    magnification_factor=magnification_factor,device=device,
    image_size=image_size, model_name=model_name, Degradation_type=Degradation_type)

# transform = transforms.Compose([
#         transforms.Resize((image_size, image_size)),
#         ])
# test_path = os.path.join('sentinel_data_s2', 'test_original')
# test_dataset = get_data_superres(test_path, magnification_factor, blur_radius, False, 'PIL', transform)
# lr_img, hr_img = test_dataset[101]
# lr_img = lr_img.to(device)
# hr_img = hr_img.to(device)

hr_img = Image.open('/Users/adrianoettari/Desktop/ASSEGNO_DI_RICERCA/pansharpening/imgs_sample/up42_sample.png')
transform = transforms.Compose([transforms.Resize((image_size, image_size))])
hr_img = transform(hr_img)
downsample = transforms.Resize((hr_img.size[0] // magnification_factor, hr_img.size[1] // magnification_factor),
                                interpolation=transforms.InterpolationMode.BICUBIC)
lr_img = downsample(hr_img)
lr_img = lr_img.filter(ImageFilter.GaussianBlur(blur_radius))
to_tensor = transforms.ToTensor()
lr_img = to_tensor(lr_img).to(device)
hr_img = to_tensor(hr_img).to(device)

superres_img = diffusion.sample(n=1,model=model, lr_img=lr_img, input_channels=lr_img.shape[0], plot_gif_bool=False)
superres_img = torch.clamp(superres_img, 0, 1)

# from PIL import Image
# superres_img_tosave = transforms.ToPILImage()(superres_img[0].cpu())
# lr_img_tosave = transforms.ToPILImage()(lr_img.cpu())
# lr_img_tosave = lr_img_tosave.resize((image_size, image_size), Image.BICUBIC)
# lr_img_tosave.save('lr_img_anime_data.jpg')
# superres_img_tosave.save('superres_img_anime_data.jpg')

fig, axs = plt.subplots(2,3, figsize=(15,10))
title_font = {'family': 'sans-serif', 'weight': 'bold', 'size': 15}

axs = axs.ravel()
axs[0].imshow(lr_img.permute(1,2,0).detach().cpu())
axs[0].set_title('low resolution image', fontdict=title_font)
axs[1].imshow(hr_img.permute(1,2,0).detach().cpu())
axs[1].set_title('high resolution image', fontdict=title_font)
axs[2].imshow(superres_img[0].permute(1,2,0).detach().cpu())
axs[2].set_title('super resolution image', fontdict=title_font)
axs[3].hist(lr_img.flatten().detach().cpu(), bins=100)
axs[3].set_title('lr image histogram', fontdict=title_font)
axs[4].hist(hr_img.flatten().detach().cpu(), bins=100)
axs[4].set_title('hr image histogram', fontdict=title_font)
axs[5].hist(superres_img.flatten().detach().cpu(), bins=100)
axs[5].set_title('sr image histogram', fontdict=title_font)

plt.show()





# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SAR to NDVI sampling %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# import torch
# import os
# from torchvision import transforms
# from UNet_model_SAR_TO_NDVI import Residual_Attention_UNet_SAR_TO_NDVI
# from train_diffusion_SAR_TO_NDVI_COMPLETE import Diffusion
# from utils import get_data_SAR_TO_NDVI
# import matplotlib.pyplot as plt

# test_path = os.path.join('SAR_TO_NDVI_dataset', 'test')
# test_dataset = get_data_SAR_TO_NDVI(test_path)

# device = 'cpu'
# SAR_channels=2
# NDVI_channels=1


# image_size = 128
# noise_schedule = 'cosine'
# noise_steps = 1500
# model_name = 'DDP_Residual_Attention_UNet_SAR_TO_NDVI'

# model = Residual_Attention_UNet_SAR_TO_NDVI(SAR_channels, NDVI_channels, device).to(device)

# snapshot_path = os.path.join('models_run', model_name, 'weights', 'snapshot.pt')

# diffusion = Diffusion(
#         noise_schedule=noise_schedule, model=model,
#         snapshot_path=snapshot_path,
#         noise_steps=noise_steps, beta_start=1e-4, beta_end=0.02,device=device,
#         image_size=image_size, model_name=model_name,
#         multiple_gpus=False, ema_smoothing=False)

# SAR_img = test_dataset[101][0]

# NDVI_pred_img = diffusion.sample(n=1,model=model, SAR_img=SAR_img, NDVI_channels=NDVI_channels, plot_gif_bool=False)

# fig, axs = plt.subplots(2,4, figsize=(15,10))
# title_font = {'family': 'sans-serif', 'weight': 'bold', 'size': 15}
# axs = axs.ravel()
# axs[0].imshow(test_dataset[101][0][0].unsqueeze(0).permute(1,2,0).detach().cpu())
# axs[0].set_title('SAR channel 1', fontdict=title_font)
# axs[1].imshow(test_dataset[101][0][1].unsqueeze(0).permute(1,2,0).detach().cpu())
# axs[1].set_title('SAR channel 2', fontdict=title_font)
# axs[2].imshow(test_dataset[101][1].permute(1,2,0).detach().cpu())
# axs[2].set_title('NDVI real', fontdict=title_font)
# axs[3].imshow(NDVI_pred_img[0].permute(1,2,0).detach().cpu())
# axs[3].set_title('NDVI predicted', fontdict=title_font)
# axs[4].hist(test_dataset[101][0][0].unsqueeze(0).flatten().detach().cpu(), bins=100)
# axs[4].set_title('SAR channel 1 histogram', fontdict=title_font)
# axs[4].set_xlim(0, 1)  # Set x-axis range
# axs[4].set_ylim(0, 800)  # Set y-axis range
# axs[5].hist(test_dataset[101][0][1].unsqueeze(0).flatten().detach().cpu(), bins=100)
# axs[5].set_title('SAR channel 2 histogram', fontdict=title_font)
# axs[5].set_xlim(0, 1)  # Set x-axis range
# axs[5].set_ylim(0, 800)  # Set y-axis range
# axs[6].hist(test_dataset[101][1].flatten().detach().cpu(), bins=100)
# axs[6].set_title('NDVI real histogram', fontdict=title_font)
# axs[6].set_xlim(0, 1)  # Set x-axis range
# axs[6].set_ylim(0, 800)  # Set y-axis range
# axs[7].hist(NDVI_pred_img[0].flatten().detach().cpu(), bins=100)
# axs[7].set_title('NDVI predicted histogram', fontdict=title_font)
# axs[7].set_xlim(0, 1)  # Set x-axis range
# axs[7].set_ylim(0, 800)  # Set y-axis range
# plt.show()


# %%
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

lr_img = Image.open(r'C:\Users\ettari\Desktop\pansharpening\assets\UP42_SUPERRESOLUTION\DownBlur\lr_image_1.png')
lr_img = np.array(lr_img)
sr_img_no_agg = Image.open(r'C:\Users\ettari\Desktop\pansharpening\assets\UP42_SUPERRESOLUTION\DownBlur\sr_image_1_no_agg.png')
# first columns and then rows
patch_lr_img = lr_img[3850:4100, 1500:1750, :3]
patch_sr_img_no_agg = sr_img_no_agg[7700:8200, 3000:3500, :3]

plt.imshow(patch_lr_img)
plt.show()
plt.imshow(patch_sr_img_no_agg)
plt.show()
# %%
