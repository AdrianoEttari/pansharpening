# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% super resolution sampling %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import torch
import os
from utils import get_data_superres
from torchvision import transforms
from UNet_model_superres import Residual_Attention_UNet_superres
from train_diffusion_superres_COMPLETE import Diffusion
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random


def super_resolver(img_path, device, model_name):
        noise_schedule = 'cosine'
        noise_steps = 1500
        blur_radius = 0.5
        hr_img = Image.open(img_path)

        magnification_factor = int([info[13:] for info in model_name.split('_') if info.startswith('magnification')][0])
        image_size = int([info[7:] for info in model_name.split('_') if info.startswith('imgsize')][0]) * magnification_factor

        transform = transforms.Compose([transforms.Resize((image_size, image_size))])
        hr_img = transform(hr_img)
        downsample = transforms.Resize((hr_img.size[0] // magnification_factor, hr_img.size[1] // magnification_factor),
                                interpolation=transforms.InterpolationMode.BICUBIC)
        lr_img = downsample(hr_img)
        lr_img = lr_img.filter(ImageFilter.GaussianBlur(blur_radius))
        to_tensor = transforms.ToTensor()
        lr_img = to_tensor(lr_img).to(device)
        hr_img = to_tensor(hr_img).to(device)

        input_channels = output_channels = hr_img.shape[0]

        model = Residual_Attention_UNet_superres(input_channels, output_channels, device).to(device)
        snapshot_path = os.path.join('models_run', model_name, 'weights', 'snapshot.pt')

        print(f'HR Image size: {image_size}, LR Image size: {image_size//magnification_factor} Magnification factor: {magnification_factor}, Channels: {input_channels}')
        Degradation_type = 'DownBlur'

        diffusion = Diffusion(
                noise_schedule=noise_schedule, model=model,
                snapshot_path=snapshot_path,
                noise_steps=noise_steps, beta_start=1e-4, beta_end=0.02, 
                magnification_factor=magnification_factor,device=device,
                image_size=image_size, model_name=model_name, Degradation_type=Degradation_type)
        
        superres_img = diffusion.sample(n=1,model=model, lr_img=lr_img, input_channels=lr_img.shape[0], generate_video=False)
        superres_img = torch.clamp(superres_img, 0, 1)

        return superres_img, lr_img, hr_img




def plot_lr_hr_sr(lr_img, hr_img, sr_img, histogram=True):
        if histogram:
                fig, axs = plt.subplots(2,3, figsize=(15,10))
                title_font = {'family': 'sans-serif', 'weight': 'bold', 'size': 15}

                axs = axs.ravel()
                axs[0].imshow(lr_img.permute(1,2,0).detach().cpu())
                axs[0].set_title('low resolution image', fontdict=title_font)
                axs[1].imshow(hr_img.permute(1,2,0).detach().cpu())
                axs[1].set_title('high resolution image', fontdict=title_font)
                axs[2].imshow(sr_img[0].permute(1,2,0).detach().cpu())
                axs[2].set_title('super resolution image', fontdict=title_font)
                axs[3].hist(lr_img.flatten().detach().cpu(), bins=100)
                axs[3].set_title('lr image histogram', fontdict=title_font)
                axs[4].hist(hr_img.flatten().detach().cpu(), bins=100)
                axs[4].set_title('hr image histogram', fontdict=title_font)
                axs[5].hist(sr_img.flatten().detach().cpu(), bins=100)
                axs[5].set_title('sr image histogram', fontdict=title_font)

                plt.show()
        else:
                fig, axs = plt.subplots(1,3, figsize=(15,10))
                title_font = {'family': 'sans-serif', 'weight': 'bold', 'size': 15}
                axs = axs.ravel()
                axs[0].imshow(lr_img.permute(1,2,0).detach().cpu())
                axs[0].set_title('low resolution image', fontdict=title_font)
                axs[1].imshow(hr_img.permute(1,2,0).detach().cpu())
                axs[1].set_title('high resolution image', fontdict=title_font)
                axs[2].imshow(sr_img[0].permute(1,2,0).detach().cpu())
                axs[2].set_title('super resolution image', fontdict=title_font)
                plt.show()


if __name__ == '__main__':
        device = 'mps'
        img_path = os.path.join('imgs_sample','up42_sample_lr.png')
        model_name = 'DDP_Residual_Attention_UNet_superres_magnification2_LRimgsize128_up42_sentinel2_patches_downblur'
        superres_img, lr_img, hr_img = super_resolver(img_path, device, model_name)
        plot_lr_hr_sr(lr_img, hr_img, superres_img, histogram=False)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SAR to NDVI sampling %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# import torch
# import os
# from torchvision import transforms
# from UNet_model_SAR_TO_NDVI import Residual_Attention_UNet_SAR_TO_NDVI
# from train_diffusion_SAR_TO_NDVI_COMPLETE import Diffusion
# from utils import get_data_SAR_TO_NDVI
# import matplotlib.pyplot as plt

# device = 'cuda'
# SAR_channels=2
# NDVI_channels=1

# image_size = 128
# noise_schedule = 'cosine'
# noise_steps = 1500
# model_name = 'DDP_Residual_Attention_EMA_UNet_SAR_TO_NDVI'

# model = Residual_Attention_UNet_SAR_TO_NDVI(SAR_channels, NDVI_channels, device).to(device)

# snapshot_path = os.path.join('models_run', model_name, 'weights', 'snapshot.pt')

# diffusion = Diffusion(
#         noise_schedule=noise_schedule, model=model,
#         snapshot_path=snapshot_path,
#         noise_steps=noise_steps, beta_start=1e-4, beta_end=0.02,device=device,
#         image_size=image_size, model_name=model_name,
#         multiple_gpus=False, ema_smoothing=False)

# test_path = os.path.join('imgs_sample', 'test_SAR_TO_NDVI')
# list_of_files = ['Victoria_0_20210830_patch_289.pt','Victoria_0_20210719_patch_214.pt',
#                  'Victoria_0_20180623_patch_62.pt', 'Victoria_2_20201115_patch_44.pt',
#                 'Victoria_2_20200630_patch_81.pt',
#                  'Victoria_2_20200730_patch_103.pt', 'Victoria_0_20220819_patch_267.pt']

# os.makedirs(os.path.join('imgs_sample', 'generated_SAR_TO_NDVI'), exist_ok=True)

# for i in range(len(list_of_files)):
#     os.makedirs(os.path.join('imgs_sample', 'generated_SAR_TO_NDVI', list_of_files[i]), exist_ok=True)
#     SAR_img = (torch.load(os.path.join(test_path, 'sar', list_of_files[i]))+1)/2
#     NDVI_img = (torch.load(os.path.join(test_path, 'opt', list_of_files[i]))+1)/2
#     NDVI_pred_img = diffusion.sample(n=5,model=model, SAR_img=SAR_img, NDVI_channels=NDVI_channels, generate_video=False)
#     plt.imshow(SAR_img[0].unsqueeze(0).permute(1,2,0).detach().cpu())
#     plt.xticks([])
#     plt.yticks([])
#     plt.savefig(os.path.join('imgs_sample', 'generated_SAR_TO_NDVI', list_of_files[i], f'SAR_0.png'))
#     plt.imshow(SAR_img[1].unsqueeze(0).permute(1,2,0).detach().cpu())
#     plt.xticks([])
#     plt.yticks([])
#     plt.savefig(os.path.join('imgs_sample', 'generated_SAR_TO_NDVI', list_of_files[i], f'SAR_1.png'))
#     plt.imshow(NDVI_img.permute(1,2,0).detach().cpu())
#     plt.xticks([])
#     plt.yticks([])
#     plt.savefig(os.path.join('imgs_sample', 'generated_SAR_TO_NDVI', list_of_files[i], f'NDVI_real.png'))
#     for j in range(5):
#         plt.imshow(NDVI_pred_img[j].permute(1,2,0).detach().cpu())
#         plt.xticks([])
#         plt.yticks([])
#         plt.savefig(os.path.join('imgs_sample', 'generated_SAR_TO_NDVI', list_of_files[i], f'NDVI_pred_{j}.png'))


# test_dataset = get_data_SAR_TO_NDVI(test_path)
# SAR_img = test_dataset[101][0]
# NDVI_img = test_dataset[101][1]

# NDVI_pred_img = diffusion.sample(n=5,model=model, SAR_img=SAR_img, NDVI_channels=NDVI_channels, generate_video=False)
# fig, axs = plt.subplots(1,5, figsize=(15,10))
# axs = axs.ravel()
# for i in range(5):
#     axs[i].imshow(NDVI_pred_img[i].permute(1,2,0).detach().cpu())
#     axs[i].set_title(f'NDVI predicted {i}')
# plt.show()

# fig, axs = plt.subplots(2,4, figsize=(15,10))
# title_font = {'family': 'sans-serif', 'weight': 'bold', 'size': 15}
# axs = axs.ravel()
# axs[0].imshow(SAR_img[0].unsqueeze(0).permute(1,2,0).detach().cpu())
# axs[0].set_title('SAR channel 1', fontdict=title_font)
# axs[1].imshow(SAR_img[1].unsqueeze(0).permute(1,2,0).detach().cpu())
# axs[1].set_title('SAR channel 2', fontdict=title_font)
# axs[2].imshow(NDVI_img.permute(1,2,0).detach().cpu())
# axs[2].set_title('NDVI real', fontdict=title_font)
# axs[3].imshow(NDVI_pred_img[0].permute(1,2,0).detach().cpu())
# axs[3].set_title('NDVI predicted', fontdict=title_font)
# axs[4].hist(SAR_img[0].unsqueeze(0).flatten().detach().cpu(), bins=100)
# axs[4].set_title('SAR channel 1 histogram', fontdict=title_font)
# axs[5].hist(SAR_img[1].unsqueeze(0).flatten().detach().cpu(), bins=100)
# axs[5].set_title('SAR channel 2 histogram', fontdict=title_font)
# axs[6].hist(NDVI_img.flatten().detach().cpu(), bins=100)
# axs[6].set_title('NDVI real histogram', fontdict=title_font)
# axs[7].hist(NDVI_pred_img[0].flatten().detach().cpu(), bins=100)
# axs[7].set_title('NDVI predicted histogram', fontdict=title_font)
# plt.show()


# %%
