#%% IMPORT DATA
import torch
from train_diffusion_superres import Diffusion
from torchvision import transforms
from utils import get_data_superres
from torch.utils.data import DataLoader
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

image_size = 512
input_channels = output_channels = 3
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'mps'
noise_schedule='cosine'
noise_steps = 1500
dataset_path = os.path.join('anime_data_50k')
magnification_factor = 8
Degradation_type = 'BlurDown'

transform = transforms.Compose([
    transforms.Resize((image_size, image_size),interpolation=transforms.InterpolationMode.BICUBIC),
    ]) # The transforms.ToTensor() is in the get_data_superres function (in there
    # first is applied this transform to y, then the resize according to the magnification_factor
    # in order to get the x which is the lr_img and finally the to_tensor for both x
    # and y is applied)

train_path = f'{dataset_path}/train_original'
valid_path = f'{dataset_path}/val_original'
test_path = f'{dataset_path}/test_original'

train_dataset = get_data_superres(root_dir=train_path, magnification_factor=magnification_factor,blur_radius=0.5, data_format='PIL', transform=transform)
val_dataset = get_data_superres(root_dir=valid_path, magnification_factor=magnification_factor,blur_radius=0.5, data_format='PIL', transform=transform)
test_dataset = get_data_superres(root_dir=test_path, magnification_factor=magnification_factor,blur_radius=0.5, data_format='PIL', transform=transform)

test_img_lr = test_dataset[110][0]
test_img_hr = test_dataset[110][1]

# test_img_path = 'anime_test.jpg'
# test_img_hr = Image.open(test_img_path)
# test_img_hr = transform(test_img_hr)
# downsample = transforms.Resize((test_img_hr.size[0] // magnification_factor, test_img_hr.size[1] // magnification_factor),
#                                        interpolation=transforms.InterpolationMode.BICUBIC)
# test_img_hr = transforms.ToTensor()(test_img_hr)
# test_img_lr = downsample(test_img_hr)
# test_img_lr = test_img_lr[:3,:,:]

# folder_test_img_lr_path = 'satellite_imgs_test'
# transform_lr = transforms.Compose([
#     transforms.Resize((image_size // magnification_factor, image_size // magnification_factor),interpolation=transforms.InterpolationMode.BICUBIC),
#     ])

#%% 
from UNet_model_superres_new import Attention_UNet_superres,Residual_Attention_UNet_superres

def model_tester(model_name_list, snapshot_name_list, test_img_lr, device, test_img_hr=None, save_path=None):
    super_lr_imgs = []
    for model_name, snapshot_name in zip(model_name_list, snapshot_name_list):
        snapshot_folder_path = os.path.join('models_run', model_name, 'weights')
        if 'residual' in model_name.lower():
            model = Residual_Attention_UNet_superres(input_channels, output_channels, device).to(device)
        else:
            model = Attention_UNet_superres(input_channels, output_channels, device).to(device)
        snapshot_path = os.path.join(snapshot_folder_path, snapshot_name)

        image_size = test_img_lr.shape[-1] * magnification_factor

        diffusion = Diffusion(
                noise_schedule=noise_schedule, model=model,
                snapshot_path=snapshot_path,
                noise_steps=noise_steps, beta_start=1e-4, beta_end=0.02, 
                magnification_factor=magnification_factor,device=device,
                image_size=image_size, model_name=model_name, Degradation_type=Degradation_type)

        super_lr_img = diffusion.sample(1, model, test_img_lr, input_channels=3, plot_gif_bool=False)
        super_lr_imgs.append(super_lr_img)


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
        # residual_img = np.abs(residual_img.permute(1,2,0).detach().cpu().numpy())
        # axs[i+1,1].imshow(residual_img, cmap='hot', interpolation='nearest')
        # axs[i+1,1].set_title('Residual Image')

    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()
# model_tester(['Attention_UNet_superres_magnification4_celeb100k'], ['snapshot_NOT_END.pt'], test_img_lr, device, test_img_hr)
# model_tester(['Attention_UNet_superres_magnification2_ANIME50k', 'Residual_Attention_UNet_superres_magnification2_ANIME50k'],
#              ['snapshot.pt', 'snapshot.pt'], test_img_lr, device, test_img_hr)
# model_tester(['Residual_Attention_UNet_superres_magnification4_ANIME50k','Attention_UNet_superres_magnification4_ANIME50k'],
#               ['snapshot.pt','snapshot.pt'], test_img_lr, device, test_img_hr)
# model_tester(['DDP_Residual_Attention_UNet_superres_magnification8_ANIME50k_512'],
#               ['snapshot.pt'], test_img_lr, device, test_img_hr)
# model_tester(['DDP_Residual_Attention_UNet_superres_magnification4_celebA_GaussBlur', 
#               'Residual_Attention_UNet_superres_magnification4_celeb50k'],
#               ['snapshot.pt', 'snapshot.pt'], test_img_lr, device, test_img_hr)
model_tester(['DDP_Residual_Attention_UNet_superres_magnification8_ANIME50k_GaussBlur_512', 
              'DDP_Residual_Attention_UNet_superres_magnification8_ANIME50k_512'],
              ['snapshot.pt', 'snapshot.pt'], test_img_lr, device, test_img_hr)

# sr_test_path = 'sr_satellite_imgs_test'
# for test_img_lr_path in os.listdir(folder_test_img_lr_path):
#     test_img_lr_full_path = os.path.join(folder_test_img_lr_path, test_img_lr_path)
#     test_img_lr = Image.open(test_img_lr_full_path)
#     test_img_lr = transform_lr(test_img_lr)
#     test_img_lr = transforms.ToTensor()(test_img_lr)
#     print(test_img_lr.shape)
#     model_tester(['DDP_Residual_Attention_UNet_superres_magnification4_DIV2k_512'],
#               ['snapshot.pt'], test_img_lr, device, save_path=os.path.join(sr_test_path, test_img_lr_path))

# sr_test_path = 'sr_DIV2k'
# for i, img_path in enumerate(test_dataset):
#     test_img_lr = img_path[0]
#     test_img_hr = img_path[1]
#     model_tester(['DDP_Residual_Attention_UNet_superres_magnification4_DIV2k_512'],
#               ['snapshot.pt'], test_img_lr, device, test_img_hr, save_path=os.path.join(sr_test_path, f'{i}.png'))


# %%
