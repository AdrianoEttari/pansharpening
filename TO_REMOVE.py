#%%
import torch
from UNet_model_superres_new import Residual_Attention_UNet_superres,Attention_UNet_superres,Residual_Attention_UNet_superres_2,Residual_MultiHeadAttention_UNet_superres,Residual_Visual_MultiHeadAttention_UNet_superres
from PIL import Image, ImageFilter
from torchvision import transforms
import numpy as np
from torch.functional import F
import matplotlib.pyplot as plt
import os
from train_diffusion_superres import Diffusion
from tqdm import tqdm 
from Aggregation_Sampling import load_from_pickle, save_to_pickle, imgs_splitter, plot_patches, merge_images

model_name = 'DDP_Residual_Attention_UNet_superres_magnification8_ANIME50k_DownBlur_512'
magnification_factor = 8
snapshot_folder_path = os.path.join('models_run', model_name, 'weights')
snapshot_path = os.path.join(snapshot_folder_path, 'snapshot.pt')
UNet_type = 'residual attention unet'
noise_schedule = 'cosine'
image_size = 256
model_input_size = 64
number_of_patches = image_size // model_input_size
input_channels = output_channels = 3
device = 'mps'
Degradation_type = 'DownBlur'
noise_steps = 1500
blur_radius = 0.5

img_test = Image.open('anime_test.jpg')
transform = transforms.Compose([ transforms.ToTensor()])
img_test = transform(img_test)
img_test = img_test[:,:,list(np.arange(50,562))]
img_test = F.interpolate(img_test.unsqueeze(0), size=(image_size, image_size), mode='bicubic', align_corners=False).squeeze(0)
plt.imshow(img_test.permute(1,2,0))

img_test_2 = Image.open('anime_data_10k/test_original/175490.jpg')
downsample = transforms.Resize((512//magnification_factor, 512//magnification_factor),
                                       interpolation=transforms.InterpolationMode.BICUBIC)
img_test_2 = downsample(img_test_2)
img_test_2 = img_test_2.filter(ImageFilter.GaussianBlur(blur_radius))
img_test_2 = transforms.ToTensor()(img_test_2)
# %%
position_patch_dic = imgs_splitter(img_test, number_of_patches, model_input_size)

os.makedirs('aggregation_sampling',exist_ok=True)
save_to_pickle(os.path.join('aggregation_sampling','inputs.pkl'), position_patch_dic)
# %%
position_patch_dic = load_from_pickle(os.path.join('aggregation_sampling','inputs.pkl'))

plot_patches(position_patch_dic)
# %%
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
# %%

diffusion = Diffusion(
        noise_schedule=noise_schedule, model=model,
        snapshot_path=snapshot_path,
        noise_steps=noise_steps, beta_start=1e-4, beta_end=0.02, 
        magnification_factor=magnification_factor,device=device,
        image_size=512, model_name=model_name, Degradation_type=Degradation_type)
# %%
# position_super_lr_patches_dic = {}

# for key,value in tqdm(position_patch_dic.items()):
#     super_lr_patch = diffusion.sample(1, model, value.to(device), input_channels=3, plot_gif_bool=False)
#     super_lr_img_test_2 = diffusion.sample(1, model, img_test_2.to(device), input_channels=3, plot_gif_bool=False)
#     position_super_lr_patches_dic[key] = super_lr_patch.to('cpu')

# save_to_pickle(os.path.join('aggregation_sampling','predictions.pkl'), position_super_lr_patches_dic)

#%%
position_super_lr_patches_dic = load_from_pickle(os.path.join('aggregation_sampling','predictions.pkl'))

plot_patches(position_super_lr_patches_dic)

# %%
merged_image = merge_images(position_super_lr_patches_dic)
plt.imshow(merged_image.permute(1,2,0))
plt.axis('off')
plt.show()


# %%
