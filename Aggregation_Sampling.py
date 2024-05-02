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

model_name = 'DDP_Residual_Attention_UNet_superres_magnification3_ANIME10k_BSRdegrplus'
snapshot_path = f'/models_run/{model_name}/weights/snapshot.pt'
UNet_type = 'Residual Attention UNet'
noise_schedule = 'cosine'
image_size = 512
magnification_factor = 4
model_input_size = image_size // magnification_factor
input_channels = output_channels = 3
device = 'cpu'
Degradation_type = 'DownBlur'
noise_steps = 1500
blur_radius = 0.5

img_test = Image.open('anime_test.jpg')
transform = transforms.Compose([ transforms.ToTensor()])
img_test = transform(img_test)
img_test = img_test[:,:,list(np.arange(50,562))]
img_test = F.interpolate(img_test.unsqueeze(0), size=(image_size, image_size), mode='bilinear', align_corners=False).squeeze(0)
plt.imshow(img_test.permute(1,2,0))

# img_test = Image.open('anime_data_10k/test_original/175490.jpg')
# downsample = transforms.Resize((model_input_size, model_input_size),
#                                        interpolation=transforms.InterpolationMode.BICUBIC)
# img_test = downsample(img_test)
# img_test = img_test.filter(ImageFilter.GaussianBlur(blur_radius))
# img_test = transforms.ToTensor()(img_test)
# %%

number_of_patches = image_size//model_input_size

width_steps = height_steps = np.cumsum(np.repeat(model_input_size,number_of_patches))
width_steps = np.insert(width_steps,0,0)
height_steps = np.insert(height_steps,0,0)

small_imgs = {}
for i in range(number_of_patches):
    for j in range(number_of_patches):
        small_imgs[i,j]=img_test[:,width_steps[i]:width_steps[i+1],height_steps[j]:height_steps[j+1]]

import pickle
def load_from_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    
def save_to_pickle(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

os.makedirs('aggregation_sampling',exist_ok=True)
save_to_pickle(os.path.join('aggregation_sampling','inputs.pkl'), small_imgs)
# %%
small_imgs = load_from_pickle(os.path.join('aggregation_sampling','inputs.pkl'))

fig, axs = plt.subplots(number_of_patches,number_of_patches,figsize=(20,20))
for i in range(number_of_patches):
    for j in range(number_of_patches):
        axs[i,j].imshow(small_imgs[i,j].permute(1,2,0))
        axs[i,j].axis('off')
plt.show()
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
super_lr_imgs = {}

for key,value in tqdm(small_imgs.items()):
    super_lr_img = diffusion.sample(1, model, value.to(device), input_channels=3, plot_gif_bool=False)
    super_lr_imgs[key] = super_lr_img.to('cpu')

save_to_pickle(os.path.join('aggregation_sampling','predictions.pkl'), super_lr_imgs)

#%%
predictions = load_from_pickle(os.path.join('aggregation_sampling','predictions.pkl'))

fig, axs = plt.subplots(number_of_patches,number_of_patches,figsize=(20,20))
for i in range(number_of_patches):
    for j in range(number_of_patches):
        axs[i,j].imshow(predictions[i,j][0].permute(1,2,0))
        axs[i,j].axis('off')
plt.show()

# %%
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def merge_images(image_dict, grid_size):
    '''
    This function merges a dictionary of images into a single image based on their positions in a grid.
    The image_dict dictionary should have tuples as keys representing the position of the image in the grid (as tuples),
    and the values should be the tensor images themselves.
    '''
    # Calculate the size of the merged image
    merged_height = grid_size[0] * next(iter(image_dict.values())).size(1)
    merged_width = grid_size[1] * next(iter(image_dict.values())).size(2)

    # Create a blank canvas for the merged image
    merged_image = torch.zeros(3, merged_height, merged_width)

    # Merge images onto the canvas based on their positions
    for position, image in image_dict.items():
        row_start = position[0] * image.size(1)
        row_end = row_start + image.size(1)
        col_start = position[1] * image.size(2)
        col_end = col_start + image.size(2)

        merged_image[:, row_start:row_end, col_start:col_end] = image

    return merged_image


merged_image = merge_images(small_imgs, (4,4))
plt.imshow(merged_image.permute(1,2,0))
plt.axis('off')
plt.show()


# %%
