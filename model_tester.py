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
Degradation_type = 'DownBlur'

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
from UNet_model_superres import Attention_UNet_superres,Residual_Attention_UNet_superres,Residual_Attention_UNet_superres_2,Residual_MultiHeadAttention_UNet_superres,Residual_Visual_MultiHeadAttention_UNet_superres

def model_tester(model_name_list, UNet_type_list, snapshot_name_list, test_img_lr, device, test_img_hr=None, save_path=None):
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
model_tester(['DDP_Residual_Attention_UNet_superres_magnification8_ANIME50k_DownBlur_512'],
             ['residual attention unet'],
              ['snapshot.pt'], test_img_lr, device, test_img_hr)
# model_tester(['DDP_Residual_Attention_UNet_superres_magnification4_celebA_GaussBlur', 
#               'Residual_Attention_UNet_superres_magnification4_celeb50k'],
#               ['snapshot.pt', 'snapshot.pt'], test_img_lr, device, test_img_hr)
# model_tester(['DDP_Residual_Attention_UNet_superres_magnification8_ANIME50k_GaussBlur_512', 
#               'DDP_Residual_Attention_UNet_superres_magnification8_ANIME50k_512'],
#               ['snapshot.pt', 'snapshot.pt'], test_img_lr, device, test_img_hr)
# model_tester(['DDP_Residual_Attention_UNet2_superres_magnification4_celebA_BlurDown', 
#               'DDP_Residual_Attention_UNet_superres_magnification4_celebA_BlurDown'],
#               ['residual attention unet 2', 'residual attention unet'],
#               ['snapshot.pt', 'snapshot.pt'], test_img_lr, device, test_img_hr)

# %%
from PIL import Image
from torchvision import transforms
from UNet_model_superres import Residual_Attention_UNet_superres
import os
from utils import get_data_superres
from torchvision import transforms
from train_diffusion_superres_COMPLETE import Diffusion

def superresoluter(lr_img):
    model_name = 'DDP_Residual_Attention_UNet_superres_EMA_magnification4_ANIME50k_DownBlur'
    device = 'mps'
    model = Residual_Attention_UNet_superres(3, 3, device).to(device)
    snapshot_path = os.path.join('models_run', model_name, 'weights','snapshot.pt')
    noise_schedule ='cosine'
    noise_steps = 1500
    magnification_factor = 4
    image_size = 256
    Degradation_type = 'DownBlur'

    diffusion = Diffusion(
            noise_schedule=noise_schedule, model=model,
            snapshot_path=snapshot_path,
            noise_steps=noise_steps, beta_start=1e-4, beta_end=0.02, 
            magnification_factor=magnification_factor,device=device,
            image_size=image_size, model_name=model_name, Degradation_type=Degradation_type)
    
    super_lr_img = diffusion.sample(1, model, lr_img, input_channels=3, plot_gif_bool=False)
    super_lr_img = super_lr_img.clip(0,1)
    super_lr_img = super_lr_img[0]
    super_lr_img = transforms.ToPILImage()(super_lr_img)
    return super_lr_img



transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
        ])
test_path = 'anime_data_10k/test_original'
magnification_factor = 4

test_data = get_data_superres(test_path, magnification_factor, 0.5, False, 'PIL', transform)

lr_img = test_data[5][0]
sr_img = superresoluter(lr_img)
# %%
lr_img = transforms.ToPILImage()(lr_img).resize((256,256),Image.BICUBIC)
from utils import convert_png_to_jpg
convert_png_to_jpg('lr_ANIME.png', 'lr_ANIME.jpg')
convert_png_to_jpg('sr_ANIME.png', 'sr_ANIME.jpg')
# %%

# %%
from utils import convert_png_to_jpg
convert_png_to_jpg('imgsli_2.png', 'imgsli_2.jpg')
# %%
