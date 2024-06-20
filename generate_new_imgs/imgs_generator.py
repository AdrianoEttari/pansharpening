# %%
from train_diffusion_generation_COMPLETE import Diffusion
from generate_new_imgs.UNet_model_generation import Residual_Attention_UNet_generation
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import os

noise_schedule = 'cosine'
input_channels = output_channels = 3
device = 'mps'
noise_steps = 1500
model_name = 'DDP_Residual_Attention_UNet_generation_Cifar10'
snapshot_path = os.path.join('..', 'models_run', model_name, 'weights', 'snapshot.pt')

image_size = 32

model = Residual_Attention_UNet_generation(input_channels, output_channels, 10, device).to(device)

diffusion = Diffusion(
        noise_schedule=noise_schedule, model=model,
        snapshot_path=snapshot_path,
        noise_steps=noise_steps, beta_start=1e-4, beta_end=0.02,
        device=device, image_size=image_size, model_name=model_name,
        multiple_gpus=False, ema_smoothing=False)


########### Sentinel Data Crops ############
# classes = {'Highway':0, 'River':1, 'HerbaceousVegetation':2,
#            'Residential':3, 'AnnualCrop':4, 'Pasture':5,
#            'Forest':6, 'PermanentCrop':7, 'Industrial':8,
#            'SeaLake':9}

############ CIFAR10 ############
classes = {'airplane':0, 'automobile':1, 'bird':2,
           'cat':3, 'deer':4, 'dog':5,
           'frog':6, 'horse':7, 'ship':8,
           'truck':9}

fig, axs = plt.subplots(2, 5, figsize=(15, 6))  # Adjust figsize as needed
axs = axs.ravel()
for i, class_ in enumerate(classes):
        prediction = diffusion.sample(n=1,model=model, target_class=torch.tensor([i], dtype=torch.int64).to(device), input_channels=input_channels, plot_gif_bool=False)
        prediction = prediction.clamp(0, 1)
        axs[i].imshow(prediction[0].permute(1,2,0).detach().cpu())
        axs[i].axis('off')
        axs[i].set_title(class_, fontsize=12)
plt.show()

# %%
