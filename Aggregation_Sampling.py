import numpy as np
import matplotlib.pyplot as plt
import torch
from numpy import pi, exp, sqrt

class split_aggregation_sampling:
    def __init__(self, img_lr, patch_size, stride, magnification_factor, diffusion_model, device):
        assert stride <= patch_size
        self.img_lr = img_lr
        self.patch_size = patch_size
        self.stride = stride
        self.magnification_factor = magnification_factor
        self.diffusion_model = diffusion_model

        self.device = device
        self.model = diffusion_model.model
        batch_size, channels, height, width = img_lr.shape

        self.patches_lr, self.patches_sr_infos = self.imgs_splitter(img_lr, patch_size, stride, magnification_factor)
        self.weight = self.gaussian_weights(patch_size*magnification_factor, patch_size*magnification_factor, batch_size)

    def imgs_splitter(self, img_to_split, patch_size, stride=None, magnification_factor=1):
        '''
        This function takes an image tensor and splits it into patches of size model_input_size x model_input_size.
        The stride is the number of pixels to skip between patches. If stride is not specified, it is set to model_input_size
        to avoid overlapping patches. 
        If you want to perform super resolution, the be > 1. The patches_lr list doesn't change,
        but the patches_sr_infos list will contain the coordinates of the patches in the high resolution image (otherwise,
        if magnification_factor=1 patches_sr_infos will containthe coordinates of the patches in the low resolution image).
        '''
        if stride is None:
            stride = patch_size  # Default non-overlapping behavior

        batch_size, channels, height, width = img_to_split.shape
        patches_lr = []
        patches_sr_infos = []
        for y in range(0, height + 1, stride):
            for x in range(0, width + 1, stride):
                if y+patch_size > height:
                    y_start = height - patch_size
                    y_end = height
                else:
                    y_start = y
                    y_end = y+patch_size
                if x+patch_size > width:
                    x_start = width - patch_size
                    x_end = width
                else:
                    x_start = x
                    x_end = x+patch_size
                if (y_start*magnification_factor, y_end*magnification_factor, x_start*magnification_factor, x_end*magnification_factor) not in patches_sr_infos:
                    patch = img_to_split[:, :,  y_start:y_end, x_start:x_end]
                    patches_lr.append(patch)
                    patches_sr_infos.append((y_start*magnification_factor, y_end*magnification_factor, x_start*magnification_factor, x_end*magnification_factor))

        return patches_lr, patches_sr_infos

    def plot_patches(self):
        '''
        ----------------- TO ADJUST -----------------
        '''
        n_patches_row = n_patches_col = int(np.sqrt(len(self.patches_lr)))
        fig, axs = plt.subplots(n_patches_row,n_patches_col,figsize=(10,10))
        for i in range(n_patches_row):
            for j in range(n_patches_col):
                pass
            pass
                
        plt.show()

    def aggregation_sampling(self):
        '''
        For each low resolution patch in self.patches_lr and info in self.patches_sr_infos, 
        '''
        img_lr = self.img_lr
        magnification_factor = self.magnification_factor

        batch_size, channels, height, width = img_lr.shape
        im_res = torch.zeros([batch_size, channels, height*magnification_factor, width*magnification_factor], dtype=img_lr.dtype, device=self.device)
        pixel_count = torch.zeros([batch_size, channels, height*magnification_factor, width*magnification_factor], dtype=img_lr.dtype, device=self.device)

        for i in range(len(self.patches_lr)):
            patch_sr = self.diffusion_model.sample(1, self.model, self.patches_lr[i].squeeze(0).to(self.device), input_channels=3, plot_gif_bool=False)
            im_res[:, :, self.patches_sr_infos[i][0]:self.patches_sr_infos[i][1], self.patches_sr_infos[i][2]:self.patches_sr_infos[i][3]] += patch_sr * self.weight
            pixel_count[:, :, self.patches_sr_infos[i][0]:self.patches_sr_infos[i][1], self.patches_sr_infos[i][2]:self.patches_sr_infos[i][3]] += self.weight
        
        assert torch.all(pixel_count != 0)
        im_res /= pixel_count

        return im_res

    def gaussian_weights(self, tile_width, tile_height, nbatches):
            """Generates a gaussian mask of weights for tile contributions"""
            latent_width = tile_width
            latent_height = tile_height

            var = 0.01
            midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
            x_probs = [exp(-(x-midpoint)*(x-midpoint)/(latent_width*latent_width)/(2*var)) / sqrt(2*pi*var) for x in range(latent_width)]
            midpoint = latent_height / 2
            y_probs = [exp(-(y-midpoint)*(y-midpoint)/(latent_height*latent_height)/(2*var)) / sqrt(2*pi*var) for y in range(latent_height)]

            weights = torch.tensor(np.outer(y_probs, x_probs)).to(torch.float32).to(self.device)
            return torch.tile(weights, (nbatches, 3, 1, 1))


if __name__ == '__main__':

    from PIL import Image
    from torchvision import transforms
    from torch.nn import functional as F
    import matplotlib.pyplot as plt
    from train_diffusion_superres import Diffusion
    from UNet_model_superres import Residual_Attention_UNet_superres
    import os

    magnification_factor = 8
    input_channels = output_channels = 3
    noise_schedule = 'cosine'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Residual_Attention_UNet_superres(input_channels, output_channels, device).to(device)
    snapshot_path = os.path.join('models_run','DDP_Residual_Attention_UNet_superres_magnification8_ANIME50k_DownBlur_512','weights','snapshot.pt')
    noise_steps = 1500
    model_input_size = 512
    model_name = 'DDP_Residual_Attention_UNet_superres_magnification8_ANIME50k_DownBlur_512'
    Degradation_type = 'DownBlur'

    img_hr = Image.open('anime_test.jpg')
    img_lr = img_hr.resize((128, 128), Image.BICUBIC)
    transform = transforms.Compose([ transforms.ToTensor()])
    img_lr = transform(img_lr).unsqueeze(0).to(device)
        
    pch_size = 64
    stride = 32

    diffusion = Diffusion(
        noise_schedule=noise_schedule, model=model,
        snapshot_path=snapshot_path,
        noise_steps=noise_steps, beta_start=1e-4, beta_end=0.02, 
        magnification_factor=magnification_factor,device=device,
        image_size=model_input_size, model_name=model_name, Degradation_type=Degradation_type)
    
    aggregation_sampling = split_aggregation_sampling(img_lr, pch_size, stride, magnification_factor, diffusion, device)
    final_pred = aggregation_sampling.aggregation_sampling()

    final_pred = transforms.ToPILImage()(final_pred.squeeze(0).cpu())
    final_pred.save('anime_test_sr_aggregation_sampling.png')