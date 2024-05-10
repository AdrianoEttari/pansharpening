import numpy as np
import matplotlib.pyplot as plt
import torch
from numpy import pi, exp, sqrt
from tqdm import tqdm 

class split_aggregation_sampling:
    def __init__(self, img_lr, patch_size, stride, magnification_factor, diffusion_model, device):
        '''
        This class is used to perform a split into patches an image (with the patchifier function)
        and also to aggregate the super resolution of the generated patches (with the aggregation_sampling function).
        '''
        assert stride <= patch_size
        self.img_lr = img_lr
        self.patch_size = patch_size
        self.stride = stride
        self.magnification_factor = magnification_factor
        self.diffusion_model = diffusion_model

        self.device = device
        self.model = diffusion_model.model
        batch_size, channels, height, width = img_lr.shape

        self.patches_lr, self.patches_sr_infos = self.patchifier(img_lr, patch_size, stride, magnification_factor)
        self.weight = self.gaussian_weights(patch_size*magnification_factor, patch_size*magnification_factor, batch_size)

    def patchifier(self, img_to_split, patch_size, stride=None, magnification_factor=1):
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

        for i in tqdm(range(len(self.patches_lr))):
            patch_sr = self.diffusion_model.sample(1, self.model, self.patches_lr[i].squeeze(0).to(self.device), input_channels=3, plot_gif_bool=False)
            im_res[:, :, self.patches_sr_infos[i][0]:self.patches_sr_infos[i][1], self.patches_sr_infos[i][2]:self.patches_sr_infos[i][3]] += patch_sr * self.weight
            pixel_count[:, :, self.patches_sr_infos[i][0]:self.patches_sr_infos[i][1], self.patches_sr_infos[i][2]:self.patches_sr_infos[i][3]] += self.weight
        
        assert torch.all(pixel_count != 0)
        im_res /= pixel_count
        im_res = torch.clamp(im_res, 0, 1)
        
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

def launch(args):
    from PIL import Image
    from torchvision import transforms
    from torch.nn import functional as F
    import matplotlib.pyplot as plt
    from train_diffusion_superres import Diffusion
    from UNet_model_superres import Residual_Attention_UNet_superres
    import os  

    snapshot_folder_path = args.snapshot_folder_path
    snapshot_name = args.snapshot_name
    magnification_factor = args.magnification_factor
    input_channels = output_channels = args.inp_out_channels
    noise_schedule = args.noise_schedule
    device = args.device
    model_input_size = args.model_input_size
    noise_steps = args.noise_steps
    model_name = args.model_name
    Degradation_type = args.Degradation_type
    patch_size = args.patch_size
    stride = args.stride
    destination_path = args.destination_path
    img_lr_path = args.img_lr_path
    Unet_type = args.UNet_type

    snapshot_path = os.path.join(snapshot_folder_path, snapshot_name)

    if Unet_type.lower() == 'residual attention unet':
        model = Residual_Attention_UNet_superres(input_channels, output_channels, device).to(device)

    print(f'You are using {Unet_type} model')

    img_lr = Image.open(img_lr_path)
    try:
        assert img_lr.size[0] == img_lr.size[1] # Image must be square
    except:
        distances = []
        possible_sizes = [64,128,256,512,1024]
        for size in possible_sizes:
            distance = 0
            distance += abs(size - img_lr.size[0])
            distance += abs(size - img_lr.size[1])
            distances.append(distance)
        new_width = new_height = possible_sizes[np.argmin(distances)]

        print(f'The image must be square but it is {img_lr.size[0],img_lr.size[1]}! It will be resized to {new_width}x{new_height}')
        img_lr = img_lr.resize((new_width, new_height), Image.BICUBIC)

    transform = transforms.Compose([transforms.ToTensor()])
    img_lr = transform(img_lr).unsqueeze(0).to(device)
        
    diffusion = Diffusion(
        noise_schedule=noise_schedule, model=model,
        snapshot_path=snapshot_path,
        noise_steps=noise_steps, beta_start=1e-4, beta_end=0.02, 
        magnification_factor=magnification_factor,device=device,
        image_size=model_input_size, model_name=model_name, Degradation_type=Degradation_type)
    
    aggregation_sampling = split_aggregation_sampling(img_lr, patch_size, stride, magnification_factor, diffusion, device)
    final_pred = aggregation_sampling.aggregation_sampling()

    final_pred = transforms.ToPILImage()(final_pred.squeeze(0).cpu())
    final_pred.save(destination_path)

if __name__ == '__main__':
    import argparse
    import os  
    parser = argparse.ArgumentParser(description=' ')
    parser.add_argument('--noise_schedule', type=str, default='cosine')
    parser.add_argument('--snapshot_name', type=str, default='snapshot.pt')
    parser.add_argument('--noise_steps', type=int, default=1000)
    parser.add_argument('--model_input_size', type=int, default=512)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--UNet_type', type=str)
    parser.add_argument('--Degradation_type', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--magnification_factor', type=int)
    parser.add_argument('--inp_out_channels', type=int, default=3)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--stride', type=int, default=32)
    parser.add_argument('--destination_path', type=str)
    parser.add_argument('--img_lr_path', type=str)
    args = parser.parse_args()
    args.snapshot_folder_path = os.path.join(os.curdir, 'models_run', args.model_name, 'weights')
    launch(args)


