import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import ToTensor, ToPILImage

def load_from_pickle(file_path):
    '''
    load_from_pickle function loads the data from a pickle file.
    It takes the file path as an argument and returns the loaded data.
    '''
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    
def save_to_pickle(file_path, data):
    '''
    save_to_pickle function saves the data to a pickle file.
    It takes the file path and the data as arguments and does not return anything.
    '''
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
        

def imgs_splitter(img_to_split, num_patches_per_row_and_col, model_input_size, overlapping=False):
    '''
    This function takes an image tensor and splits it into patches of size model_input_size x model_input_size.
    '''

    number_of_patches = num_patches_per_row_and_col

    if overlapping:
        pass
    else:
        width_steps = height_steps = np.cumsum(np.repeat(model_input_size,number_of_patches))
        width_steps = np.insert(width_steps,0,0)
        height_steps = np.insert(height_steps,0,0)

    position_patch_dic = {}
    for i in range(number_of_patches):
        for j in range(number_of_patches):
            position_patch_dic[i,j]=img_to_split[:,width_steps[i]:width_steps[i+1],height_steps[j]:height_steps[j+1]]

    return position_patch_dic

def plot_patches(position_patch_dic):
    n_patches_row = n_patches_col = int(np.sqrt(len(position_patch_dic)))
    fig, axs = plt.subplots(n_patches_row,n_patches_col,figsize=(10,10))
    for i in range(n_patches_row):
        for j in range(n_patches_col):
            if len(position_patch_dic[i,j].shape)==4:
                axs[i,j].imshow(position_patch_dic[i,j][0].permute(1,2,0))
                axs[i,j].axis('off')
            else:
                axs[i,j].imshow(position_patch_dic[i,j].permute(1,2,0))
                axs[i,j].axis('off')
    plt.show()

def merge_images(image_dict):
    '''
    This function merges a dictionary of images into a single image based on their positions in a grid.
    The image_dict dictionary should have tuples as keys representing the position of the image in the grid (as tuples),
    and the values should be the tensor images themselves.
    '''
    for key, value in image_dict.items():
        if len(value.shape) == 4:
            image_dict[key] = value.squeeze(0)
    # Calculate the size of the merged image
    grid_size = (int(np.sqrt(len(image_dict))),int(np.sqrt(len(image_dict))))
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


class ImageSpliterTh:
    def __init__(self, im, pch_size, stride, sf=1):
        '''
        Input:
            im: n x c x h x w, torch tensor, float, low-resolution image in SR
            pch_size, stride: patch setting
            sf: scale factor in image super-resolution
        '''
        assert stride <= pch_size
        self.stride = stride
        self.pch_size = pch_size
        self.sf = sf

        bs, chn, height, width= im.shape
        self.height_starts_list = self.extract_starts(height)
        self.width_starts_list = self.extract_starts(width)
        self.length = self.__len__()
        self.num_pchs = 0

        self.im_ori = im
        self.im_res = torch.zeros([bs, chn, height*sf, width*sf], dtype=im.dtype, device=im.device)
        self.pixel_count = torch.zeros([bs, chn, height*sf, width*sf], dtype=im.dtype, device=im.device)
        self.weight = self._gaussian_weights(pch_size, pch_size, bs, im.device)

    def extract_starts(self, length):
        if length <= self.pch_size:
            starts = [0,]
        else:
            starts = list(range(0, length, self.stride))
            for i in range(len(starts)):
                if starts[i] + self.pch_size > length:
                    starts[i] = length - self.pch_size
            starts = sorted(set(starts), key=starts.index)
        return starts

    def __len__(self):
        return len(self.height_starts_list) * len(self.width_starts_list)

    def __iter__(self):
        return self

    def __next__(self):
        if self.num_pchs < self.length:
            w_start_idx = self.num_pchs // len(self.height_starts_list)
            w_start = self.width_starts_list[w_start_idx]
            w_end = w_start + self.pch_size

            h_start_idx = self.num_pchs % len(self.height_starts_list)
            h_start = self.height_starts_list[h_start_idx]
            h_end = h_start + self.pch_size

            pch = self.im_ori[:, :, h_start:h_end, w_start:w_end,]

            h_start *= self.sf
            h_end *= self.sf
            w_start *= self.sf
            w_end *= self.sf

            self.w_start, self.w_end = w_start, w_end
            self.h_start, self.h_end = h_start, h_end

            self.num_pchs += 1
        else:
            raise StopIteration()

        return pch, (h_start, h_end, w_start, w_end)

    def _gaussian_weights(self, tile_width, tile_height, nbatches, device):
        """Generates a gaussian mask of weights for tile contributions"""
        from numpy import pi, exp, sqrt
        import numpy as np

        latent_width = tile_width
        latent_height = tile_height

        var = 0.01
        midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        x_probs = [exp(-(x-midpoint)*(x-midpoint)/(latent_width*latent_width)/(2*var)) / sqrt(2*pi*var) for x in range(latent_width)]
        midpoint = latent_height / 2
        y_probs = [exp(-(y-midpoint)*(y-midpoint)/(latent_height*latent_height)/(2*var)) / sqrt(2*pi*var) for y in range(latent_height)]

        weights = np.outer(y_probs, x_probs)
        return torch.tile(torch.tensor(weights, device=device), (nbatches, 3, 1, 1))

    def update(self, pch_res, index_infos):
        '''
        Input:
            pch_res: n x c x pch_size x pch_size, float
            index_infos: (h_start, h_end, w_start, w_end)
        '''
        if index_infos is None:
            w_start, w_end = self.w_start, self.w_end
            h_start, h_end = self.h_start, self.h_end
        else:
            h_start, h_end, w_start, w_end = index_infos

        self.im_res[:, :, h_start:h_end, w_start:w_end] += pch_res
        self.pixel_count[:, :, h_start:h_end, w_start:w_end] += 1

    def update_gaussian(self, pch_res, index_infos):
        '''
        Input:
            pch_res: n x c x pch_size x pch_size, float
            index_infos: (h_start, h_end, w_start, w_end)
        '''
        if index_infos is None:
            w_start, w_end = self.w_start, self.w_end
            h_start, h_end = self.h_start, self.h_end
        else:
            h_start, h_end, w_start, w_end = index_infos

        self.im_res[:, :, h_start:h_end, w_start:w_end] += pch_res * self.weight
        self.pixel_count[:, :, h_start:h_end, w_start:w_end] += self.weight

    def gather(self):
        assert torch.all(self.pixel_count != 0)
        return self.im_res.div(self.pixel_count)

### ----------------------------------------------------------------------------------- ###
# FUNCTIONS FROM StableSR/scripts/wavelet_color_fix.py https://github.com/IceClear/StableSR/blob/main/scripts/wavelet_color_fix.py
def adain_color_fix(target: Image, source: Image):
    # Convert images to tensors
    to_tensor = ToTensor()
    target_tensor = to_tensor(target).unsqueeze(0)
    source_tensor = to_tensor(source).unsqueeze(0)

    # Apply adaptive instance normalization
    result_tensor = adaptive_instance_normalization(target_tensor, source_tensor)

    # Convert tensor back to image
    to_image = ToPILImage()
    result_image = to_image(result_tensor.squeeze(0).clamp_(0.0, 1.0))

    return result_image

def wavelet_color_fix(target: Image, source: Image):
    # Convert images to tensors
    to_tensor = ToTensor()
    target_tensor = to_tensor(target).unsqueeze(0)
    source_tensor = to_tensor(source).unsqueeze(0)

    # Apply wavelet reconstruction
    result_tensor = wavelet_reconstruction(target_tensor, source_tensor)

    # Convert tensor back to image
    to_image = ToPILImage()
    result_image = to_image(result_tensor.squeeze(0).clamp_(0.0, 1.0))

    return result_image

def calc_mean_std(feat, eps=1e-5):
    """Calculate mean and std for adaptive_instance_normalization.
    Args:
        feat (Tensor): 4D tensor.
        eps (float): A small value added to the variance to avoid
            divide-by-zero. Default: 1e-5.
    """
    size = feat.size()
    assert len(size) == 4, 'The input feature should be 4D tensor.'
    b, c = size[:2]
    feat_var = feat.reshape(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().reshape(b, c, 1, 1)
    feat_mean = feat.reshape(b, c, -1).mean(dim=2).reshape(b, c, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat):
    """Adaptive instance normalization.
    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.
    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def wavelet_blur(image, radius: int):
    """
    Apply wavelet blur to the input tensor.
    """
    # input shape: (1, 3, H, W)
    # convolution kernel
    kernel_vals = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device)
    # add channel dimensions to the kernel to make it a 4D tensor
    kernel = kernel[None, None]
    # repeat the kernel across all input channels
    kernel = kernel.repeat(3, 1, 1, 1)
    image = F.pad(image, (radius, radius, radius, radius), mode='replicate')
    # apply convolution
    output = F.conv2d(image, kernel, groups=3, dilation=radius)
    return output

def wavelet_decomposition(image, levels=5):
    """
    Apply wavelet decomposition to the input tensor.
    This function only returns the low frequency & the high frequency.
    """
    high_freq = torch.zeros_like(image)
    for i in range(levels):
        radius = 2 ** i
        low_freq = wavelet_blur(image, radius)
        high_freq += (image - low_freq)
        image = low_freq

    return high_freq, low_freq

def wavelet_reconstruction(content_feat, style_feat):
    """
    Apply wavelet decomposition, so that the content will have the same color as the style.
    """
    # calculate the wavelet decomposition of the content feature
    content_high_freq, content_low_freq = wavelet_decomposition(content_feat)
    del content_low_freq
    # calculate the wavelet decomposition of the style feature
    style_high_freq, style_low_freq = wavelet_decomposition(style_feat)
    del style_high_freq
    # reconstruct the content feature with the style's high frequency
    return content_high_freq + style_low_freq
### ----------------------------------------------------------------------------------- ###

if __name__ == '__main__':
    # from PIL import Image
    # from torchvision import transforms
    # from torch.nn import functional as F
    # image_size = 256

    # img_test = Image.open('anime_test.jpg')
    # transform = transforms.Compose([ transforms.ToTensor()])
    # img_test = transform(img_test)
    # img_test = img_test[:,:,list(np.arange(50,562))]
    # img_test = F.interpolate(img_test.unsqueeze(0), size=(image_size, image_size), mode='bilinear', align_corners=False).squeeze(0)

    # position_patch_dic = imgs_splitter(img_test, 4, 64)
    # print(position_patch_dic)
    # plot_patches(position_patch_dic)
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

    position_patch_dic = imgs_splitter(img_test, number_of_patches, model_input_size)
    os.makedirs('aggregation_sampling',exist_ok=True)
    save_to_pickle(os.path.join('aggregation_sampling','inputs.pkl'), position_patch_dic)

    position_patch_dic = load_from_pickle(os.path.join('aggregation_sampling','inputs.pkl'))

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

    diffusion = Diffusion(
            noise_schedule=noise_schedule, model=model,
            snapshot_path=snapshot_path,
            noise_steps=noise_steps, beta_start=1e-4, beta_end=0.02, 
            magnification_factor=magnification_factor,device=device,
            image_size=512, model_name=model_name, Degradation_type=Degradation_type)
    
    position_super_lr_patches_dic = {}

    for key,value in tqdm(position_patch_dic.items()):
        super_lr_patch = diffusion.sample(1, model, value.to(device), input_channels=3, plot_gif_bool=False)
        position_super_lr_patches_dic[key] = super_lr_patch.to('cpu')

    save_to_pickle(os.path.join('aggregation_sampling','predictions.pkl'), position_super_lr_patches_dic)

    #%%
    # position_super_lr_patches_dic = load_from_pickle(os.path.join('aggregation_sampling','predictions.pkl'))

    # plot_patches(position_super_lr_patches_dic)
    # %%
    # merged_image = merge_images(position_super_lr_patches_dic)
    # plt.imshow(merged_image.permute(1,2,0))
    # plt.axis('off')
    # plt.show()





        