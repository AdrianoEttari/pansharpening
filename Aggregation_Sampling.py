import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch

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
            axs[i,j].imshow(position_patch_dic[i,j].permute(1,2,0))
            axs[i,j].axis('off')
    plt.show()

def merge_images(image_dict):
    '''
    This function merges a dictionary of images into a single image based on their positions in a grid.
    The image_dict dictionary should have tuples as keys representing the position of the image in the grid (as tuples),
    and the values should be the tensor images themselves.
    '''
    grid_size = image_dict.keys()[len(image_dict)-1]
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
class Aggregation_Sampling():
    pass


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





        