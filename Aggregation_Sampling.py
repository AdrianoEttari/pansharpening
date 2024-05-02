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
    from PIL import Image
    from torchvision import transforms
    from torch.nn import functional as F
    image_size = 256

    img_test = Image.open('anime_test.jpg')
    transform = transforms.Compose([ transforms.ToTensor()])
    img_test = transform(img_test)
    img_test = img_test[:,:,list(np.arange(50,562))]
    img_test = F.interpolate(img_test.unsqueeze(0), size=(image_size, image_size), mode='bilinear', align_corners=False).squeeze(0)

    position_patch_dic = imgs_splitter(img_test, 4, 64)
    print(position_patch_dic)
    plot_patches(position_patch_dic)



        