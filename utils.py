from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import os
from torchvision import transforms


class get_data_superres(Dataset):
    '''
    This class allows to store the data in a Dataset that can be used in a DataLoader
    like that train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True).

    -Input:
        root_dir: path to the folder where the data is stored. 
        magnification_factor: factor by which the original images are downsampled.
        data_format: 'PIL' or 'numpy' or 'torch'. The format of the images in the dataset.
        transform: a torchvision.transforms.Compose object with the transformations that will be applied to the images.
    -Output:
        A Dataset object that can be used in a DataLoader.

    __getitem__ returns x and y. The split in batches must be done in the DataLoader (not here).
    '''
    def __init__(self, root_dir, magnification_factor, data_format='PIL', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.magnification_factor = magnification_factor    
        self.original_imgs_dir = os.path.join(self.root_dir)
        self.data_format = data_format
        self.y_filenames = sorted(os.listdir(self.original_imgs_dir))

    def __len__(self):
        return len(self.y_filenames)

    def __getitem__(self, idx):
        y_path = os.path.join(self.original_imgs_dir, self.y_filenames[idx])

        if self.data_format == 'PIL':
            y = Image.open(y_path)
        elif self.data_format == 'numpy':
            y = np.load(y_path)
            y = Image.fromarray((y*255).astype(np.uint8))
        elif self.data_format == 'torch':
            to_pil = transforms.ToPILImage()
            y = torch.load(y_path)
            y = to_pil(y)

        # Downsample the original image
        downsample = transforms.Resize((y.size[0] // self.magnification_factor, y.size[1] // self.magnification_factor))
        x = downsample(y)

        if self.transform:
            x = self.transform(x)
            y = self.transform(y)

        return x, y