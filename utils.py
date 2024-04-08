from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import os
from torchvision import transforms
import shutil
import random
from tqdm import tqdm

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

        if self.transform:
            y = self.transform(y)

        # Downsample the original image
        
        downsample = transforms.Resize((y.size[0] // self.magnification_factor, y.size[1] // self.magnification_factor),
                                       interpolation=transforms.InterpolationMode.BICUBIC)
        try:
            x = downsample(y)
        except:
            x = downsample(y.to('cpu')).to(y.device)

        to_tensor = transforms.ToTensor()
        x = to_tensor(x)
        y = to_tensor(y)

        return x, y
    
class data_organizer():
    '''
    This class allows to organize the data inside main_folder (provided in the __init__) 
    into train_original, val_original and test_original folders that will be created inside
    main_folder.
    ATTENTION: it is tailored for the super resolution problem.
    '''
    def __init__(self, main_folder):
        self.main_folder = main_folder
        self.train_folder = os.path.join(main_folder, 'train_original')
        self.val_folder = os.path.join(main_folder, 'val_original')
        self.test_folder = os.path.join(main_folder, 'test_original')
        os.makedirs(self.train_folder, exist_ok=True)
        os.makedirs(self.val_folder, exist_ok=True)
        os.makedirs(self.test_folder, exist_ok=True)
            
    def get_all_files_in_folder_and_subfolders(self, folder):
        '''
        Input:
            folder: path to the folder where the files and subfolders are stored.

        Output:
            all_files: list with the full path of all the files in the folder and its subfolders.
        '''
        all_files = []
        for root, dirs, files in os.walk(folder):
            for file in files:
                file_path = os.path.join(root, file)
                all_files.append(file_path)
        return all_files

    def split_files(self, split_ratio=(0.8, 0.15, 0.05)):
        '''
        This function splits the files in the main folder into train, val and test folders.

        Input:
            split_ratio: tuple with the ratio of files that will be assigned to the train, val and test folders.
        Output:
            None
        '''
        all_files = self.get_all_files_in_folder_and_subfolders(self.main_folder)
        # Get a list of all files in the input folder
        random.shuffle(all_files)  # Shuffle the files randomly

        # Calculate the number of files for each split
        total_files = len(all_files)
        train_size = int(total_files * split_ratio[0])
        val_size = int(total_files * split_ratio[1])

        # Assign files to the respective splits
        train_files = all_files[:train_size]
        val_files = all_files[train_size:train_size + val_size]
        test_files = all_files[train_size + val_size:]

        # Move files to the output folders
        self.move_files(train_files, self.train_folder)
        self.move_files(val_files, self.val_folder)
        self.move_files(test_files, self.test_folder)

    def move_files(self, files_full_path, destination_folder):
        '''
        This function moves the files to the destination folder.

        Input:
            files_full_path: list with the full path of the files that will be moved.
            destination_folder: path to the folder where the files will be moved.

        Output:
            None
        '''
        for file_full_path in tqdm(files_full_path,desc='Moving files'):
            destination_path = os.path.join(destination_folder, os.path.basename(file_full_path))
            shutil.move(file_full_path, destination_path)

def convert_png_to_jpg(png_file, jpg_file):
    try:
        # Open the PNG image
        with Image.open(png_file) as img:
            # Convert RGBA images to RGB
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            # Save as JPG
            img.save(jpg_file, 'JPEG')
        print("Conversion successful!")
    except Exception as e:
        print("Conversion failed:", e)

if __name__=="__main__":
    main_folder = 'DIV2k_split'
    data_organizer = data_organizer(main_folder)
    data_organizer.split_files(split_ratio=(0.85,0.1,0.05))
    for root, dirs, files in os.walk(main_folder):
        for file in files:
            if file == '.DS_Store':
                os.remove(os.path.join(root, file))
