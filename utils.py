from torch.utils.data import Dataset
import torch
from PIL import Image, ImageFilter
import numpy as np
import os
from torchvision import transforms
import shutil
import random
from tqdm import tqdm
from PIL import Image
from degradation_from_BSRGAN import degradation_bsrgan_plus, single2uint, imread_uint

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
    def __init__(self, root_dir, magnification_factor, blur_radius=0.5, data_format='PIL', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.magnification_factor = magnification_factor    
        self.original_imgs_dir = os.path.join(self.root_dir)
        self.blur_radius = blur_radius
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

        if self.blur_radius > 0:
            x = x.filter(ImageFilter.GaussianBlur(self.blur_radius))
    
        to_tensor = transforms.ToTensor()
        x = to_tensor(x)
        y = to_tensor(y)

        return x, y
        
class get_data_superres_BSRGAN(Dataset):
    '''
    This class allows to store the data in a Dataset that can be used in a DataLoader
    like that train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True).

    -Input:
        root_dir: path to the folder where the data is stored. 
        magnification_factor: factor by which the original images are downsampled.
        model_input_size: size of the input images to the model.
        num_crops: number of crops to be generated from each image.
        destination_folder: path to the folder where the lr and hr images will be saved.
    -Output:
        A Dataset object that can be used in a DataLoader.

    __getitem__ returns x and y. The split in batches must be done in the DataLoader (not here).
    '''
    def __init__(self, root_dir, magnification_factor, model_input_size, num_crops, destination_folder=None):
        self.root_dir = root_dir
        self.magnification_factor = magnification_factor
        self.model_input_size = model_input_size
        self.original_imgs_dir = os.path.join(self.root_dir)
        self.y_filenames = sorted(os.listdir(self.original_imgs_dir))
        self.num_crops = num_crops
        self.x_images, self.y_images = self.BSR_degradation()
        if destination_folder is not None:
            self.dataset_saver(destination_folder)

    def BSR_degradation(self):
        '''
        This function takes as input the path of the original images, the magnification factor, the model input size
        and also the the number of crops to be generated from each image. It returns two lists with the lr and hr images.
        '''
        x_images = []
        y_images = []
        for i in tqdm(range(len(self.y_filenames))):
            y_path = os.path.join(self.original_imgs_dir, self.y_filenames[i])
            for _ in range(self.num_crops):
                y = imread_uint(y_path, 3)
                x, y = degradation_bsrgan_plus(y, sf=self.magnification_factor, lq_patchsize=self.model_input_size)
                x = single2uint(x)
                y = single2uint(y)
                to_tensor = transforms.ToTensor()
                x = to_tensor(x)
                y = to_tensor(y)
                x_images.append(x)
                y_images.append(y)
        
        # Shuffle the lists
        combined = list(zip(x_images, y_images))
        random.shuffle(combined)
        x_images[:], y_images[:] = zip(*combined)

        return x_images, y_images

    def dataset_saver(self, destination_folder):
        '''
        This function saves the lr and hr images in the destination_folder with the following paths: 
        <destination_folder>/lr/x_<i>.png and <destination_folder>/hr/y_<i>.png.
        '''
        os.makedirs(destination_folder, exist_ok=True)
        os.makedirs(os.path.join(destination_folder, 'lr'), exist_ok=True)
        os.makedirs(os.path.join(destination_folder, 'hr'), exist_ok=True)
        for i in range(len(self.x_images)):
            x = self.x_images[i]
            y = self.y_images[i]
            x_path = os.path.join(destination_folder, 'lr',  f"x_{i}.png")
            y_path = os.path.join(destination_folder, 'hr',  f"y_{i}.png")
            x = x.permute(1, 2, 0).clamp(0, 1).numpy()
            y = y.permute(1, 2, 0).clamp(0, 1).numpy()
            x = Image.fromarray((x * 255).astype(np.uint8))
            y = Image.fromarray((y * 255).astype(np.uint8))
            x.save(x_path)
            y.save(y_path)

    def __len__(self):
        return len(self.x_images)

    def __getitem__(self, idx):
        x = self.x_images[idx]
        y = self.y_images[idx]

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

def img_splitter(source_folder, destination_folder, desired_width, threshold_rate=0.2):
    '''
    This function takes the images into the source_folder and checks if they match the desired_width (=desired_height, considering
    that the images are square). If they are smaller than the desired_width-threshold, they are not split or resized and
    we just discard them; by contradiction, if they are larger than the desired_width (desired_height) but smaller than the desired_width+threshold
    (desired_height+threshold) we just crop them to the desired_width(desired_height); finally, if they are larger than the desired_width+threshold,
    we split them into overlapping patches of size desired_width x desired_width.
    '''
    os.makedirs(destination_folder, exist_ok=True)
    for img_relative_path in tqdm(os.listdir(source_folder)):
        counter = len(os.listdir(destination_folder))
        img_path = os.path.join(source_folder, img_relative_path)
        img = Image.open(img_path)
        img = np.array(img)
        width = img.shape[1]
        height = img.shape[0]

        threshold = desired_width * threshold_rate

        if (width < desired_width - threshold) or (height < desired_width - threshold):
            print(f"Image {img_relative_path} is too small to be split or resized.")
        elif (width > desired_width) and (width < desired_width + threshold) and (height > desired_width) and (height < desired_width + threshold):
            # No need to resize, just save
            save_path = os.path.join(destination_folder, f"cropped_{counter}.png")
            Image.fromarray(img).save(save_path)
            counter += 1
        else:
            # Perform cropping
            for i in range(0, width - desired_width, desired_width // 2):  # Overlapping by half width
                for j in range(0, height - desired_width, desired_width // 2):  # Overlapping by half height
                    cropped_img = img[j:j + desired_width, i:i + desired_width]
                    save_path = os.path.join(destination_folder, f"cropped_{counter}.png")
                    Image.fromarray(cropped_img).save(save_path)
                    counter += 1

if __name__=="__main__":
    main_folder = 'DIV2K'
    data_organizer = data_organizer(main_folder)
    data_organizer.split_files(split_ratio=(0.85,0.1,0.05))
    for root, dirs, files in os.walk(main_folder):
        for file in files:
            if file == '.DS_Store':
                os.remove(os.path.join(root, file))

    # source_folder = 'satellite_imgs_test'
    # destination_folder = 'satellite_imgs_test_cropped'
    # desired_width = 128
    # img_splitter(source_folder, destination_folder, desired_width)
