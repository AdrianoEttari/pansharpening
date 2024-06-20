
import torch
import numpy as np
import os
import shutil
import random
from tqdm import tqdm
from PIL import Image,ImageDraw,ImageFont,ImageFilter
from scipy.linalg import orth
import imageio
import cv2
 
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

def gif_maker(frames, frame_stride=1, destination_path='output.gif'):
    '''
    This function saves in a the desired path the frames in a gif format

    *Input:
        - frames: list of frames to be saved in a gif
        - frame_stride: int. The jump between frames that will be saved in the gif (e.g. if frame_stride=5 i=0 frame is considered, then i=5 frame is considered, etc.)
    *Output:
        - None
    '''
    images = []
    if frames[0].max()<100:
        frames = [torch.clamp(frame[0],0,1)*255 for frame in frames]
    
    for i,frame in enumerate(tqdm(frames)):
        if (i % frame_stride == 0) or (i == len(frames)-1):
            frame = frame.type(torch.uint8)
            np_frame = frame.permute(1, 2, 0).detach().cpu().numpy()
            image = Image.fromarray(np_frame)

            draw = ImageDraw.Draw(image)

            font = ImageFont.load_default()
            text = f'frame {i}'

            text_position = (10, 10)
            text_color = (255, 255, 255)  # white text
            
            outline_color = (0, 0, 0)  # black outline
            draw.text((text_position[0]-1, text_position[1]-1), text, font=font, fill=outline_color)
            draw.text((text_position[0]+1, text_position[1]-1), text, font=font, fill=outline_color)
            draw.text((text_position[0]-1, text_position[1]+1), text, font=font, fill=outline_color)
            draw.text((text_position[0]+1, text_position[1]+1), text, font=font, fill=outline_color)

            draw.text(text_position, text, font=font, fill=text_color)

            images.append(image)

    imageio.mimsave(destination_path, images, duration=0.0005)

def video_maker(frames, video_path='output.mp4', fps=50):
    '''
    Convert a sequence of frames to a video.

    *Input:
        - frames: list of frames to be saved in a video
        - video_path: path to save the video file
        - fps: frames per second
    *Output:
        - None
    '''
    if frames[0].max() < 100:
        frames = [torch.clamp(frame[0], 0, 1) * 255 for frame in frames]
    
    height, width = frames[0][0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    print('Creating video... with frames:', len(frames))
    for i,frame in enumerate(frames):
        frame = frame.type(torch.uint8)
        np_frame = frame.permute(1, 2, 0).detach().cpu().numpy()
        frame_bgr = cv2.cvtColor(np_frame, cv2.COLOR_RGB2BGR)

        # Create an Image object for drawing
        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)

        # Add text (frame title)
        font = ImageFont.load_default()

        text = f'Frame {i}'

        text_position = (10, 10)
        text_color = (255, 255, 255)  # white text
        outline_color = (0, 0, 0)  # black outline
        draw.text((text_position[0] - 1, text_position[1] - 1), text, font=font, fill=outline_color)
        draw.text((text_position[0] + 1, text_position[1] - 1), text, font=font, fill=outline_color)
        draw.text((text_position[0] - 1, text_position[1] + 1), text, font=font, fill=outline_color)
        draw.text((text_position[0] + 1, text_position[1] + 1), text, font=font, fill=outline_color)
        draw.text(text_position, text, font=font, fill=text_color)

        # Convert back to BGR for OpenCV
        frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Write frame to video
        video.write(frame_bgr)
    
    video.release()
    # to convert an mp4 into gif, run into terminal ffmpeg -i <input.mp4> -vf "fps=100,scale=320:-1:flags=lanczos" <output.gif>

    
if __name__=="__main__":
    main_folder = 'up42_sentinel2_patches'
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
