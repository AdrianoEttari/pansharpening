# %%
from PIL import Image
import numpy as np
from tqdm import tqdm
import os

root = os.path.join('..', 'anime_data')
sets = ['train_original', 'val_original', 'test_original']

for set in sets:
    set_path = os.path.join(root, set)
    for file in tqdm(os.listdir(set_path)):
        file_path = os.path.join(set_path, file)
        img = Image.open(file_path)
        img = np.array(img)
        if len(img.shape) != 3:
            os.remove(file_path)
        elif img.shape[2] == 4:
            os.remove(file_path)
            img = img[:,:,:3]
            img = Image.fromarray(img)
            img.save(file_path)
        elif img.shape[2] != 3:
            os.remove(file_path)
