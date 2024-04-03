#%%
#%%
import os
import shutil
from tqdm import tqdm

anime_data_path = '/Users/adrianoettari/Desktop/ASSEGNO_DI_RICERCA/pansharpening/anime_data'
anime_imgs = os.listdir(anime_data_path)
train_folder_path = '/Users/adrianoettari/Desktop/ASSEGNO_DI_RICERCA/pansharpening/anime_data_50k/train_original'
os.makedirs(train_folder_path, exist_ok=True)
for i in tqdm(range(50000)):
    file_name=anime_imgs[i]
    source_path = os.path.join(anime_data_path,file_name)
    shutil.move(source_path, train_folder_path)

val_folder_path = '/Users/adrianoettari/Desktop/ASSEGNO_DI_RICERCA/pansharpening/anime_data_50k/val_original'
os.makedirs(val_folder_path, exist_ok=True)
for i in tqdm(range(50000, 55000)):
    file_name=anime_imgs[i]
    source_path = os.path.join(anime_data_path,file_name)
    shutil.move(source_path, val_folder_path)

test_folder_path = '/Users/adrianoettari/Desktop/ASSEGNO_DI_RICERCA/pansharpening/anime_data_50k/test_original'
os.makedirs(test_folder_path, exist_ok=True)
for i in tqdm(range(55000, 56000)):
    file_name=anime_imgs[i]
    source_path = os.path.join(anime_data_path,file_name)
    shutil.move(source_path, test_folder_path)

#%%
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

width = []
height = []
channels = []

for file in os.listdir(train_folder_path):
    img = Image.open(os.path.join(train_folder_path,file))
    img_array = np.array(img)
    width.append(img_array.shape[0])
    height.append(img_array.shape[1])
    channels.append(img_array.shape[2])

plt.hist(width, bins=100)
plt.show()
plt.hist(height, bins=100)
plt.show()
plt.hist(channels, bins=100)
plt.show()


# %%
# from PIL import Image
# import os
# import numpy as np
# from tqdm import tqdm

# test_path = '/Users/adrianoettari/Desktop/ASSEGNO_DI_RICERCA/pansharpening/Humans_test/test_original'
# counter = 0
# for file in tqdm(os.listdir(test_path)):
#     img = Image.open(os.path.join(test_path,file))
#     img = np.array(img)
    
#     try:
#         if img.shape[2] != 3:
#             os.remove(os.path.join(test_path,file))
#             counter+=1
#     except:
#         os.remove(os.path.join(test_path,file))
#         counter+=1
# print(counter)  
# %%

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
