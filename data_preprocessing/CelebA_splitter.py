#%%
import os
import shutil
from tqdm import tqdm

celebA_path = '/Users/adrianoettari/Desktop/ASSEGNO_DI_RICERCA/pansharpening/celebA_aligned'
celbA_images = os.listdir(celebA_path)
train_folder_path = '/Users/adrianoettari/Desktop/ASSEGNO_DI_RICERCA/pansharpening/celebA_100k/train_original'
os.makedirs(train_folder_path, exist_ok=True)
for i in tqdm(range(100000)):
    file_name=celbA_images[i]
    source_path = os.path.join(celebA_path,file_name)
    shutil.move(source_path, train_folder_path)

val_folder_path = '/Users/adrianoettari/Desktop/ASSEGNO_DI_RICERCA/pansharpening/celebA_100k/val_original'
os.makedirs(val_folder_path, exist_ok=True)
for i in tqdm(range(100000, 120000)):
    file_name=celbA_images[i]
    source_path = os.path.join(celebA_path,file_name)
    shutil.move(source_path, val_folder_path)

test_folder_path = '/Users/adrianoettari/Desktop/ASSEGNO_DI_RICERCA/pansharpening/celebA_100k/test_original'
os.makedirs(test_folder_path, exist_ok=True)
for i in tqdm(range(120000, 121000)):
    file_name=celbA_images[i]
    source_path = os.path.join(celebA_path,file_name)
    shutil.move(source_path, test_folder_path)

#%%
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

train_folder_path = os.path.join('..','celebA_100k','train_original')

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
