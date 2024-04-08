#%%
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
from tqdm import tqdm

folder_path = '/Users/adrianoettari/Desktop/ASSEGNO_DI_RICERCA/pansharpening/DIV2K_valid_LR_bicubic/X2'
os.makedirs('DIV2k_split', exist_ok=True)

for img_relative_path in tqdm(os.listdir(folder_path)):
    img_path = os.path.join(folder_path, img_relative_path)
    img = Image.open(img_path)
    img = np.array(img)
    width = img.shape[1]
    height = img.shape[0]
    if width//2 < 350:
        img_split_1 = img[:512, :512]
        img_split_1 = Image.fromarray(img_split_1)
        img_split_1.save(f'DIV2k_split/{img_relative_path[:-4]}_1.png')
    else:
        img_split_1 = img[:512, :512]
        img_split_2 = img[:512, 512:]
        img_split_1 = Image.fromarray(img_split_1)
        img_split_2 = Image.fromarray(img_split_2)
        img_split_1.save(f'DIV2k_split/{img_relative_path[:-4]}_1.png')
        img_split_2.save(f'DIV2k_split/{img_relative_path[:-4]}_2.png')

# %%
widths = []
heights = []
new_folder_path = 'DIV2k_split'

for img_relative_path in os.listdir(new_folder_path):
    img_path = os.path.join(new_folder_path, img_relative_path)
    img = Image.open(img_path)
    width, height = img.size
    widths.append(width)
    heights.append(height)

plt.hist(widths, bins=50)
plt.show()
plt.hist(heights, bins=50)
plt.show()