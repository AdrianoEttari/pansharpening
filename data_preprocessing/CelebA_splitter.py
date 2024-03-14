#%%
import os
import shutil

celebA_path = '/Users/adrianoettari/Desktop/ASSEGNO_DI_RICERCA/pansharpening/CelebA'
destination_folder_path = '/Users/adrianoettari/Desktop/ASSEGNO_DI_RICERCA/pansharpening/CelebA_50k'
os.makedirs(destination_folder_path, exist_ok=True)
for i in range(50000):
    file_name=os.listdir(celebA_path)[i]
    source_path = os.path.join(celebA_path,file_name)
    shutil.copy(source_path, destination_folder_path)

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
