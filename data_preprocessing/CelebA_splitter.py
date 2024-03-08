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
