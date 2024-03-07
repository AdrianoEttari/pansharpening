# %% All the images must have the same number of channels
import os
from PIL import Image
import numpy as np

animal_classes = os.listdir('animal_imgs')
for animal in animal_classes:
    for img_path in os.listdir(os.path.join('animal_imgs', animal)):
        img = np.array(Image.open(os.path.join('animal_imgs', animal, img_path)))
        try:
            if img.shape[2] != 3:
                os.remove(os.path.join('animal_imgs', animal, img_path))
                new_img = img[:, :, :3]
                Image.fromarray(new_img).save(os.path.join('animal_imgs', animal, img_path))
        except:
            os.remove(os.path.join('animal_imgs', animal, img_path))
#%% Check the width and height distributions of the images
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

animal_classes = os.listdir('animal_imgs')
widths = []
heights = []
for animal in animal_classes:
    for img_path in os.listdir(os.path.join('animal_imgs', animal)):
        img = Image.open(os.path.join('animal_imgs', animal, img_path))
        widths.append(img.size[0])
        heights.append(img.size[1])

plt.hist(widths, bins=50)
plt.show()
plt.hist(heights, bins=50)
plt.show()

# %% REMOVE ALL THE IMAGES WITH HEIGHT OR WIDTH HIGHER THAN 1000

animal_classes = os.listdir('animal_imgs')
counter_removed = 0
for animal in animal_classes:
    for img_path in os.listdir(os.path.join('animal_imgs', animal)):
        img = Image.open(os.path.join('animal_imgs', animal, img_path))
        if img.size[0] > 1000 or img.size[1] > 1000:
            os.remove(os.path.join('animal_imgs', animal, img_path))
            counter_removed += 1
print(counter_removed)
# %% Check the number of images per class
total_imgs = 0
for animal in animal_classes:
    print(animal, len(os.listdir(os.path.join('animal_imgs', animal))))
    total_imgs += len(os.listdir(os.path.join('animal_imgs', animal)))
print(total_imgs)
# %% Balance the classes

#%% Organize the data in training, validation and test set
import os
import shutil

train_size = 0.8
val_size = 0.15
test_size = 0.05

animal_classes = os.listdir('animal_imgs')
for animal in animal_classes:
    os.makedirs(os.path.join('animal_imgs', 'train', animal), exist_ok=True)
    os.makedirs(os.path.join('animal_imgs', 'val', animal), exist_ok=True)
    os.makedirs(os.path.join('animal_imgs', 'test', animal), exist_ok=True)
    imgs = os.listdir(os.path.join('animal_imgs', animal))
    train_imgs = imgs[:int(len(imgs)*train_size)]
    val_imgs = imgs[int(len(imgs)*train_size):int(len(imgs)*(train_size+val_size))]
    test_imgs = imgs[int(len(imgs)*(train_size+val_size)):]
    for img in train_imgs:
        shutil.move(os.path.join('animal_imgs', animal, img), os.path.join('animal_imgs', 'train', animal, img))
    for img in val_imgs:
        shutil.move(os.path.join('animal_imgs', animal, img), os.path.join('animal_imgs', 'val', animal, img))
    for img in test_imgs:
        shutil.move(os.path.join('animal_imgs', animal, img), os.path.join('animal_imgs', 'test', animal, img))
    shutil.rmtree(os.path.join('animal_imgs', animal))


#%%


# %%
