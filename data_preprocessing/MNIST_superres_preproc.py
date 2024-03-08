#%% Download the MNIST dataset from the torchvision library datasets and assign the dataset to mnist_dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders_MNIST(batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_dataset = datasets.MNIST(root='./MNIST_data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    val_dataset = datasets.MNIST(root='./MNIST_data', train=False, download=True, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader

train_loader, val_loader = get_dataloaders_MNIST(batch_size=16)
mnist_dataset_training = train_loader.dataset
mnist_dataset_val = val_loader.dataset
# %% Save just the images (not the labels) in a folder path MNIST_data_superres/train_original and MNIST_data_superres/val_original
import torch
import shutil
import os

os.makedirs('MNIST_data_superres/train_original', exist_ok=True)
os.makedirs('MNIST_data_superres/val_original', exist_ok=True)

for i, (img, label) in enumerate(mnist_dataset_training):
    torch.save(img, f'MNIST_data_superres/train_original/{i}_label{label}.pt')

for i, (img, label) in enumerate(mnist_dataset_val):
    torch.save(img, f'MNIST_data_superres/val_original/{i}_label{label}.pt')

shutil.rmtree('MNIST_data')
    
#%% Use the get_data class to create a dataset with the downsampled images as x and
# the original images as y. The downsampled images are obtained by resizing the original
from utils import get_data_superres
from torchvision import transforms
transform = transforms.Compose([transforms.ToTensor()])
dataset = get_data_superres('MNIST_data_superres/train_original', 4, data_format='torch', transform=transform)

import matplotlib.pyplot as plt
fig, axs = plt.subplots(5, 2, figsize=(10, 20))
for i in range(5):
    x, y = dataset[i]
    axs[i, 0].imshow(x.permute(1,2,0).numpy())
    axs[i, 0].set_title('Downsampled')
    axs[i, 1].imshow(y.permute(1,2,0).numpy())
    axs[i, 1].set_title('Original')
plt.show()
#%%
from torch.utils.data import DataLoader
from utils import get_data_superres
from torchvision import transforms
import matplotlib.pyplot as plt

transform = transforms.Compose([transforms.ToTensor()])
dataset = get_data_superres('MNIST_data_superres/train_original', 4, data_format='torch', transform=transform)

train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

for x, y in train_loader:
    plt.imshow(x[0].permute(1,2,0).numpy())
    plt.show()
    plt.imshow(y[0].permute(1,2,0).numpy())
    plt.show()
    break


# %%
