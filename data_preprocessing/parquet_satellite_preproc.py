
# %%
from fsspec.parquet import open_parquet_file
import pyarrow.parquet as pq
from io import BytesIO
from PIL import Image

PARQUET_FILE = '../part_00001.parquet' # parquet number
ROW_INDEX = 42 # row number (about 500 per parquet)

# url = "https://huggingface.co/datasets/Major-TOM/Core-S2L2A/resolve/main/images/{}.parquet".format(PARQUET_FILE)
# with open_parquet_file(PARQUET_FILE,columns = ["thumbnail"]) as f:
#     with pq.ParquetFile(f) as pf:
#         first_row_group = pf.read_row_group(ROW_INDEX, columns=['thumbnail'])

# stream = BytesIO(first_row_group['thumbnail'][0].as_py())
# image = Image.open(stream)

#%%
import pandas as pd

df = pd.read_parquet('../part_00001.parquet')
# %%
from torchvision import transforms
import torch
import matplotlib.pyplot as plt

# Initialize an empty list to store the individual channel tensors
channel_tensors = []

# Define the channel order
channels = [ 'B04', 'B02', 'B03']

for channel in channels:
    # Load the image data from BytesIO
    stream = BytesIO(df.iloc[100][channel])
    image = Image.open(stream)
    
    # Convert image to tensor
    image_tensor = transforms.ToTensor()(image)
    image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())
    # Append the channel tensor to the list
    channel_tensors.append(image_tensor)

# Concatenate the channel tensors along the channel dimension (dim=0)
rgb_tensor = torch.cat(channel_tensors, dim=0)

plt.imshow(rgb_tensor.permute(1,2,0))
# %%
