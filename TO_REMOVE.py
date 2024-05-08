#%%
from Aggregation_Sampling_mine import aggregation_sampling
from PIL import Image
from torchvision import transforms
import numpy as np
from torch.nn import functional as F
import matplotlib.pyplot as plt

image_size = 256
img_test = Image.open('anime_test.jpg')
transform = transforms.Compose([ transforms.ToTensor()])
img_test = transform(img_test)
img_test = img_test[:,:,list(np.arange(50,562))]
img_test_1 = F.interpolate(img_test.unsqueeze(0), size=(image_size, image_size), mode='bicubic', align_corners=False).squeeze(0)
img_test_2 = F.interpolate(img_test.unsqueeze(0), size=(image_size, image_size), mode='bilinear', align_corners=False).squeeze(0)

#%%
overlapping_pixels = 30
img_test_1 = img_test_1[:, :, list(np.arange(img_test_1.shape[2]//2+overlapping_pixels))]
img_test_2 = img_test_2[:, :, list(np.arange(img_test_2.shape[2]//2-overlapping_pixels,img_test_2.shape[2]))]
# %%
# plt.imshow(img_test_1.permute(1,2,0))
# plt.show()
# plt.imshow(img_test_2.permute(1,2,0))
# plt.show()
# %%
list1 = [0,256,128,128+overlapping_pixels]
list2 = [0,256,128-overlapping_pixels,128]
aggregated_img = aggregation_sampling(img_test_1, img_test_2, [list1, list2], overlapping_pixels)

# %%
plt.imshow(aggregated_img.permute(1,2,0))
plt.show()
# %%
