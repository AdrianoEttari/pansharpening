#%%
import torch
from UNet_model_superres_new import Residual_Attention_UNet_superres,Attention_UNet_superres,Residual_Attention_UNet_superres_2,Residual_MultiHeadAttention_UNet_superres,Residual_Visual_MultiHeadAttention_UNet_superres
from PIL import Image, ImageFilter
from torchvision import transforms
import numpy as np
from torch.functional import F
import matplotlib.pyplot as plt
import os
from train_diffusion_superres import Diffusion
from tqdm import tqdm 
from Aggregation_Sampling import load_from_pickle, save_to_pickle, imgs_splitter, plot_patches, merge_images

model_name = 'DDP_Residual_Attention_UNet_superres_magnification8_ANIME50k_DownBlur_512'
magnification_factor = 8
snapshot_folder_path = os.path.join('models_run', model_name, 'weights')
snapshot_path = os.path.join(snapshot_folder_path, 'snapshot.pt')
UNet_type = 'residual attention unet'
noise_schedule = 'cosine'
image_size = 256
model_input_size = 64
number_of_patches = image_size // model_input_size
input_channels = output_channels = 3
device = 'mps'
Degradation_type = 'DownBlur'
noise_steps = 1500
blur_radius = 0.5

img_test = Image.open('anime_test.jpg')
transform = transforms.Compose([ transforms.ToTensor()])
img_test = transform(img_test)
img_test = img_test[:,:,list(np.arange(50,562))]
img_test = F.interpolate(img_test.unsqueeze(0), size=(image_size, image_size), mode='bicubic', align_corners=False).squeeze(0)
plt.imshow(img_test.permute(1,2,0))

# img_test_2 = Image.open('anime_data_10k/test_original/175490.jpg')
# downsample = transforms.Resize((512//magnification_factor, 512//magnification_factor),
#                                        interpolation=transforms.InterpolationMode.BICUBIC)
# img_test_2 = downsample(img_test_2)
# img_test_2 = img_test_2.filter(ImageFilter.GaussianBlur(blur_radius))
# img_test_2 = transforms.ToTensor()(img_test_2)
# %%
position_patch_dic = imgs_splitter(img_test, number_of_patches, model_input_size)

os.makedirs('aggregation_sampling',exist_ok=True)
save_to_pickle(os.path.join('aggregation_sampling','inputs.pkl'), position_patch_dic)
# %%
position_patch_dic = load_from_pickle(os.path.join('aggregation_sampling','inputs.pkl'))

plot_patches(position_patch_dic)
# %%
if UNet_type.lower() == 'attention unet':
    print('Using Attention UNet')
    model = Attention_UNet_superres(input_channels, output_channels, device).to(device)
elif UNet_type.lower() == 'residual attention unet':
    print('Using Residual Attention UNet')
    model = Residual_Attention_UNet_superres(input_channels, output_channels, device).to(device)
elif UNet_type.lower() == 'residual attention unet 2':
    print('Using Residual Attention UNet 2')
    model = Residual_Attention_UNet_superres_2(input_channels, output_channels, device).to(device)
elif UNet_type.lower() == 'residual multihead attention unet':
    print('Using Residual MultiHead Attention UNet')
    model = Residual_MultiHeadAttention_UNet_superres(input_channels, output_channels, device).to(device)
elif UNet_type.lower() == 'residual visual multihead attention unet':
    print('Using Residual Visual MultiHead Attention UNet')
    model = Residual_Visual_MultiHeadAttention_UNet_superres(input_channels, image_size ,output_channels, device).to(device)
else:
    raise ValueError('The UNet type must be either Attention UNet or Residual Attention UNet or Residual Attention UNet 2 or Residual MultiHead Attention UNet or Residual Visual MultiHeadAttention UNet superres')
# %%

diffusion = Diffusion(
        noise_schedule=noise_schedule, model=model,
        snapshot_path=snapshot_path,
        noise_steps=noise_steps, beta_start=1e-4, beta_end=0.02, 
        magnification_factor=magnification_factor,device=device,
        image_size=512, model_name=model_name, Degradation_type=Degradation_type)
# %%
# position_super_lr_patches_dic = {}

# for key,value in tqdm(position_patch_dic.items()):
#     super_lr_patch = diffusion.sample(1, model, value.to(device), input_channels=3, plot_gif_bool=False)
#     super_lr_img_test_2 = diffusion.sample(1, model, img_test_2.to(device), input_channels=3, plot_gif_bool=False)
#     position_super_lr_patches_dic[key] = super_lr_patch.to('cpu')

# save_to_pickle(os.path.join('aggregation_sampling','predictions.pkl'), position_super_lr_patches_dic)

#%%
position_super_lr_patches_dic = load_from_pickle(os.path.join('aggregation_sampling','predictions.pkl'))

plot_patches(position_super_lr_patches_dic)

# %%
merged_image = merge_images(position_super_lr_patches_dic)
merged_image = torch.clamp(merged_image, 0, 1)
plt.imshow(merged_image.permute(1,2,0))
plt.axis('off')
plt.show()

print(f'From {img_test.shape} to {merged_image.shape}')
# %%
tensor_to_pil = transforms.ToPILImage()
img_test = torch.clamp(img_test, 0, 1)
pil_img_test = tensor_to_pil(img_test)
pil_img_test = pil_img_test.resize((1024,1024), Image.BICUBIC)
pil_merged_image = tensor_to_pil(merged_image)
pil_merged_image = pil_merged_image.resize((1024,1024), Image.BICUBIC)
pil_img_test.save(os.path.join('lr_img_collage.png'))
pil_merged_image.save(os.path.join('super_lr_img_collage.png'))
# %%
from Aggregation_Sampling import ImageSpliterTh
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

img_test = Image.open('anime_test.jpg')
transform = transforms.Compose([transforms.Resize((512,512)), transforms.ToTensor()])
img_test = transform(img_test).unsqueeze(0)

pch_size = 64
stride = 59
aggregation_sampling = ImageSpliterTh(img_test, pch_size, stride, sf=1)

# %%
print(aggregation_sampling.num_pchs)
for im_lq_pch, index_infos in aggregation_sampling:
    print(im_lq_pch.shape, index_infos)
    plt.imshow(im_lq_pch[0].permute(1,2,0))
    plt.show()
print(aggregation_sampling.num_pchs)
# %%

im_spliter = ImageSpliterTh(img_test, pch_size, stride, sf=1)
for im_lq_pch, index_infos in im_spliter:
    # seed_everything(opt.seed)
    # init_latent = model.get_first_stage_encoding(model.encode_first_stage(im_lq_pch))  # move to latent space
    # text_init = ['']*opt.n_samples
    # semantic_c = model.cond_stage_model(text_init)
    noise = torch.randn_like(init_latent)
    # If you would like to start from the intermediate steps, you can add noise to LR to the specific steps.
    # t = repeat(torch.tensor([999]), '1 -> b', b=im_lq_bs.size(0))
    # t = t.to(device).long()
    # x_T = model.q_sample_respace(x_start=init_latent, t=t, sqrt_alphas_cumprod=sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod, noise=noise)
    # x_T = noise
    # samples, _ = model.sample_canvas(cond=semantic_c, struct_cond=init_latent, batch_size=im_lq_pch.size(0), timesteps=opt.ddpm_steps, time_replace=opt.ddpm_steps, x_T=x_T, return_intermediates=True, tile_size=int(opt.input_size/8), tile_overlap=opt.tile_overlap, batch_size_sample=opt.n_samples)
    # _, enc_fea_lq = vq_model.encode(im_lq_pch)
    x_samples = vq_model.decode(samples * 1. / model.scale_factor, enc_fea_lq)
    if opt.colorfix_type == 'adain':
        x_samples = adaptive_instance_normalization(x_samples, im_lq_pch)
    elif opt.colorfix_type == 'wavelet':
        x_samples = wavelet_reconstruction(x_samples, im_lq_pch)
    im_spliter.update_gaussian(x_samples, index_infos)
im_sr = im_spliter.gather()
im_sr = torch.clamp((im_sr+1.0)/2.0, min=0.0, max=1.0)
#%%

init_latent = model.get_first_stage_encoding(model.encode_first_stage(im_lq_bs))  # move to latent space
text_init = ['']*opt.n_samples
semantic_c = model.cond_stage_model(text_init)
noise = torch.randn_like(init_latent)
# If you would like to start from the intermediate steps, you can add noise to LR to the specific steps.
t = repeat(torch.tensor([999]), '1 -> b', b=im_lq_bs.size(0))
t = t.to(device).long()
x_T = model.q_sample_respace(x_start=init_latent, t=t, sqrt_alphas_cumprod=sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod, noise=noise)
# x_T = noise
samples, _ = model.sample_canvas(cond=semantic_c, struct_cond=init_latent, batch_size=im_lq_bs.size(0), timesteps=opt.ddpm_steps, time_replace=opt.ddpm_steps, x_T=x_T, return_intermediates=True, tile_size=int(opt.input_size/8), tile_overlap=opt.tile_overlap, batch_size_sample=opt.n_samples)
_, enc_fea_lq = vq_model.encode(im_lq_bs)
x_samples = vq_model.decode(samples * 1. / model.scale_factor, enc_fea_lq)
if opt.colorfix_type == 'adain':
    x_samples = adaptive_instance_normalization(x_samples, im_lq_bs)
elif opt.colorfix_type == 'wavelet':
    x_samples = wavelet_reconstruction(x_samples, im_lq_bs)
im_sr = torch.clamp((x_samples+1.0)/2.0, min=0.0, max=1.0)

if upsample_scale > opt.upscale:
    im_sr = F.interpolate(
                im_sr,
                size=(int(im_lq_bs.size(-2)*opt.upscale/upsample_scale),
                    int(im_lq_bs.size(-1)*opt.upscale/upsample_scale)),
                mode='bicubic',
                )
    im_sr = torch.clamp(im_sr, min=0.0, max=1.0)

im_sr = im_sr.cpu().numpy().transpose(0,2,3,1)*255   # b x h x w x c

if flag_pad:
    im_sr = im_sr[:, :ori_h, :ori_w, ]

for jj in range(im_lq_bs.shape[0]):
    img_name = str(Path(im_path_bs[jj]).name)
    basename = os.path.splitext(os.path.basename(img_name))[0]
    outpath = str(Path(opt.outdir)) + '/' + basename + '.png'
    Image.fromarray(im_sr[jj, ].astype(np.uint8)).save(outpath)