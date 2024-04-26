import os
# import logging
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
# from torchvision import datasets
import imageio
# import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from utils import get_data_superres, get_data_superres_BSRGAN
# import copy

from UNet_model_superres_new import Residual_Attention_UNet_superres, Attention_UNet_superres, Residual_MultiHeadAttention_UNet_superres

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler

class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super(VGGPerceptualLoss, self).__init__()
        self.vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        # By using .features, we are considering just the convolutional part of the VGG network
        # without considering the avg pooling and the fully connected layers which map the features to the 1000 classes
        # and so, solve the classification task.
        # self.vgg = nn.Sequential(*[self.vgg[i] for i in range(8)]) 
        self.vgg.to(device)
        self.vgg.eval()  # Set VGG to evaluation mode
        self.device = device
        # Freeze all VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False
        
    def preprocess_image(self, image):
        '''
        The VGG network wants input sized (224,224), normalized and as pytorch tensor. 
        '''
        transform = transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        
        if image.shape[-1] != 224:
            try:
                image = F.interpolate(image, size=(224, 224), mode='bicubic', align_corners=False)
            except:
                image = F.interpolate(image.to('cpu'),  size=(224, 224), mode='bicubic', align_corners=False).to(self.device)
        
        return transform(image)
    
    def forward(self, x, y):
        x = self.preprocess_image(x)
        y = self.preprocess_image(y)

        x_features = self.vgg(x)
        y_features = self.vgg(y)

        return torch.mean((x_features-y_features)**2)

class CombinedLoss(nn.Module):
    def __init__(self, first_loss, second_loss, weight_first=0.5):
        super(CombinedLoss, self).__init__()
        self.first_loss = first_loss
        self.second_loss = second_loss
        self.weight_first = weight_first  

    def forward(self, predicted, target):
        first_loss_value = self.first_loss(predicted, target)
        second_loss_value = self.second_loss(predicted, target)
        combined_loss = self.weight_first * first_loss_value + (1-self.weight_first) * second_loss_value
        return combined_loss
    
class CombinedLoss_MSE_PercLoss(nn.Module):
    def __init__(self, MSE_loss, Perc_loss, weight_MSE_loss=0.5):
        super(CombinedLoss_MSE_PercLoss, self).__init__()
        self.MSE_loss = MSE_loss
        self.Perc_loss = Perc_loss
        self.weight_MSE_loss = weight_MSE_loss

    def forward(self, predicted_noise, target_noise, hr_img, hr_img_noised, alpha_hat_t, epoch):
        alpha_hat_t = alpha_hat_t[:, None, None, None]
        MSE_loss_value = self.MSE_loss(predicted_noise, target_noise)
        if epoch > 50:
            hr_img_denoised = (hr_img_noised - torch.sqrt(1-alpha_hat_t)*predicted_noise)/torch.sqrt(alpha_hat_t)
            Perc_loss_value = self.Perc_loss(hr_img_denoised, hr_img)
            combined_loss = self.weight_MSE_loss * MSE_loss_value + (1-self.weight_MSE_loss) * Perc_loss_value
            return combined_loss
        else:
            return MSE_loss_value

class Diffusion:
    def __init__(
            self,
            noise_schedule: str,
            model: nn.Module,
            snapshot_path: str,
            noise_steps=1000,
            beta_start=1e-4,
            beta_end=0.02,
            device='cuda',
            magnification_factor=4,
            image_size=224,
            model_name='superres',
            Degradation_type='BSRGAN'):
    
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.image_size = image_size
        self.model_name = model_name
        self.magnification_factor = magnification_factor
        self.device = device
        self.snapshot_path = snapshot_path
        self.Degradation_type=Degradation_type
        
        self.model = model.to(self.device)
        # epoch_run is used by _save_snapshot and _load_snapshot to keep track of the current epoch
        self.epochs_run = 0
        # If a snapshot exists, we load it
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot()

        self.noise_schedule = noise_schedule

        if self.noise_schedule == 'linear':
            self.beta = self.prepare_noise_schedule().to(self.device) 
            self.alpha = 1. - self.beta
            self.alpha_hat = torch.cumprod(self.alpha, dim=0) # Notice that beta is not just a number. It is a tensor of shape (noise_steps,).
        # If we are in the step t then we index the tensor with t. To get alpha_hat we compute the cumulative product of the tensor.

        elif self.noise_schedule == 'cosine':
            self.alpha_hat = self.prepare_noise_schedule().to(self.device)
            self.beta = self.from_alpha_hat_to_beta()
            self.alpha = 1. - self.beta

    def from_alpha_hat_to_beta(self):
        '''
        This function is necessary because it allows to get from the alpha hat that we got with the cosine schedule
        the alpha and the beta which are necessary in order to compute the denoised image during sampling.
        The reason we need this function is that with the linear schedule we start from beta, then we calculate alpha and so
        alpha hat, whereas with the cosine schedule we start from alpha hat, then we must calculate beta and so alpha
        because we need them to compute the denoised image.

        Input:
            alpha_hat: a tensor of shape (noise_steps,) that contains the alpha_hat values for each noise step.
        
        Output:
            beta: a tensor of shape (noise_steps,) that contains the beta values for each noise step.
        '''
        beta = []
        for t in range(len(self.alpha_hat)-1, 0, -1):
            beta.append(1 -(self.alpha_hat[t]/self.alpha_hat[t-1]))
        beta.append(1 - self.alpha_hat[0])
        beta =  torch.tensor(beta[::-1], dtype=self.alpha_hat.dtype, device=self.alpha_hat.device)
        return beta

    def prepare_noise_schedule(self):
        '''
        In this function we set the noise schedule to use. Basically, we need to know how much gaussian noise we want to add
        for each noise step.

        Input:
            noise_schedule: the name of the noise schedule to use. It can be either 'linear' or 'cosine'.

        Output:
            if noise_schedule == 'linear':
                self.beta: a tensor of shape (noise_steps,) that contains the beta values for each noise step.
            elif noise_schedule == 'cosine':
                self.alpha_hat: a tensor of shape (noise_steps,) that contains the alpha_hat values for each noise step.
        '''
        if self.noise_schedule == 'linear':
            return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        elif self.noise_schedule == 'cosine':
            f_t = torch.cos(((((torch.arange(self.noise_steps)/self.noise_steps)+0.008)/(1+0.008))*torch.pi/2))**2 # Here we apply the formula of the OpenAI paper https://arxiv.org/pdf/2102.09672.pdf
            alpha_hat = f_t/f_t[0]  
            return alpha_hat

    def noise_images(self, x, t):
        '''
        ATTENTION: The error epsilon is random, but how much of it we add to move forward depends on the Beta schedule.

        Input:
            x: the image at time t=0
            t: the current timestep
        
        Output:
            x_t: the image at the current timestep (x_t)
            epsilon: the error that we add to x_t to move forward
        '''
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None] # Each None is a new dimension (e.g.
        # if a tensor has shape (2,3,4), a[None,None,:,None] will be shaped (1,1,2,1,3,4)). It doens't add
        # them exatly in the same place, but it adds them in the place where the None is.
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x, dtype=torch.float32) # torch.randn_like() returns a tensor of the same shape of x with random values from a standard gaussian
        # (notice that the values inside x are not relevant)
        # epsilon = 2 * (epsilon - epsilon.min()) / (epsilon.max() - epsilon.min()) - 1
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n):
        '''
        During the training we sample t from a Uniform discrete distribution (from 1 to T)

        For each image that I have in the training, I want to sample a timestep t from a uniform distribution
        (notice that it is not the same for each image). 

        Input:
            n: the number of images we want to sample the timesteps for

        Output:
            t: a tensor of shape (n,) that contains the timesteps for each image
        '''
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    
    def sample(self, n, model, lr_img, input_channels=3, plot_gif_bool=False):
        '''
        As the name suggests this function is used for sampling. Therefore we want to 
        loop backward (moreover, notice that in the sample we want to perform EVERY STEP CONTIGUOUSLY).

        What we do is to predict the noise conditioned by the time step and by the low resolution image.

        Input:
            n: the number of images we want to sample
            lr_img: the low resolution image
            input_channels: the number of input channels
            plot_gif_bool: if True, the function will plot a gif with the generated images for each class
        
        Output:
            x: a tensor of shape (n, input_channels, self.image_size, self.image_size) with the generated images
        '''
        lr_img = lr_img.to(self.device).unsqueeze(0)

        frames = []
        model.eval() # disables dropout and batch normalization
        with torch.no_grad(): # disables gradient calculation
            if self.Degradation_type.lower() == 'blurdown':
                x = torch.randn((n, input_channels, self.image_size, self.image_size))
            elif self.Degradation_type.lower() == 'bsrgan':
                x = torch.randn((n, input_channels, self.image_size*self.magnification_factor, self.image_size*self.magnification_factor))
            # x = x-x.min()/(x.max()-x.min()) # normalize the values between 0 and 1
            x = x.to(self.device) # generates n noisy images of shape (3, self.image_size, self.image_size)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0): 
                t = (torch.ones(n) * i).long().to(self.device) # tensor of shape (n) with all the elements equal to i.
                # Basically, each of the n image will be processed with the same integer time step t.
                
                predicted_noise = model(x, t, lr_img, self.magnification_factor)

                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    # If i>1 then we add noise to the image we have sampled (remember that from x_t we sample x_{t-1}).
                    # If i==1 we sample x_0, which is the final image we want to generate, so we don't add noise.
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x) # we don't add noise (it's equal to 0) in the last time step because it would just make the final outcome worse.
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                if plot_gif_bool == True:
                    plt.imshow(x[0][0].cpu().numpy())
                    plt.savefig(os.path.join(os.getcwd(), 'models_run', self.model_name, 'results', f'frame_{i}.png'))
                    plt.title(f't-step={i}', fontsize=30)
                    frames.append(imageio.imread(os.path.join(os.getcwd(), 'models_run', self.model_name, 'results', f'frame_{i}.png')))
                    os.remove(os.path.join(os.getcwd(), 'models_run', self.model_name, 'results', f'frame_{i}.png'))
        if plot_gif_bool == True:
            imageio.mimsave(os.path.join(os.getcwd(), 'models_run', self.model_name, 'results', 'gif_result.gif'), frames, duration=0.25) 
        model.train() # enables dropout and batch normalization
        # x = (x.clamp(-1, 1) + 1) / 2 # clamp takes a minimum and a maximum. All the terms that you pass
        # as input to it are then modified: if their are less than the minimum, clamp outputs the minimum, 
        # otherwise outputs them. The same (but opposit reasoning) for the maximum.
        # +1 and /2 just to bring the values back to 0 to 1.
        # x = (x * 255).type(torch.uint8)
        return x

    def _save_snapshot(self, epoch, model):
        '''
        This function saves the model state, the optimizer state and the current epoch.
        It is a mandatory function in order to be fault tolerant.

        Input:
            epoch: the current epoch
            model: the model to save

        Output:
            None
        '''
        snapshot = {
            "MODEL_STATE": model.module.state_dict(),
            "EPOCHS_RUN": epoch,
            # "OPTIMIZER":self.optimizer.state_dict(),
            # "LR_SCHEDULER":self.lr_scheduler.state_dict(),
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def _load_snapshot(self):
        '''
        This function loads the model state, the optimizer state and the current epoch from a snapshot.
        It is a mandatory function in order to be fault tolerant. The reason is that if the training is interrupted, we can resume
        it from the last snapshot.
        '''
        from collections import OrderedDict
        print(self.device)
        print(self.snapshot_path)

        snapshot = torch.load(self.snapshot_path, map_location='cpu')
        model_state = OrderedDict((key.replace('module.', ''), value) for key, value in snapshot['MODEL_STATE'].items())
        self.model.module.load_state_dict(model_state)
        self.model.module.to(self.device)
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def early_stopping(self, patience, epochs_without_improving):
        '''
        This function checks if the validation loss is increasing. If it is for more than patience times,
        then it returns True (that will correspond to breaking the training loop).
        '''
        if epochs_without_improving >= patience:
            print('Early stopping! Training stopped')
            return True

    def train(self, lr, epochs, check_preds_epoch, train_loader, val_loader, patience, loss, verbose):
        '''
        This function performs the training of the model, saves the snapshots and the model at the end of the training each self.every_n_epochs epochs.

        Input:
            lr: the learning rate
            epochs: the number of epochs
            check_preds_epoch: the frequency at which the model will save the predictions
            train_loader: the training loader
            val_loader: the validation loader
            patience: the number of epochs after which the training will be stopped if the validation loss is increasing
            loss: the loss function to use
            verbose: if True, the function will use the tqdm during the training and the validation
        '''

        model = self.model

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # optimizer = torch.optim.AdamW(model.parameters(), lr=lr) # AdamW is a variant of Adam that adds weight decay (L2 regularization)
        # Basically, weight decay is a regularization technique that penalizes large weights. It's a way to prevent overfitting. In AdamW, 
        # the weight decay is added to the gradient and not to the weights. This is because the weights are updated in a different way in AdamW.
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        if loss == 'MSE':
            loss_function = nn.MSELoss()
        elif loss == 'MAE':
            loss_function = nn.L1Loss()
        elif loss == 'Huber':
            loss_function = nn.HuberLoss() 
        elif loss == 'MSE+Perceptual_noise':
            vgg_loss = VGGPerceptualLoss(self.device)
            mse_loss = nn.MSELoss()
            loss_function = CombinedLoss(first_loss=mse_loss, second_loss=vgg_loss, weight_first=0.3)
        elif loss == 'MSE+Perceptual_imgs':
            vgg_loss = VGGPerceptualLoss(self.device)
            mse_loss = nn.MSELoss()
            loss_function = CombinedLoss_MSE_PercLoss(MSE_loss=mse_loss, Perc_loss=vgg_loss, weight_MSE_loss=0.3)
        
        epochs_without_improving = 0
        best_loss = float('inf')  

        for epoch in range(self.epochs_run, epochs):
            train_loader.sampler.set_epoch(epoch)
            if verbose:
                pbar_train = tqdm(train_loader,desc='Training', position=0)
                if val_loader is not None:
                    pbar_val = tqdm(val_loader,desc='Validation', position=0)
            else:
                pbar_train = train_loader
                if val_loader is not None:
                    pbar_val = val_loader

            running_train_loss = 0.0
            running_val_loss = 0.0
            
            model.train()
            for i,(lr_img,hr_img) in enumerate(pbar_train):
                
                lr_img = lr_img.to(self.device)
                hr_img = hr_img.to(self.device)

                t = self.sample_timesteps(hr_img.shape[0]).to(self.device)
                # t is a unidimensional tensor of shape (images.shape[0] that is the batch_size) with random integers from 1 to noise_steps.
                x_t, noise = self.noise_images(hr_img, t) # get batch_size noise images

                optimizer.zero_grad() # set the gradients to 0

                predicted_noise = model(x_t, t, lr_img, self.magnification_factor) 

                if loss == 'MSE' or loss == 'MAE' or loss == 'Huber' or loss == 'MSE+Perceptual_noise':
                    train_loss = loss_function(predicted_noise, noise)
                elif loss == 'MSE+Perceptual_imgs':
                    train_loss = loss_function(predicted_noise, noise, hr_img, x_t, self.alpha_hat[t], epoch)
                
                train_loss.backward() # compute the gradients
                optimizer.step() # update the weights
                
                if verbose:
                    pbar_train.set_postfix(LOSS=train_loss.item()) # set_postfix just adds a message or value displayed after the progress bar. In this case the loss of the current batch.
            
                running_train_loss += train_loss.item()
            
            # scheduler.step()

            running_train_loss /= len(train_loader.dataset) # at the end of each epoch I want the average loss
            print(f"Epoch {epoch}: Running Train ({loss}) {running_train_loss}")

            if self.device==0 and epoch % check_preds_epoch == 0:
                if val_loader is None:
                    self._save_snapshot(epoch, model)

                fig, axs = plt.subplots(5,4, figsize=(15,15))
                for i in range(5):
                    lr_img = val_loader.dataset[i][0]
                    hr_img = val_loader.dataset[i][1]

                    superres_img = self.sample(n=1,model=model, lr_img=lr_img, input_channels=lr_img.shape[0], plot_gif_bool=False)
                    residual_img = hr_img.to(self.device) - superres_img[0].to(self.device)

                    axs[i,0].imshow(lr_img.permute(1,2,0).cpu().numpy())
                    axs[i,0].set_title('Low resolution image')
                    axs[i,1].imshow(hr_img.permute(1,2,0).cpu().numpy())
                    axs[i,1].set_title('High resolution image')
                    axs[i,2].imshow(superres_img[0].permute(1,2,0).cpu().numpy())
                    axs[i,2].set_title('Super resolution image')
                    axs[i,3].hist(residual_img.permute(1,2,0).cpu().numpy().ravel())
                    axs[i,3].set_title('Residual image')

                plt.savefig(os.path.join(os.getcwd(), 'models_run', self.model_name, 'results', f'superres_{epoch}_epoch.png'))

            if val_loader is not None:
                with torch.no_grad():
                    model.eval()
                    
                    for (lr_img,hr_img) in pbar_val:
                        lr_img = lr_img.to(self.device)
                        hr_img = hr_img.to(self.device)

                        t = self.sample_timesteps(hr_img.shape[0]).to(self.device)
                        # t is a unidimensional tensor of shape (images.shape[0] that is the batch_size)with random integers from 1 to noise_steps.
                        x_t, noise = self.noise_images(hr_img, t) # get batch_size noise images
                        
                        predicted_noise = model(x_t, t, lr_img, self.magnification_factor) 

                        if loss == 'MSE' or loss == 'MAE' or loss == 'Huber' or loss == 'MSE+Perceptual_noise':
                            val_loss = loss_function(predicted_noise, noise)
                        elif loss == 'MSE+Perceptual_imgs':    
                            val_loss = loss_function(predicted_noise, noise, hr_img, x_t, self.alpha_hat[t], epoch)

                        if verbose:
                            pbar_val.set_postfix(LOSS=val_loss.item()) # set_postfix just adds a message or value
                        # displayed after the progress bar. In this case the loss of the current batch.

                        running_val_loss += val_loss.item()

                    running_val_loss /= len(val_loader.dataset)
                    print(f"Epoch {epoch}: Running Val loss ({loss}){running_val_loss}")


            if val_loader is not None:
                if running_val_loss < best_loss - 0:
                    best_loss = running_val_loss
                    epochs_without_improving = 0
                    if self.device==0:
                        self._save_snapshot(epoch, model)
                else:
                    epochs_without_improving += 1

                if self.early_stopping(patience, epochs_without_improving):
                    break
            print('Epochs without improving: ', epochs_without_improving)

def launch(args):
    '''
    This function is the main and call the training, the sampling and all the other functions in the Diffusion class.

    Input:
        image_size: the size of the images in the dataset
        dataset_path: the path of the dataset
        batch_size: the batch size
        lr: the learning rate
        epochs: the number of epochs
        noise_schedule: the noise schedule (linear, cosine)
        check_preds_epoch: specifies the frequency, in terms of epochs, at which the model will perform predictions and save them. Moreover,
            if val_loader=None then the weights of the model will be saved at this frequency.
        snapshot_name: the name of the snapshot file
        snapshot_folder_path: the folder path where the snapshots will be saved
        model_name: the name of the model
        noise_steps: the number of noise steps
        patience: the number of epochs after which the training will be stopped if the validation loss is increasing
        input_channels: the number of input channels
        output_channels: the number of output channels
        plot_gif_bool: if True, the function will plot a gif with the generated images for each class
        magnification_factor: the magnification factor (i.e. the factor by which the image is magnified in the super-resolution task)
        loss: the loss function to use
        UNet_type: the type of UNet to use (Attention UNet, Residual Attention UNet)
        Degradation_type: the type of degradation to use (BSRGAN, BlurDown)

    Output:
        None
    '''
    image_size = args.image_size
    dataset_path = args.dataset_path
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    noise_schedule = args.noise_schedule
    check_preds_epoch = args.check_preds_epoch
    snapshot_name = args.snapshot_name
    snapshot_folder_path = args.snapshot_folder_path
    model_name = args.model_name
    noise_steps = args.noise_steps
    patience = args.patience
    input_channels, output_channels = args.inp_out_channels, args.inp_out_channels
    plot_gif_bool = args.plot_gif_bool
    magnification_factor = args.magnification_factor
    loss = args.loss
    UNet_type = args.UNet_type
    Degradation_type = args.Degradation_type
    num_crops = args.num_crops

    print(f'Using {Degradation_type} degradation')
    os.makedirs(snapshot_folder_path, exist_ok=True)
    os.makedirs(os.path.join(os.curdir, 'models_run', model_name, 'results'), exist_ok=True)
    
    init_process_group(backend="nccl") # nccl stands for NVIDIA Collective Communication Library. It is used for distributed comunications across multiple GPUs.

    if Degradation_type.lower() == 'blurdown':
        if image_size % magnification_factor != 0:
            raise ValueError('The image size must be a multiple of the magnification factor')
        
        transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        ]) # The transforms.ToTensor() is in the get_data_superres function (in there
        # first is applied this transform to y, then the resize according to the magnification_factor
        # in order to get the x which is the lr_img and finally the to_tensor for both x
        # and y is applied)

        train_path = f'{dataset_path}/train_original'
        valid_path = f'{dataset_path}/val_original'

        train_dataset = get_data_superres(train_path, magnification_factor, 0.5, 'PIL', transform)
        val_dataset = get_data_superres(valid_path, magnification_factor, 0.5, 'PIL', transform)
        
    elif Degradation_type.lower() == 'bsrgan':
        train_path = f'{dataset_path}/train_original'
        valid_path = f'{dataset_path}/val_original'

        train_dataset = get_data_superres_BSRGAN(train_path, magnification_factor, image_size, num_crops=num_crops, destination_folder=os.path.join(dataset_path+'_Dataset', 'train'))
        val_dataset = get_data_superres_BSRGAN(valid_path, magnification_factor, image_size, num_crops=num_crops, destination_folder=os.path.join(dataset_path+'_Dataset', 'val'))

    else:
        raise ValueError('The degradation type must be either BSRGAN or BlurDown')

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, sampler=DistributedSampler(train_dataset))
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,shuffle=False, sampler=DistributedSampler(val_dataset))

    gpu_id = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(int(gpu_id))
    device = gpu_id

    if UNet_type.lower() == 'attention unet':
        print('Using Attention UNet')
        model = Attention_UNet_superres(input_channels, output_channels, device).to(device)
    elif UNet_type.lower() == 'residual attention unet':
        print('Using Residual Attention UNet')
        model = Residual_Attention_UNet_superres(input_channels, output_channels, device).to(device)
    elif UNet_type.lower() == 'residual multihead attention unet':
        print('Using Residual MultiHead Attention UNet')
        model = Residual_MultiHeadAttention_UNet_superres(input_channels, output_channels, device).to(device)
    else:
        raise ValueError('The UNet type must be either Attention UNet or Residual Attention UNet or Residual MultiHead Attention UNet')
    
    print("Num params: ", sum(p.numel() for p in model.parameters()))

    model = DDP(model, device_ids=[device], find_unused_parameters=True)

    snapshot_path = os.path.join(snapshot_folder_path, snapshot_name)

    diffusion = Diffusion(
        noise_schedule=noise_schedule, model=model,
        snapshot_path=snapshot_path,
        noise_steps=noise_steps, beta_start=1e-4, beta_end=0.02, 
        magnification_factor=magnification_factor,device=device,
        image_size=image_size, model_name=model_name, Degradation_type=Degradation_type)
        
    # Training 
    diffusion.train(
        lr=lr, epochs=epochs, check_preds_epoch=check_preds_epoch,
        train_loader=train_loader, val_loader=val_loader, patience=patience, loss=loss, verbose=True)
    
    destroy_process_group()

    # Sampling
    fig, axs = plt.subplots(5,4, figsize=(15,15))
    for i in range(5):
        lr_img = train_dataset[i][0]
        hr_img = train_dataset[i][1]

        superres_img = diffusion.sample(n=1,model=model, lr_img=lr_img, input_channels=lr_img.shape[0], plot_gif_bool=plot_gif_bool)
        residual_img = hr_img.to(device) - superres_img[0].to(device)

        axs[i,0].imshow(lr_img.permute(1,2,0).cpu().numpy())
        axs[i,0].set_title('Low resolution image')
        axs[i,1].imshow(hr_img.permute(1,2,0).cpu().numpy())
        axs[i,1].set_title('High resolution image')
        axs[i,2].imshow(superres_img[0].permute(1,2,0).cpu().numpy())
        axs[i,2].set_title('Super resolution image')
        axs[i,3].hist(residual_img.permute(1,2,0).cpu().numpy().ravel())
        axs[i,3].set_title('Residual image')

    plt.savefig(os.path.join(os.getcwd(), 'models_run', model_name, 'results', 'superres_results.png'))


if __name__ == '__main__':
    import argparse     
    parser = argparse.ArgumentParser(description=' ')
    parser.add_argument('--epochs', type=int, default=501)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_size', type=int)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--check_preds_epoch', type=int, default=20)
    parser.add_argument('--noise_schedule', type=str, default='cosine')
    parser.add_argument('--snapshot_name', type=str, default='snapshot.pt')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--noise_steps', type=int, default=200)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument('--inp_out_channels', type=int, default=3) # input channels must be the same of the output channels
    parser.add_argument('--plot_gif_bool', type=bool, default=False)
    parser.add_argument('--loss', type=str)
    parser.add_argument('--magnification_factor', type=int)
    parser.add_argument('--UNet_type', type=str, default='Residual Attention UNet') # 'Attention UNet' or 'Residual Attention UNet' or 'Residual MultiHead Attention UNet'
    parser.add_argument('--Degradation_type', type=str, default='BlurDown') # 'BSRGAN' or 'BlurDown'
    parser.add_argument('--num_crops', type=int, default=1)
    args = parser.parse_args()
    args.snapshot_folder_path = os.path.join(os.curdir, 'models_run', args.model_name, 'weights')
    launch(args)