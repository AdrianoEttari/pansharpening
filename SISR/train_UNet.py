import os
import logging
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torchvision import datasets
import imageio
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from utils import get_data_superres
import copy

from UNet_model_SISR import Residual_Attention_UNet_superres, Attention_UNet_superres, EMA

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F

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

class SISR_UNet():
    def __init__(
            self,
            model: nn.Module,
            snapshot_path: str,
            magnification_factor=4,
            device='cuda',
            image_size=224,
            model_name='UNet superres'):

        self.image_size = image_size
        self.model_name = model_name
        self.magnification_factor = magnification_factor
        self.device = device
        self.snapshot_path = snapshot_path
        self.model = model.to(self.device)
        
        # If a snapshot exists, we load it
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot()

        # epoch_run is used by _save_snapshot and _load_snapshot to keep track of the current epoch (MAYBE WE CAN REMOVE IT FROM HERE)
        self.epochs_run = 0

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
            "MODEL_STATE": model.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def _load_snapshot(self):
        '''
        This function loads the model state, the optimizer state and the current epoch from a snapshot.
        It is a mandatory function in order to be fault tolerant. The reason is that if the training is interrupted, we can resume
        it from the last snapshot.
        '''
        snapshot = torch.load(self.snapshot_path, map_location=self.device)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def early_stopping(self, epoch, model, patience, epochs_without_improving):
        '''
        This function checks if the validation loss is increasing. If it is for more than patience times, then it returns True (that will correspond to breaking the training loop)
        and saves the weights of the model.
        '''
        if epochs_without_improving >= patience:
            print('Early stopping! Training stopped')
            self._save_snapshot(epoch, model)
            return True

    def train(self, lr, epochs, save_every, train_loader, val_loader, patience, loss, verbose):
        '''
        This function performs the training of the model, saves the snapshots and the model at the end of the training each self.every_n_epochs epochs.

        Input:
            lr: the learning rate
            epochs: the number of epochs
            save_every: the frequency at which the model weights will be saved
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

        ema = EMA(beta=0.995)
        ema_model = copy.deepcopy(model).eval().requires_grad_(False)

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
        elif loss == 'Perceptual':
            loss_function = VGGPerceptualLoss(self.device)
        
        epochs_without_improving = 0
        best_loss = float('inf')  

        for epoch in range(epochs):
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

                optimizer.zero_grad() # set the gradients to 0

                sr_img = model(lr_img, self.magnification_factor).to(self.device)

                train_loss = loss_function(sr_img, hr_img)
                
                train_loss.backward() # compute the gradients
                optimizer.step() # update the weights
                ema.step_ema(ema_model, model)
                
                if verbose:
                    pbar_train.set_postfix(LOSS=train_loss.item()) # set_postfix just adds a message or value displayed after the progress bar. In this case the loss of the current batch.
            
                running_train_loss += train_loss.item()
            
            # scheduler.step()

            running_train_loss /= len(train_loader.dataset) # at the end of each epoch I want the average loss
            print(f"Epoch {epoch}: Running Train ({loss}) {running_train_loss}")

            if epoch % save_every == 0:
                self._save_snapshot(epoch, ema_model)
                fig, axs = plt.subplots(5,4, figsize=(15,15))
                for i in range(5):
                    lr_img = val_loader.dataset[i][0]
                    hr_img = val_loader.dataset[i][1]

                    superres_img = self.sample(model=model, lr_img=lr_img)
                    superres_img_ema = self.sample(model=ema_model, lr_img=lr_img)

                    axs[i,0].imshow(lr_img.permute(1,2,0).cpu().numpy())
                    axs[i,0].set_title('Low resolution image')
                    axs[i,1].imshow(hr_img.permute(1,2,0).cpu().numpy())
                    axs[i,1].set_title('High resolution image')
                    axs[i,2].imshow(superres_img[0].permute(1,2,0).cpu().numpy())
                    axs[i,2].set_title('Super resolution image (model)')
                    axs[i,3].imshow(superres_img_ema[0].permute(1,2,0).cpu().numpy())
                    axs[i,3].set_title('Super resolution image (EMA model)')

                plt.savefig(os.path.join(os.getcwd(), 'models_run', self.model_name, 'results', f'superres_{epoch}_epoch.png'))


            if val_loader is not None:
                with torch.no_grad():
                    model.eval()
                    
                    for (lr_img,hr_img) in pbar_val:
                        lr_img = lr_img.to(self.device)

                        
                        sr_img = model(lr_img, self.magnification_factor).to(self.device)

                        val_loss = loss_function(sr_img, hr_img)

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
                else:
                    epochs_without_improving += 1

                if self.early_stopping(epoch, ema_model, patience, epochs_without_improving):
                    break
            print('Epochs without improving: ', epochs_without_improving)

    def sample(self, model, lr_img):
        '''
        This function samples n images from the model.

        Input:
            model: the model to sample from
            lr_img: the low resolution image

        Output:
            the sampled images
        '''
        model.eval()
        with torch.no_grad():
            sr_img = model(lr_img[None, ...].to(self.device), self.magnification_factor).to('cpu')
        return sr_img

def launch(args):
    '''
    This function is the main and call the training, the sampling and all the other functions in the Diffusion class.

    Input:
        image_size: the size of the images in the dataset
        dataset_path: the path of the dataset
        batch_size: the batch size
        lr: the learning rate
        epochs: the number of epochs
        save_every: specifies the frequency, in terms of epochs, at which the model weights will be saved
        snapshot_name: the name of the snapshot file
        snapshot_folder_path: the folder path where the snapshots will be saved
        model_name: the name of the model
        device: the device to use (cuda, cpu, mps)
        patience: the number of epochs after which the training will be stopped if the validation loss is increasing
        input_channels: the number of input channels
        output_channels: the number of output channels
        magnification_factor: the magnification factor (i.e. the factor by which the image is magnified in the super-resolution task)
        loss: the loss function to use
        UNet_type: the type of UNet to use (Attention UNet, Residual Attention UNet)

    Output:
        None
    '''
    image_size = args.image_size
    dataset_path = args.dataset_path
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    save_every = args.save_every
    snapshot_name = args.snapshot_name
    snapshot_folder_path = args.snapshot_folder_path
    model_name = args.model_name
    device = args.device
    patience = args.patience
    input_channels, output_channels = args.inp_out_channels, args.inp_out_channels
    magnification_factor = args.magnification_factor
    loss = args.loss
    UNet_type = args.UNet_type

    if image_size % magnification_factor != 0:
        raise ValueError('The image size must be a multiple of the magnification factor')
    
    os.makedirs(snapshot_folder_path, exist_ok=True)
    os.makedirs(os.path.join(os.curdir, 'models_run', model_name, 'results'), exist_ok=True)

    transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    ]) # The transforms.ToTensor() is in the get_data_superres function (in there
    # first is applied this transform to y, then the resize according to the magnification_factor
    # in order to get the x which is the lr_img and finally the to_tensor for both x
    # and y is applied)

    train_path = f'{dataset_path}/train_original'
    valid_path = f'{dataset_path}/val_original'

    train_dataset = get_data_superres(train_path, magnification_factor, 'PIL', transform)
    val_dataset = get_data_superres(valid_path, magnification_factor, 'PIL', transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)


    if UNet_type.lower() == 'attention unet':
        model = Attention_UNet_superres(input_channels, output_channels, device).to(device)
    elif UNet_type.lower() == 'residual attention unet':
        model = Residual_Attention_UNet_superres(input_channels, output_channels, device).to(device)
    else:
        raise ValueError('The UNet type must be either Attention UNet or Residual Attention UNet')
    print("Num params: ", sum(p.numel() for p in model.parameters()))

    snapshot_path = os.path.join(snapshot_folder_path, snapshot_name)

    SISR = SISR_UNet(model=model, snapshot_path=snapshot_path, magnification_factor=magnification_factor, device=device, image_size=image_size, model_name=model_name)

    # Training 
    SISR.train(
        lr=lr, epochs=epochs, save_every=save_every,
        train_loader=train_loader, val_loader=val_loader, patience=patience, loss=loss, verbose=True)
    
    # Sampling
    # Notice that the sampling is made with the EMA model because we just save the weights of the EMA model
    fig, axs = plt.subplots(5,3, figsize=(15,15))
    for i in range(5):
        lr_img = train_dataset[i][0]
        hr_img = train_dataset[i][1]

        superres_img = SISR.sample(model=model, lr_img=lr_img, input_channels=lr_img.shape[0])

        axs[i,0].imshow(lr_img.permute(1,2,0).cpu().numpy())
        axs[i,0].set_title('Low resolution image')
        axs[i,1].imshow(hr_img.permute(1,2,0).cpu().numpy())
        axs[i,1].set_title('High resolution image')
        axs[i,2].imshow(superres_img[0].permute(1,2,0).cpu().numpy())
        axs[i,2].set_title('Super resolution image')

    plt.savefig(os.path.join(os.getcwd(), 'models_run', model_name, 'results', 'superres_results.png'))


if __name__ == '__main__':
    import argparse     
    parser = argparse.ArgumentParser(description=' ')
    parser.add_argument('--epochs', type=int, default=501)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_size', type=int)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--save_every', type=int, default=20)
    parser.add_argument('--snapshot_name', type=str, default='snapshot.pt')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument('--inp_out_channels', type=int, default=3) # input channels must be the same of the output channels
    parser.add_argument('--loss', type=str)
    parser.add_argument('--magnification_factor', type=int, default=4)
    parser.add_argument('--UNet_type', type=str, default='Residual Attention UNet') # 'Attention UNet' or 'Residual Attention UNet'
    args = parser.parse_args()
    args.snapshot_folder_path = os.path.join(os.curdir, 'models_run', args.model_name, 'weights')
    launch(args)