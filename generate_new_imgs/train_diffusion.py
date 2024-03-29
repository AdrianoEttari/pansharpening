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

from UNet_model import SimpleUNet

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
            image_size=224,
            model_name='img_generation'):
    
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.image_size = image_size
        self.model_name = model_name
        self.device = device

        self.snapshot_path = snapshot_path
        self.model = model.to(self.device)

        # If a snapshot exists, we load it
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot()

        # epoch_run is used by _save_snapshot and _load_snapshot to keep track of the current epoch (MAYBE WE CAN REMOVE IT FROM HERE)
        self.epochs_run = 0

        self.noise_schedule = noise_schedule

        if self.noise_schedule == 'linear':
            self.beta = self.prepare_noise_schedule().to(self.device) 
            self.alpha = 1. - self.beta
            self.alpha_hat = torch.cumprod(self.alpha, dim=0) # Notice that beta is not just a number, it is a tensor of shape (noise_steps,).
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
        ATTENTION: The error Ɛ is random, but how much of it we add to move forward depends on the Beta schedule.

        Input:
            x: the image at time t=0
            t: the current timestep
        
        Output:
            x_t: the image at the current timestep (x_t)
            Ɛ: the error that we add to x_t to move forward
        '''
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None] # Each None is a new dimension (e.g.
        # if a tensor has shape (2,3,4), a[None,None,:,None] will be shaped (1,1,2,1,3,4)). It doens't add
        # them exatly in the same place, but it adds them in the place where the None is.
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x, dtype=torch.float32) # torch.randn_like() returns a tensor of the same shape of x with random values from a standard gaussian
        # (notice that the values inside x are not relevant)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

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
    
    def sample(self, n, target_class=None, cfg_scale=3, input_channels=3, plot_gif_bool=False):
        '''
        As the name suggests this function is used for sampling. Therefore we want to 
        loop backward (moreover, notice that in the sample we want to perform EVERY STEP CONTIGUOUSLY).

        What we do is to predict the noise conditionally, then if the cfg_scale is > 0,
        we also predict the noise unconditionally. Eventually, we apply the formula
        out of the CFG paper using the torch.lerp function which does exactly the same

        predicted_noise = uncond_predicted_noise + cfg_scale * (predicted_noise - uncond_predicted_noise)

        and this will be our final predicted noise.

        Input:
            n: the number of images we want to sample
            target_class: the target class for the images
            cfg_scale: the scale of the CFG noise
            input_channels: the number of input channels
            plot_gif_bool: if True, the function will plot a gif with the generated images for each class
        
        Output:
            x: a tensor of shape (n, input_channels, self.image_size, self.image_size) with the generated images
        '''
        model = self.model

        frames = []
        model.eval() # disables dropout and batch normalization
        with torch.no_grad(): # disables gradient calculation
            x = torch.randn((n, input_channels, self.image_size, self.image_size)).to(self.device) # generates n noisy images of shape (3, self.image_size, self.image_size)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0): 
                t = (torch.ones(n) * i).long().to(self.device) # tensor of shape (n) with all the elements equal to i.
                # Basically, each of the n image will be processed with the same integer time step t.

                predicted_noise = model(x, t, target_class)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale) # Compute the formula of the CFG paper https://arxiv.org/abs/2207.12598

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
                    os.makedirs(f'{os.getcwd()}/gif_result', exist_ok=True)
                    plt.imshow(x[0][0].cpu().numpy(), cmap='gray')
                    plt.savefig(f'{os.getcwd()}/gif_result/frame_{i}.png')
                    plt.title(f't-step={i}', fontsize=30)
                    frames.append(imageio.imread(f'{os.getcwd()}/gif_result/frame_{i}.png'))
                    os.remove(f'{os.getcwd()}/gif_result/frame_{i}.png')
        if plot_gif_bool == True:
            imageio.mimsave(os.path.join(f'{os.getcwd()}/gif_result',f'gif_{self.model_name}_{target_class}.gif'), frames, duration=0.25) 
        model.train() # enables dropout and batch normalization
        # x = (x.clamp(-1, 1) + 1) / 2 # clamp takes a minimum and a maximum. All the terms that you pass
        # as input to it are then modified: if their are less than the minimum, clamp outputs the minimum, 
        # otherwise outputs them. The same (but opposit reasoning) for the maximum.
        # +1 and /2 just to bring the values back to 0 to 1.
        # x = (x * 255).type(torch.uint8)
        return x

    def _save_snapshot(self, epoch):
        '''
        This function saves the model state, the optimizer state and the current epoch.
        It is a mandatory function in order to be fault tolerant.

        Input:
            epoch: the current epoch

        Output:
            None
        '''
        snapshot = {
            "MODEL_STATE": self.model.state_dict(),
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

    def early_stopping(self, epoch, patience, epochs_without_improving):
        '''
        This function checks if the validation loss is increasing. If it is for more than patience times, then it returns True (that will correspond to breaking the training loop)
        and saves the weights of the model.
        '''
        if epochs_without_improving >= patience:
            print('Early stopping! Training stopped')
            self._save_snapshot(epoch)
            return True

    def train(self, lr, epochs, save_every, train_loader, val_loader, patience, verbose):
        '''
        This function performs the training of the model, saves the snapshots and the model at the end of the training each self.every_n_epochs epochs.

        Input:
            lr: the learning rate
            epochs: the number of epochs
            save_every: the frequency at which the model weights will be saved
            train_loader: the training loader
            val_loader: the validation loader
            patience: the number of epochs after which the training will be stopped if the validation loss is increasing
            verbose: if True, the function will use the tqdm during the training and the validation
        '''
        model = self.model

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr) # AdamW is a variant of Adam that adds weight decay (L2 regularization)
        # Basically, weight decay is a regularization technique that penalizes large weights. It's a way to prevent overfitting. In AdamW, 
        # the weight decay is added to the gradient and not to the weights. This is because the weights are updated in a different way in AdamW.
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        mse = nn.MSELoss()

        epochs_without_improving = 0
        best_loss = float('inf')  

        for epoch in range(epochs):
            if verbose:
                pbar_train = tqdm(train_loader,desc='Training')
                if val_loader is not None:
                    pbar_val = tqdm(val_loader,desc='Validation')
            else:
                pbar_train = train_loader
                if val_loader is not None:
                    pbar_val = val_loader

            running_train_loss = 0.0
            running_val_loss = 0.0

            model.train()
            for i, (img,label) in enumerate(pbar_train):
                label = label.to(self.device)
                img = img.to(self.device)

                t = self.sample_timesteps(img.shape[0]).to(self.device)
                # t is a unidimensional tensor of shape (images.shape[0] that is the batch_size) with random integers from 1 to noise_steps.
                x_t, noise = self.noise_images(img, t) # here I'm going to get batch_size noise images

                optimizer.zero_grad() # set the gradients to 0
                if np.random.random() < 0.1: # 10% of the time, don't pass labels (we train 10% of the times uncoditionally and 90% conditionally)
                    label = None
                predicted_noise = model(x_t, t, label) # here we pass the plain labels to the model (e.g. 0,1,2,...,9 if there are 10 classes)

                train_loss = mse(predicted_noise, noise)
                train_loss.backward() # compute the gradients
                optimizer.step() # update the weights

                pbar_train.set_postfix(MSE_LOSS=train_loss.item()) # set_postfix just adds a message or value displayed after the progress bar. In this case the loss of Batch i.
            
                running_train_loss += train_loss.item()
            
            scheduler.step()

            running_train_loss /= len(train_loader.dataset) # at the end of each epoch I want the average loss
            print(f"Epoch {epoch}: Running Train loss (MSE) {running_train_loss}")
            # logger.info(f"Epoch {epoch}: Running Train loss {running_train_loss}")
            # train_writer.add_scalar('Loss/train (MSE)', running_train_loss, epoch)

            if epoch % save_every == 0:
                self._save_snapshot(epoch)
                fig, axs = plt.subplots(10,5, figsize=(15,10))
                for i in range(10):
                    prediction = self.sample(5, target_class=torch.tensor([i], dtype=torch.int64).to(self.device), input_channels=train_loader.dataset[0][0].shape[0], plot_gif_bool=False) 
                    for j in range(5):
                        axs[i,j].imshow(prediction[j][0].cpu().numpy(), cmap='gray')
                        axs[i,j].axis('off')
                fig.savefig(os.path.join(os.curdir, 'models_run', self.model_name, 'results', f'predictions_{epoch}_epoch.png'))

            if val_loader is not None:
                with torch.no_grad():
                    model.eval()
                    for i, (img,label) in enumerate(pbar_val):
                        label = label.to(self.device)
                        img = img.to(self.device)

                        t = self.sample_timesteps(img.shape[0]).to(self.device)
                        # t is a unidimensional tensor of shape (images.shape[0] that is the batch_size)with random integers from 1 to noise_steps.
                        x_t, noise = self.noise_images(img, t) # here I'm going to get batch_size noise images
                        
                        if np.random.random() < 0.1: # 10% of the time, don't pass labels (we train 10% of the times uncoditionally and 90% conditionally)
                            label = None
                        predicted_noise = model(x_t, t, label) # here we pass the plain labels to the model (i.e. 0,1,2,...,9 if there are 10 classes)
                        # predicted_noise = model(x_t, t, None)

                        val_loss = mse(predicted_noise, noise)

                        pbar_val.set_postfix(MSE_LOSS=val_loss.item()) # set_postfix just adds a message or value
                        # displayed after the progress bar. In this case the loss of Batch i.

                        running_val_loss += val_loss.item()

                    running_val_loss /= len(val_loader.dataset)
                    print(f"Epoch {epoch}: Running Val loss (MSE){running_val_loss}")


            if val_loader is not None:
                if running_val_loss < best_loss - 0:
                    best_loss = running_val_loss
                    epochs_without_improving = 0
                else:
                    epochs_without_improving += 1

                if self.early_stopping(epoch, patience, epochs_without_improving):
                    break
                

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
        save_every: Specifies the frequency, in terms of epochs, at which the model weights will be saved
        snapshot_name: the name of the snapshot file
        snapshot_folder_path: the folder path where the snapshots will be saved
        model_name: the name of the model
        noise_steps: the number of noise steps
        device: the device to use (cuda, cpu, mps)
        patience: the number of epochs after which the training will be stopped if the validation loss is increasing
        num_classes: the number of classes
        input_channels: the number of input channels
        output_channels: the number of output channels
        plot_gif_bool: if True, the function will plot a gif with the generated images for each class
        verbose: if True, the function will use the tqdm during the training and the validation

    Output:
        None
    '''
    image_size = args.image_size
    dataset_path = args.dataset_path
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    noise_schedule = args.noise_schedule
    save_every = args.save_every
    snapshot_name = args.snapshot_name
    snapshot_folder_path = args.snapshot_folder_path
    model_name = args.model_name
    noise_steps = args.noise_steps
    device = args.device
    patience = args.patience
    num_classes = args.num_classes
    input_channels, output_channels = args.inp_out_channels, args.inp_out_channels
    plot_gif_bool = args.plot_gif_bool

    os.makedirs(snapshot_folder_path, exist_ok=True)
    os.makedirs(os.path.join(os.curdir, 'models_run', model_name, 'results'), exist_ok=True)

    # transform = transforms.Compose([
    # transforms.Resize((image_size, image_size)),
    # transforms.ToTensor(),
    # ])

    # train_path = f'{dataset_path}/train'
    # valid_path = f'{dataset_path}/val'

    # train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
    # valid_dataset = datasets.ImageFolder(root=valid_path, transform=transform)
    # train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)


    def get_dataloaders_MNIST(batch_size):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

        train_dataset = datasets.MNIST(root='./MNIST_data', train=True, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        val_dataset = datasets.MNIST(root='./MNIST_data', train=False, download=True, transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        return train_loader, val_loader
    
    train_loader, val_loader = get_dataloaders_MNIST(batch_size)
    width = next(iter(train_loader))[0].size()[-1]

    model = SimpleUNet(width, num_classes, input_channels, output_channels, device).to(device)
    print("Num params: ", sum(p.numel() for p in model.parameters()))

    snapshot_path = os.path.join(snapshot_folder_path, snapshot_name)

    diffusion = Diffusion(
        noise_schedule=noise_schedule, model=model,
        snapshot_path=snapshot_path,
        noise_steps=noise_steps, beta_start=1e-4, beta_end=0.02,
        device=device,image_size=image_size, model_name=model_name)

    # Training 
    diffusion.train(
        lr=lr, epochs=epochs, save_every=save_every,
        train_loader=train_loader, val_loader=val_loader,
        patience=patience,
        verbose=True)

    # Sampling
    fig, axs = plt.subplots(10,5, figsize=(15,10))
    for i in range(10):
        prediction = diffusion.sample(5, target_class=torch.tensor([i], dtype=torch.int64).to(device), input_channels=input_channels, plot_gif_bool=plot_gif_bool) 
        for j in range(5):
            axs[i,j].imshow(prediction[j][0].cpu().numpy(), cmap='gray')
            axs[i,j].axis('off')
    fig.savefig(os.path.join(os.curdir, 'models_run', model_name, 'results', f'predictions_final.png'))
    plt.show()

if __name__ == '__main__':
    import argparse     
    parser = argparse.ArgumentParser(description=' ')
    parser.add_argument('--epochs', type=int, default=501)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_size', type=int)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--save_every', type=int, default=20)
    parser.add_argument('--noise_schedule', type=str, default='cosine')
    parser.add_argument('--snapshot_name', type=str, default='snapshot.pt')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--noise_steps', type=int, default=200)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument('--num_classes', type=int)
    parser.add_argument('--inp_out_channels', type=int, default=3) # input channels must be the same of the output channels
    parser.add_argument('--plot_gif_bool', type=bool, default=False)
    
    args = parser.parse_args()
    args.snapshot_folder_path = os.path.join(os.curdir, 'models_run', args.model_name, 'weights')
    launch(args)