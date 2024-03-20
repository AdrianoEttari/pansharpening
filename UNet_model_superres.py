import torch.nn as nn
import math
import torch
import torch.nn.functional as F


# https://medium.com/mlearning-ai/self-attention-in-convolutional-neural-networks-172d947afc00
# https://github.com/fastai/fastai2/blob/master/fastai2/layers.py
# https://gist.github.com/shuuchen/6d39225b018d30ccc86ef10a4042b1aa
# https://github.com/labmlai/annotated_deep_learning_paper_implementations/tree/05321d644e4fed67d8b2856adc2f8585e79dfbee/labml_nn/diffusion/ddpm
# https://www.kaggle.com/code/truthisneverlinear/attention-u-net-pytorch

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta    # set the beta parameter for the exponential moving average
        self.step = 0       # step counter (initialized at 0) to track when to start updating the moving average

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()): #iterate over all parameters in the current and moving average models
            # get the old and new weights for the current and moving average models
            old_weight, up_weight = ma_params.data, current_params.data
            # update the moving average model parameter
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        # if there is no old weight, return the new weight
        if old is None:
            return new
        # compute the weighted average of the old and new weights using the beta parameter
        return old * self.beta + (1 - self.beta) * new # beta is usually around 0.99
        # therefore the new weights influence the ma parameters only a little bit
        # (which prevents outliers to have a big effect) whereas the old weights
        # are more important.

    def step_ema(self, ema_model, model, step_start_ema=2000):
        '''
        We'll let the EMA update start just after a certain number of iterations
        (step_start_ema) to give the main model a quick warmup. During the warmup
        we'll just reset the EMA parameters to the main model one.
        After the warmup we'll then always update the weights by iterating over all
        parameters and apply the update_average function.
        '''
        # if we are still in the warmup phase, reset the moving average model to the current model
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        # otherwise update the moving average model parameters using the current model parameters
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        # reset the parameters of the moving average model to the current model parameters
        ema_model.load_state_dict(model.state_dict()) # we set the weights of ema_model
        # to the ones of model.

class AttentionBlock(nn.Module):
    def __init__(self, f_g, f_l, f_int, device):
        '''
        AttentionBlock: Applies an attention mechanism to the input data.
        
        Args:
            f_g (int): Number of channels in the 'g' input (image on the up path).
            f_l (int): Number of channels in the 'l' input (residual image).
            f_int (int): Number of channels in the intermediate layer.
            device: Device where the operations should be performed.
        '''
        super().__init__()
        self.w_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=True).to(device),
            nn.BatchNorm2d(f_int).to(device)
        ) # Computes a 1x1 convolution of the 'g' input to reduce its channel dimension to f_int and applies batch normalization.
        
        self.w_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=True).to(device),
            nn.BatchNorm2d(f_int).to(device)
        ) # Computes a 1x1 convolution of the 'x' input to reduce its channel dimension to f_int and applies batch normalization.
        
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True).to(device),
            nn.BatchNorm2d(1).to(device),
            nn.Sigmoid()
        ) # Computes a 1x1 convolution of the element-wise sum of the processed 'g' and 'x' inputs, followed by batch normalization and a sigmoid activation.
        
        self.relu = nn.ReLU(inplace=False)

    def forward(self, g, x):
        '''
        Forward pass for the AttentionBlock.

        Args:
            g (torch.Tensor): The 'g' input (image on the up path).
            x (torch.Tensor): The 'x' input (residual image).

        Returns:
            torch.Tensor: The output of the attention mechanism applied to the input data.
        '''
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return psi * x
    
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, device, kernel_size_down=None):
        super().__init__()
        self.time_mlp =  self._make_te(time_emb_dim, out_ch, device=device)
        self.batch_norm1 = nn.BatchNorm2d(out_ch, device=device)
        self.batch_norm2 = nn.BatchNorm2d(out_ch, device=device)
        self.relu = nn.ReLU(inplace=False) # inplace=True MEANS THAT IT WILL MODIFY THE INPUT DIRECTLY, WITHOUT ASSIGNING IT TO A NEW VARIABLE (THIS SAVES SPACE IN MEMORY, BUT IT MODIFIES THE INPUT)
        self.conv1 = nn.Sequential(
                                  nn.Conv2d(in_ch, out_ch,
                                            kernel_size=3, stride=1,
                                            padding=1, bias=True,
                                            device=device),
                                  self.batch_norm1,
                                  self.relu)
        self.conv_upsampled_lr_img = nn.Conv2d(in_ch, out_ch, 3, padding=1) ############# TO CHECK
        self.conv2 = nn.Sequential(
                                  nn.Conv2d(out_ch, out_ch,
                                            kernel_size=3, stride=1,
                                            padding=1, bias=True,
                                            device=device),
                                  self.batch_norm2,
                                  self.relu)
        self.transform = nn.Sequential(
                                  nn.Conv2d(out_ch, out_ch,
                                            kernel_size=kernel_size_down, stride=2,
                                            padding=1, bias=True,
                                            device=device))
        
        
    def _make_te(self, dim_in, dim_out, device):
        '''
        This function creates a time embedding layer.
        '''
        return torch.nn.Sequential(
            torch.nn.Linear(dim_in, dim_out, device=device),
            torch.nn.SiLU(),
            torch.nn.Linear(dim_out, dim_out, device=device)
        )
    
    def forward(self, x, t, x_skip):
        # print(f"DOWN\n x_shape: {x.shape}")
        # print("[batch_size, channels, image_size[0], image_size[1]]")
        # FIRST CONV
        h = self.conv1(x)
        # SUM THE X-SKIP IMAGE (x+upsampled_lr_img) WITH THE INPUT IMAGE\
        if x_skip is not None:
            x_skip = self.conv_upsampled_lr_img(x_skip)
            h = h + x_skip
        # print(f"conv1_shape: {h.shape}")
        # TIME EMBEDDING
        time_emb = self.relu(self.time_mlp(t))
        # EXTEND LAST 2 DIMENSIONS
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # ADD TIME CHANNEL
        h = h + time_emb
        # print(f"add time: {h.shape}")
        # SECOND CONV
        h = self.conv2(h)
        # print(f"conv2_shape: {h.shape}")
        output = self.transform(h)
        # print(f"out_shape: {output.shape}")
        return output

class UpConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, device, kernel_size_up=None):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch*2, device=device)
        self.batch_norm1_x = nn.BatchNorm2d(out_ch, device=device)
        self.batch_norm1_residual_x = nn.BatchNorm2d(out_ch, device=device)
        self.batch_norm2 = nn.BatchNorm2d(out_ch, device=device)
        self.relu = nn.ReLU(inplace=False)
        self.conv1_x = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True, device=device)
        self.conv1_residual_x = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True, device=device)
        self.conv2 = nn.Conv2d(2*out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True, device=device)
        self.transform = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=kernel_size_up, stride=2, padding=1, bias=True, device=device)
        self.attention = AttentionBlock(f_g=out_ch, f_l=out_ch, f_int=out_ch, device=device)

    def _make_te(self, dim_in, dim_out, device):
        return torch.nn.Sequential(
            torch.nn.Linear(dim_in, dim_out, device=device),
            torch.nn.SiLU(),
            torch.nn.Linear(dim_out, dim_out, device=device)
        )
    
    def forward(self, x, residual_x, t):
        # FIRST CONVOLUTION FOR X
        h = self.relu(self.batch_norm1_x(self.conv1_x(x)))
        # FIRST CONVOLUTION FOR RESIDUAL_X
        residual_x = self.relu(self.batch_norm1_residual_x(self.conv1_residual_x(residual_x)))
        # ATTENTION BLOCK
        residual_x = self.attention(h, residual_x)
        # ADD residual_x AS ADDITIONAL CHANNELS
        h = torch.cat((residual_x, h), dim=1)
        # TIME EMBEDDING
        time_emb = self.relu(self.time_mlp(t))
        # EXTEND THE LAST 2 DIMENSIONS
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # ADD TIME CHANNEL
        h = h + time_emb
        # SECOND CONV
        h = self.relu(self.batch_norm2(self.conv2(h)))
        output = self.transform(h)
        return output

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return out

class RRDB(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=3):
        super(RRDB, self).__init__()
        self.blocks = nn.Sequential(
            *[ResidualBlock(in_channels, in_channels) for _ in range(num_blocks)]
        )
        self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.blocks(x)
        out = self.conv_out(out)
        out += x
        return out
    
class SimpleUNet_superres(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self, starting_width, image_channels=3, out_dim=3, device='cuda'):
        super().__init__()
        self.image_channels = image_channels
        self.down_channels = (32,64,128,256) # Note that there are 3 downsampling layers and 3 upsampling layers.
        # To understand why len(self.down_channels)=4, you have to imagine that the first layer 
        # has a Conv2D(16,32), the second layer has a Conv2D(32,64) and the third layer has a Conv2D(64,128).
        self.up_channels = (256,128,64,32)
        self.out_dim = out_dim 
        self.time_emb_dim = 100 # Refers to the number of dimensions or features used to represent time.
        self.device = device
        # It's important to note that the dimensionality of time embeddings should be chosen carefully,
        # considering the trade-off between model complexity and the amount of available data.

        self.kernel_size_down_list = self.kernel_size_down_list_maker(starting_width, self.down_channels)
        self.kernel_size_up_list = self.kernel_size_down_list[::-1]
        

        # INITIAL PROJECTION
        # print(f"Initial\nin_channels: {self.image_channels}, out_channels: {self.down_channels[0]}, kernel_size: 3")
        self.conv0 = nn.Conv2d(self.image_channels, self.down_channels[0], 3, padding=1) # SINCE THERE IS PADDING 1 AND STRIDE 1,  THE OUTPUT IS THE SAME SIZE OF THE INPUT

        self.LR_encoder = RRDB(in_channels=image_channels, out_channels=image_channels, num_blocks=3)

        self.conv_upsampled_lr_img = nn.Conv2d(self.image_channels, self.down_channels[0], 3, padding=1)

        # DOWNSAMPLE
        self.downs = nn.ModuleList([
            ConvBlock(in_ch=self.down_channels[i],
                      out_ch=self.down_channels[i+1],
                      time_emb_dim=self.time_emb_dim,
                      device=self.device,
                      kernel_size_down=self.kernel_size_down_list[i]) \
            for i in range(len(self.down_channels)-1)])
        
        # UPSAMPLE
        self.ups = nn.ModuleList([
            UpConvBlock(in_ch=self.up_channels[i],
                        out_ch=self.up_channels[i+1],
                        time_emb_dim=self.time_emb_dim,
                        device=self.device,
                        kernel_size_up=self.kernel_size_up_list[i]) \
            for i in range(len(self.up_channels)-1)])

        # OUTPUT
        self.output = nn.Conv2d(self.up_channels[-1], self.out_dim, 1)

    # TIME EMBEDDING
    def pos_encoding(self, t, channels, device):
        inv_freq = 1.0 / (
            10000**(torch.arange(0, channels, 2, device= device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def convolution_output_size(self, input_size, stride, padding, kernel_size):
        '''
        This function returns the spatial dimensions of the output of the convolutional layer (it's
        the same for the width and the height). It depends by the space dimensions of the input (input_size),
        the stride, the padding and the kernel_size.
        '''
        return (input_size+2*padding-kernel_size)/stride + 1

    def kernel_size_down_list_maker(self, starting_width: int, down_channels: tuple):
        '''
        This function returns a list of kernel sizes for the downsampling part of the UNet.
        Basically, starting from the starting_width which is the space dimension of the input, it
        assigns a kernel size of 4 if the space dimension is even, otherwise it assigns a kernel size of 3.
        The upsampling part of the UNet will have the same kernel sizes but in reverse order.
        '''
        kernel_size_down_list = []
        output_size = starting_width
        for _ in range(len(down_channels)-1):
            if output_size % 2 == 0:
                output_size = self.convolution_output_size(output_size, stride=2, padding=1, kernel_size=4)
                kernel_size_down_list.append(4)
            else:
                output_size = self.convolution_output_size(output_size, stride=2, padding=1, kernel_size=3)
                kernel_size_down_list.append(3)

        return kernel_size_down_list
    
    def forward(self, x, timestep, lr_img, magnification_factor):
        t = timestep.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_emb_dim, device=self.device)

        # INITIAL CONVOLUTION
        x = self.conv0(x)

        # LR ENCODER
        lr_img = self.LR_encoder(lr_img)

        
        # UPSAMPLE LR IMAGE
        try:
            upsampled_lr_img = F.interpolate(lr_img, scale_factor=magnification_factor, mode='bicubic')
        except:
            upsampled_lr_img = F.interpolate(lr_img.to('cpu'), scale_factor=magnification_factor, mode='bicubic').to(self.device)

        upsampled_lr_img = self.conv_upsampled_lr_img(upsampled_lr_img) ############# TO CHECK
        
        # SUM THE UP SAMPLED LR IMAGE WITH THE INPUT IMAGE
        
        x = x + upsampled_lr_img
        x_skip = x.clone()
        # UNet
        residual_inputs = []
        
        for i, down in enumerate(self.downs):
            # Downsample
            if i == 0:
                x = down(x, t, x_skip)
            else:
                x = down(x, t, None)
            residual_inputs.append(x)
        for up in self.ups:
            # I need to start from the last one and moving to the first one; that's why we use .pop()
            # print('UP TIMEEE---------------------------------')
            residual_x = residual_inputs.pop() 
            x = up(x, residual_x, t)
        return self.output(x)



if __name__=="__main__":
    model = SimpleUNet_superres(224, device='cpu')
    # print(f'This model has {sum(p.numel() for p in model.parameters())} parameters')

    def print_parameter_count(model):
        total_params = 0
        for name, param in model.named_parameters():
            num_params = param.numel()
            total_params += num_params
            print(f"Layer: {name}, Parameters: {num_params}")
        print("=" * 50)
        print(f"Total Parameters: {total_params}")

    print_parameter_count(model)


