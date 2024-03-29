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
    def __init__(self, f_g, f_x, f_int, device):
        '''
        AttentionBlock: Applies an attention mechanism to the input data.
        
        Args:
            f_g (int): Number of channels in the 'g' input (image on the up path).
            f_x (int): Number of channels in the 'x' input (residual image).
            f_int (int): Number of channels in the intermediate layer.
            device: Device where the operations should be performed.
        '''
        super().__init__()
        self.w_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=True).to(device),
            # nn.BatchNorm2d(f_int).to(device)
        ) # Computes a 1x1 convolution of the 'g' input to reduce its channel dimension to f_int and applies batch normalization.
        
        self.w_x = nn.Sequential(
            nn.Conv2d(f_x, f_int, kernel_size=2, stride=2, padding=0, bias=True).to(device),
            # nn.BatchNorm2d(f_int).to(device)
        ) # Computes a 1x1 convolution of the 'x' input to reduce its channel dimension to f_int and applies batch normalization.

        # self.conv_upsample_g = nn.ConvTranspose2d(in_channels=f_int, out_channels=f_int, kernel_size=3, stride=None, padding=1)

        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True).to(device),
            # nn.BatchNorm2d(1).to(device),
            nn.Sigmoid()
        ) # Computes a 1x1 convolution of the element-wise sum of the processed 'g' and 'x' inputs, followed by batch normalization and a sigmoid activation.
        
        self.relu = nn.ReLU(inplace=False)

        self.result = nn.Sequential(
            nn.Conv2d(f_x, f_x, kernel_size=1, stride=1, padding=0, bias=True).to(device),
            nn.BatchNorm2d(f_x).to(device)
        )
                                                                        
    def forward(self, x, g):
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
        upsample_psi = F.interpolate(psi, scale_factor=2, mode='nearest')
        upsample_psi = upsample_psi.repeat_interleave(repeats=x.shape[1], dim=1)
        result = self.result(upsample_psi * x)
        return result
    
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, device):
        super().__init__()
        self.time_mlp =  self._make_te(time_emb_dim, out_ch, device=device)
        self.batch_norm1 = nn.BatchNorm2d(out_ch, device=device)
        self.batch_norm2 = nn.BatchNorm2d(out_ch, device=device)
        self.relu = nn.ReLU(inplace=False) # inplace=True MEANS THAT IT WILL MODIFY THE INPUT DIRECTLY, WITHOUT ASSIGNING IT TO A NEW VARIABLE (THIS SAVES SPACE IN MEMORY, BUT IT MODIFIES THE INPUT)
        self.conv1 = nn.Sequential(
                                  nn.Conv2d(in_ch, out_ch,
                                            kernel_size=3, stride=1,
                                            padding='same', bias=True,
                                            device=device),
                                  self.batch_norm1,
                                  self.relu)
        self.conv_upsampled_lr_img = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Sequential(
                                  nn.Conv2d(out_ch, out_ch,
                                            kernel_size=3, stride=1,
                                            padding='same', bias=True,
                                            device=device),
                                  self.batch_norm2,
                                  self.relu)
            
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
        # FIRST CONV
        h = self.conv1(x)
        # SUM THE X-SKIP IMAGE (x+upsampled_lr_img) WITH THE INPUT IMAGE
        if x_skip is not None:
            x_skip = self.conv_upsampled_lr_img(x_skip)
            h = h + x_skip
        # TIME EMBEDDING
        time_emb = self.relu(self.time_mlp(t))
        # EXTEND LAST 2 DIMENSIONS
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # ADD TIME CHANNEL
        h = h + time_emb
        # SECOND CONV
        output = self.conv2(h)
        return output

class UpConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, device):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch, device=device)
        self.batch_norm = nn.BatchNorm2d(out_ch, device=device)
        self.relu = nn.ReLU(inplace=False)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding='same', bias=True, device=device)
        self.transform = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=True, output_padding=1, device=device)

    def _make_te(self, dim_in, dim_out, device):
        return torch.nn.Sequential(
            torch.nn.Linear(dim_in, dim_out, device=device),
            torch.nn.SiLU(),
            torch.nn.Linear(dim_out, dim_out, device=device)
        )
    
    def forward(self, x, t):
        # TIME EMBEDDING
        time_emb = self.relu(self.time_mlp(t))
        # EXTEND THE LAST 2 DIMENSIONS
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # ADD TIME CHANNEL
        x = x + time_emb
        # SECOND CONV
        x = self.relu(self.batch_norm(self.conv(x)))
        output = self.transform(x)
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
    
class gating_signal(nn.Module):
    def __init__(self, in_dim, out_dim, device):
        super(gating_signal, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding='same', device=device)
        self.batch_norm = nn.BatchNorm2d(out_dim, device=device)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        return self.relu(x)


class Attention_UNet_superres(nn.Module):
    def __init__(self, image_channels=3, out_dim=3, device='cuda'):
        super().__init__()
        self.image_channels = image_channels
        self.down_channels = (16,32,64,128,256) # Note that there are 4 downsampling layers and 4 upsampling layers.
        # To understand why len(self.down_channels)=5, you have to imagine that the first layer 
        # has a Conv2D(16,32), the second layer has a Conv2D(32,64) and the third layer has a Conv2D(64,128)...
        self.up_channels = (256,128,64,32,16)
        self.out_dim = out_dim 
        self.time_emb_dim = 100 # Refers to the number of dimensions or features used to represent time.
        self.device = device
        # It's important to note that the dimensionality of time embeddings should be chosen carefully,
        # considering the trade-off between model complexity and the amount of available data.

        # INITIAL PROJECTION
        self.conv0 = nn.Conv2d(self.image_channels, self.down_channels[0], 3, padding=1) # SINCE THERE IS PADDING 1 AND STRIDE 1,  THE OUTPUT IS THE SAME SIZE OF THE INPUT

        # LR ENCODER
        self.LR_encoder = RRDB(in_channels=image_channels, out_channels=image_channels, num_blocks=3)

        # UPSAMPLE LR IMAGE
        self.conv_upsampled_lr_img = nn.Conv2d(self.image_channels, self.down_channels[0], 3, padding=1)
        
        # DOWNSAMPLE
        self.conv_blocks = nn.ModuleList([
            ConvBlock(in_ch=self.down_channels[i],
                      out_ch=self.down_channels[i+1],
                      time_emb_dim=self.time_emb_dim,
                      device=self.device) \
            for i in range(len(self.down_channels)-2)])
        
        self.downs = nn.ModuleList([
            nn.Conv2d(self.down_channels[i+1], self.down_channels[i+1], kernel_size=3, stride=2, padding=1, bias=True, device=device)\
        for i in range(len(self.down_channels)-2)])

        # BOTTLENECK
        self.bottle_neck = ConvBlock(in_ch=self.down_channels[-2],
                                        out_ch=self.down_channels[-1],
                                        time_emb_dim=self.time_emb_dim,
                                        device=self.device)
        
        # UPSAMPLE
        self.gating_signals = nn.ModuleList([
            gating_signal(self.up_channels[i], self.up_channels[i+1], self.device) \
            for i in range(len(self.up_channels)-2)])
        
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(self.up_channels[i+1], self.up_channels[i+1], self.up_channels[i+1], self.device) \
            for i in range(len(self.up_channels)-2)])
        
        self.ups = nn.ModuleList([
            UpConvBlock(in_ch=self.up_channels[i], out_ch=self.up_channels[i], time_emb_dim=self.time_emb_dim, device=self.device) \
            for i in range(len(self.up_channels)-2)])
        
        self.up_convs = nn.ModuleList([
            nn.Conv2d(int(self.up_channels[i]*3/2), self.up_channels[i+1], kernel_size=3, stride=1, padding=1, bias=True).to(self.device) \
            for i in range(len(self.up_channels)-2)])

        # OUTPUT
        self.output = nn.Conv2d(self.up_channels[-2], self.out_dim, 1)

    # TIME EMBEDDING
    def pos_encoding(self, t, channels, device):
        inv_freq = 1.0 / (
            10000**(torch.arange(0, channels, 2, device= device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    
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

        upsampled_lr_img = self.conv_upsampled_lr_img(upsampled_lr_img)
        
        # SUM THE UP SAMPLED LR IMAGE WITH THE INPUT IMAGE
        x = x + upsampled_lr_img
        x_skip = x.clone()

        # UNET (DOWNSAMPLE)        
        residual_inputs = []
        for i, (conv_block, down) in enumerate(zip(self.conv_blocks, self.downs)):
            if i == 0:
                x = conv_block(x, t, x_skip)
            else:
                x = conv_block(x, t, None)
            residual_inputs.append(x)
            x = down(x)
        
        # UNET (BOTTLENECK)
        x = self.bottle_neck(x, t, None)

        # UNET (UPSAMPLE)
        for i, (gating_signal, attention_block, up, up_conv) in enumerate(zip(self.gating_signals,self.attention_blocks,self.ups, self.up_convs)):
            gating = gating_signal(x)
            attention = attention_block(residual_inputs[-(i+1)], gating)
            x = up(x, t)
            x = torch.cat([x, attention], dim=1)
            x = up_conv(x)

        return self.output(x)



if __name__=="__main__":
    model = Attention_UNet_superres(224, device='cpu')
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


