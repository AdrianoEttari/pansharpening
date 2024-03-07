import torch.nn as nn
import math
import torch
import torch.nn.functional as F


# https://medium.com/mlearning-ai/self-attention-in-convolutional-neural-networks-172d947afc00
# https://github.com/fastai/fastai2/blob/master/fastai2/layers.py
# https://gist.github.com/shuuchen/6d39225b018d30ccc86ef10a4042b1aa
# https://github.com/labmlai/annotated_deep_learning_paper_implementations/tree/05321d644e4fed67d8b2856adc2f8585e79dfbee/labml_nn/diffusion/ddpm
# https://www.kaggle.com/code/truthisneverlinear/attention-u-net-pytorch
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
        self.relu = nn.ReLU(inplace=False) # inplace=False MEANS THAT IT WILL MODIFY THE INPUT DIRECTLY, WITHOUT ASSIGNING IT TO A NEW VARIABLE (THIS SAVES SPACE IN MEMORY, BUT IT MODIFIES THE INPUT)
        self.conv1 = nn.Sequential(
                                  nn.Conv2d(in_ch, out_ch,
                                            kernel_size=3, stride=1,
                                            padding=1, bias=True,
                                            device=device),
                                  self.batch_norm1,
                                  self.relu)
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
        return torch.nn.Sequential(
            torch.nn.Linear(dim_in, dim_out, device=device),
            torch.nn.SiLU(),
            torch.nn.Linear(dim_out, dim_out, device=device)
        )
    
    def forward(self, x, t):
        # print(f"DOWN\n x_shape: {x.shape}")
        # print("[batch_size, channels, image_size[0], image_size[1]]")
        # FIRST CONV
        h = self.conv1(x)
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

class SimpleUNet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self, starting_width, num_classes=10, image_channels=3, out_dim=3, device='cuda'):
        super().__init__()
        self.image_channels = image_channels
        self.down_channels = (16,32,64,128) # Note that there are 3 downsampling layers and 3 upsampling layers.
        # To understand why len(self.down_channels)=4, you have to imagine that the first layer 
        # has a Conv2D(16,32), the second layer has a Conv2D(32,64) and the third layer has a Conv2D(64,128).
        self.up_channels = (128,64,32,16)
        self.out_dim = out_dim 
        self.time_emb_dim = 100 # Refers to the number of dimensions or features used to represent time.
        self.device = device
        self.num_classes = num_classes
        # It's important to note that the dimensionality of time embeddings should be chosen carefully,
        # considering the trade-off between model complexity and the amount of available data.

        self.kernel_size_down_list = self.kernel_size_down_list_maker(starting_width, self.down_channels)
        self.kernel_size_up_list = self.kernel_size_down_list[::-1]

        # INITIAL PROJECTION
        # print(f"Initial\nin_channels: {self.image_channels}, out_channels: {self.down_channels[0]}, kernel_size: 3")
        self.conv0 = nn.Conv2d(self.image_channels, self.down_channels[0], 3, padding=1) # SINCE THERE IS PADDING 1, THE OUTPUT IS THE SAME SIZE OF THE INPUT

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

        # LABEL EMBEDDING
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(self.num_classes, self.time_emb_dim).to(device=self.device)

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
        This function returns the space dimensions of the output of the convolutional layer (it's
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
    
    def forward(self, x, timestep, y=None):
        t = timestep.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_emb_dim, device=self.device)

        if y is not None:
            t += self.label_emb(y)
        # INITIAL CONVOLUTION
        x = self.conv0(x)

        # UNet
        residual_inputs = []

        for down in self.downs:
            # Downsample
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            # I need to start from the last one and moving to the first one; that's why we use .pop()
            residual_x = residual_inputs.pop()
            x = up(x, residual_x, t)
        return self.output(x)



if __name__=="__main__":
    model = SimpleUNet(224, device='cpu')
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

    # from torchviz import make_dot

    # input_size = 224
    # x = torch.rand((1, 3, input_size, input_size))
    # t = torch.tensor([5])
    # y = model(x,t)
    # dot = make_dot(y, params=dict(model.named_parameters()))
    # dot.render("model_graph")


