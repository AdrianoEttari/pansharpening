# %%
import torch
import torch.nn as nn
# from PIL import Image
# from torchvision.transforms import ToTensor
# from torchinfo import summary
# %%
class ImageEncoding(nn.Module):
    def __init__(self, input_channels, patch_size = (16,16), stride = None, embedding_dim:int = None):
        super().__init__()

        self.patch_size = patch_size

        if stride is None:
            self.stride = patch_size
        else:
            self.stride = stride

        if embedding_dim is None:
            self.embedding_dim = input_channels * patch_size[0] * patch_size[1] # P^2*C as in the original paper
        else:
            self.embedding_dim = embedding_dim

        self.conv = nn.Conv2d(in_channels = input_channels, out_channels = self.embedding_dim, kernel_size=self.patch_size, stride=self.stride) # (batch_size, num_channel, height, widht) -> (batch_size, embedding_dim, num_patches_y, num_patches_x)
        self.flatten = nn.Flatten(start_dim=2, end_dim=3) # (batch_size, embedding_dim, num_patches_y, num_patches_x) -> (batch_size, embedding_dim, num_patches)
    
    def forward(self, x):
        # x = (batch_size, channel, height, width)
        x = self.conv(x) # (batch_size, embedding_dim, num_patches_y, num_patches_x)
        x = self.flatten(x) # (batch_size, embedding_dim, num_patches)
        # we want (batch_size, num_patches, embedding_dim)
        x = x.transpose(1,2)
        return x
    
class PositionalEncoding(nn.Module):

    def __init__(self, image_size, batch_size, embedding_dim, patch_size=(16,16)) -> None:
        super().__init__()
        # create the token embedding
        self.embedding_dim = embedding_dim
        self.class_token = nn.Parameter(torch.rand(batch_size, 1, self.embedding_dim), requires_grad=True) # x_0

        # create the positional embedding, in the paper it is used a standard learnable 1D position embeddings
        self.num_patch = image_size[0]*image_size[1] // (patch_size[0]*patch_size[1])
        self.position_emb = nn.Parameter(torch.rand(1, self.num_patch+1, self.embedding_dim), requires_grad=True)
        # TODO I want to try also the sin, cos positional embedding of the original transformer


    def forward(self, x):
        x = torch.cat([self.class_token, x], dim=1) # (batch_size, num_patches+1, embedding_dim)
        x = x + self.position_emb
        return x
    
class Embedding(nn.Module):

    def __init__(self, input_channels, image_size, batch_size, patch_size = (16,16), stride = None, embedding_dim=None) -> None:
        super().__init__()

        self.image_embedding = ImageEncoding(input_channels, patch_size, stride, embedding_dim)
        self.embedding_dim = self.image_embedding.embedding_dim  
        self.positional_encoding = PositionalEncoding(image_size, batch_size, self.embedding_dim, patch_size)
        self.num_patches = self.positional_encoding.num_patch

    def forward(self, x):
        x = self.image_embedding(x)
        # x = self.positional_encoding(x)
        return x

class MultiheadAttention(nn.Module):

    def __init__(self, head:int, embedding_dim:int, ) -> None:
        super().__init__()

        self.head = head
        self.embedding_dim = embedding_dim

        assert embedding_dim % head == 0, "embedding_dim must be divisible by head"
        self.head_dim = embedding_dim // head

        self.w_q = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.w_k = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.w_v = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)

        self.w_o = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)

    def attention(self, q, k, v, mask=None):
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention = torch.softmax(attention_scores, dim=-1)

        x = torch.matmul(attention, v)
        return x

    def forward(self, q,k,v ,mask= None):
        # q,k,v = (batch_size, num_patches+1, embedding_dim)
        q1 = self.w_q(q)
        k1 = self.w_k(k)
        v1 = self.w_v(v)

        # -view->((batch_size, num_patches+1, head, head_dim)  -transpose-> (batch_size, head, num_patches+1, head_dim)
        q1 = q1.view(q1.shape[0], q1.shape[1], self.head, self.head_dim).transpose(1,2) 
        k1 = k1.view(k1.shape[0], k1.shape[1], self.head, self.head_dim).transpose(1,2)
        v1 = v1.view(v1.shape[0], v1.shape[1], self.head, self.head_dim).transpose(1,2)

        x = self.attention(q1, k1, v1, mask=mask) # (batch_size, head, num_patches+1, head_dim)
        x = x.transpose(1,2) # (batch_size, num_patches+1, head, head_dim)
        x = x.flatten(start_dim=2, end_dim=3) # (batch_size, num_patches+1, embedding_dim)

        x = self.w_o(x)
        return x

class Visual_MultiHeadAttention(nn.Module):

    def __init__(self, image_size, input_channels, patch_size, batch_size, num_heads, embedding_dim=None):
        super().__init__()

        self.embedding = Embedding(input_channels, image_size, batch_size, patch_size, embedding_dim = embedding_dim)
        self.embedding_dim = self.embedding.embedding_dim
        num_patches = self.embedding.num_patches
        self.self_attention_block = MultiheadAttention(num_heads, self.embedding_dim)
        num_patches = self.embedding.num_patches
        # self.norm1 = nn.LayerNorm(normalized_shape=[num_patches+1, self.self_attention_block.embedding_dim])
        # self.norm2 = nn.LayerNorm(normalized_shape=[num_patches+1, self.self_attention_block.embedding_dim])
        self.norm1 = nn.LayerNorm(normalized_shape=[num_patches, self.self_attention_block.embedding_dim])
        self.norm2 = nn.LayerNorm(normalized_shape=[num_patches, self.self_attention_block.embedding_dim])

    def forward(self,x):
        x = self.embedding(x)
        skip1 = x
        x = self.norm1(x)
        x = self.self_attention_block(x,x,x)
        x = skip1 + x
        x = self.norm2(x)

        return x
# %%
import torch
import torch.nn as nn

# image_size = (28,28)
# input_channels = 1
# patch_size = (7,7)
# batch_size = 64
# num_heads = 7
# mlp_dim = 128
# num_enc_layers = 4 
# n_classes = 10

image_size = (224,224)
input_channels = 3
patch_size = (16,16)
batch_size = 64
num_heads = 4
mlp_dim = 128
num_enc_layers = 4 
n_classes = 10

# model = Vit(image_size, input_channels, patch_size, batch_size, num_heads, mlp_dim, num_enc_layers, n_classes)
model = Visual_MultiHeadAttention(image_size, input_channels, patch_size, batch_size, num_heads)
x = torch.randn((batch_size, input_channels, image_size[0], image_size[1]))
multihead_output = model(x)
# %% 
def calculate_conv_output_height_width(input_shape, kernel_size, stride):
    """
    Calculate the output shape of a convolution operation.
    
    Args:
    - input_shape (tuple): Shape of the input tensor (batch_size, channels, height, width).
    - kernel_size (tuple): Size of the convolutional kernel (height, width).
    - stride (tuple): Stride of the convolution operation (vertical_stride, horizontal_stride).
    
    Returns:
    - output_shape (tuple): Shape of the output tensor after convolution (batch_size, channels, height, width).
    """
    batch_size, channels, height, width = input_shape
    kernel_height, kernel_width = kernel_size
    vertical_stride, horizontal_stride = stride
    
    output_height = ((height - kernel_height) // vertical_stride) + 1
    output_width = ((width - kernel_width) // horizontal_stride) + 1

    return output_height, output_width

h_from_convolution, w_from_convolution = calculate_conv_output_height_width(x.shape, patch_size, patch_size)
multihead_output = multihead_output.reshape(batch_size, input_channels * patch_size[0] * patch_size[1], h_from_convolution, w_from_convolution)
# %%
transposed_conv = nn.ConvTranspose2d(in_channels=768, out_channels=3, kernel_size=patch_size, stride=patch_size)
multihead_output = transposed_conv(multihead_output)

#%%