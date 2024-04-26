import torch.nn as nn
import torch
import math
import torch.nn.functional as F




def scaled_dot_product(q, k, v, mask=None):
    # q, k, v = 30 x 8 x 200 x 64
    d_k = q.size()[-1] # 64
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k) # 30 x 8 x 200 x 200
    if mask is not None:
        scaled = scaled.permute(1, 0, 2, 3) + mask
        scaled = scaled.permute(1, 0, 2, 3)
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v) # 30 x 8 x 200 x 64
    return values, attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model # 512
        self.num_heads = num_heads # 8
        self.head_dim = d_model // num_heads # 64
        self.qkv_layer = nn.Linear(d_model , 3 * d_model) # (512, 1536)
        self.linear_layer = nn.Linear(d_model, d_model) # (512, 512)
    
    def forward(self, x, mask):
        batch_size, sequence_length, d_model  = x.size() # 30 x 200 x 512
        qkv = self.qkv_layer(x) # 30 x 200 x 1536
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim) # 30 x 200 x 8 x 192
        qkv = qkv.permute(0, 2, 1, 3) # 30 x 8 x 200 x 192
        q, k, v = qkv.chunk(3, dim=-1) # each 30 x 8 x 200 x 64
        values, attention = scaled_dot_product(q, k, v, mask) ## values = 30 x 8 x 200 x 64  ## attention = 30 x 8 x 200 x 200
        values = values.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, self.num_heads * self.head_dim) # 30 x 200 x 512
        out = self.linear_layer(values)
        return out
    
# from einops import rearrange, repeat
# x = rearrange(x, 'b c h w -> b (h w) c')
# x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)