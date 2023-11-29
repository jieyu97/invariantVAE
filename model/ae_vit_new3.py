import torch
from einops import rearrange
from torch import nn, Tensor
from torch.nn import functional as F
import math
from functools import partial
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block as transformer_block
from timm.models.registry import register_model

from utils import device

import logging
from typing import Callable, List, Optional, Sequence, Tuple, Union



class Transformer(nn.Module):
    def __init__(self, base_dim, depth, heads, mlp_ratio):
        super(Transformer, self).__init__()
        
        embed_dim = base_dim * heads

        self.blocks = nn.ModuleList([
            transformer_block(
                dim=embed_dim,
                num_heads=heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6)
            )
            for i in range(depth)])

    def forward(self, x):
    
        h, w = x.shape[2:4]
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        for blk in self.blocks:
            x = blk(x)

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        return x

        
class AEens(nn.Module):

    def __init__(self,
                 latent_dim,
                 mlp_ratio=4):
        super(AEens, self).__init__()
        
        self.latent_dim = latent_dim
        self.mlp_ratio = mlp_ratio
        
        image_size = 81
        patch_size = 9
        stride = 9
        padding = 0
        
        n_nodes = 1024
        base_dims = [81, 20, 4] #[36, 36*16, 36*16*16] #base_dims
        depth = [2, 3, 4] #depth
        heads = [2, 4, 8] #heads
        
        self.embed_dim = base_dims[-1] * heads[-1] #32
        
        self.norm = nn.LayerNorm([base_dims[-1]*heads[-1], 9, 9], eps=1e-6)
                
        self.flat = nn.Flatten()
        self.e_linear1 = nn.Linear(self.embed_dim*9*9, n_nodes)
        self.e_linear2 = nn.Linear(n_nodes, latent_dim)
        self.d_linear1 = nn.Linear(latent_dim, n_nodes)
        self.d_linear2 = nn.Linear(n_nodes, self.embed_dim*9*9)
        self.unflat = nn.Unflatten(1, (self.embed_dim, 9, 9))
        
        # Encoder layers
        # (, 1, 81, 81) --> (, 81*2, 9, 9)
        self.patch_embed = nn.Conv2d(1, base_dims[0] * heads[0], 
                                     kernel_size=9, stride=9, 
                                     padding=0, bias=True)
        self.pos_embed = nn.Parameter(
            torch.randn(1, base_dims[0] * heads[0], 9, 9),
            requires_grad=True)
        
        self.transformer_e0 = Transformer(base_dims[0], depth[0], heads[0], mlp_ratio) # (, 2*81, 9, 9)
        self.transformer_e1 = Transformer(base_dims[1], depth[1], heads[1], mlp_ratio) # (, 4*20, 9, 9)
        self.transformer_e2 = Transformer(base_dims[2], depth[2], heads[2], mlp_ratio) # (, 8*4, 9, 9)
        
        self.block_resize_e1 = Transformer_resize(dim_in=base_dims[0] * heads[0], 
                                                  dim_out=base_dims[1] * heads[1],
                                                  num_heads=heads[0], 
                                                  mlp_ratio=mlp_ratio)
                                                  
        self.block_resize_e2 = Transformer_resize(dim_in=base_dims[1] * heads[1], 
                                                  dim_out=base_dims[2] * heads[2],
                                                  num_heads=heads[1], 
                                                  mlp_ratio=mlp_ratio)

        # Decoder layers
        #  (, 81*2, 26, 26) --> (, 1, 81, 81)
        self.patch_decode = nn.ConvTranspose2d(base_dims[0] * heads[0], 1,
                                     kernel_size=9, stride=9, 
                                     padding=0, bias=True)
        
        self.transformer_d2 = Transformer(base_dims[2], depth[2], heads[2], mlp_ratio) # (, 8*4, 9, 9)
        self.transformer_d1 = Transformer(base_dims[1], depth[1], heads[1], mlp_ratio) # (, 4*20, 9, 9)
        self.transformer_d0 = Transformer(base_dims[0], depth[0], heads[0], mlp_ratio) # (, 2*81, 9, 9)
        
        self.block_resize_d2 = Transformer_resize(dim_in=base_dims[2] * heads[2], 
                                                  dim_out=base_dims[1] * heads[1], 
                                                  num_heads=heads[2], 
                                                  mlp_ratio=mlp_ratio)
                                                  
        self.block_resize_d1 = Transformer_resize(dim_in=base_dims[1] * heads[1], 
                                                  dim_out=base_dims[0] * heads[0], 
                                                  num_heads=heads[1], 
                                                  mlp_ratio=mlp_ratio)


    def encoder(self, input: Tensor, **kwargs):
        x = input
        
        x = self.patch_embed(x) # (, 1, 81, 81) --> (, 81*2, 9, 9)
        pos_embed = self.pos_embed # (1, 81*2, 9, 9)
        x = x + pos_embed # (, 81*2, 9, 9)
        
        x = self.transformer_e0(x) # (, 81*2, 9, 9)
        
        x = self.block_resize_e1(x) # (, 81*2, 9, 9) --> (, 20*4, 9, 9)
        
        x = self.transformer_e1(x) # (, 20*4, 9, 9)
        
        x = self.block_resize_e2(x) # (, 20*4, 9, 9) --> (, 4*8, 9, 9)
        
        x = self.transformer_e2(x) # (, 4*8, 9, 9)
        
        x = self.flat(x)
        # low-dimensional latent representation:
        x = self.e_linear1(x)
        z = self.e_linear2(x)
        
        return z


    def decoder(self, input: Tensor, **kwargs): 
        z = input
        
        y = self.d_linear1(z)
        y = self.d_linear2(y)
        y = self.unflat(y)
        y = self.norm(y)
        
        y = self.transformer_d2(y) # (, 4*8, 9, 9)
        
        y = self.block_resize_d2(y) # (, 4*8, 9, 9) --> (, 20*4, 9, 9)
        
        y = self.transformer_d1(y) # (, 20*4, 9, 9)
        
        y = self.block_resize_d1(y) # (, 20*4, 9, 9) --> (, 81*2, 9, 9)
        
        y = self.transformer_d0(y) # (, 81*2, 9, 9)
        
        y = self.patch_decode(y) # (, 81*2, 9, 9) --> (, 1, 81, 81)
        
        return y
        
        
    def forward(self, input: Tensor, **kwargs):
        # input: (bs*50, 1, 81, 81), output: (bs*50, 1, 81, 81)
        target = input
        x = input       
        z = self.encoder(x)
        latent = z
        y = self.decoder(z)
        recons = y

        return [recons, target, latent]
        
        
    def generate(self, input: Tensor, **kwargs):
        # input: (bs*50, n_latent), output: (bs*50, 1, 81, 81)
        latent = input
        new_samples = self.decoder(latent)

        return new_samples
        
        
    def loss_function(self, *args, **kwargs): 
        # input & recons: (bs*50, 1, 81, 81)
        recons = args[0]
        target = args[1]
        
        # L1 loss as reconstruction loss / mae
        mae = F.l1_loss(recons, target)
        
        # RMSE
        mse = F.mse_loss(recons, target)
        rmse = torch.sqrt(mse)
            
        return {'loss': mae, 'MAE': mae, 'RMSE': rmse}
        


class Transformer_resize(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, mlp_ratio):
        super(Transformer_resize, self).__init__()
        
        self.block = Block_resize(dim_in=dim_in, dim_out=dim_out, 
                                  num_heads=num_heads, 
                                  mlp_ratio=mlp_ratio, 
                                  qkv_bias=True,
                                  norm_layer=partial(nn.LayerNorm, eps=1e-6) )

    def forward(self, x):
    
        h, w = x.shape[2:4]
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        x = self.block(x)

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        return x
        

class Block_resize(nn.Module):

    def __init__(
            self,
            dim_in,
            dim_out,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim_in)
        self.attn = Attention_resize(
            dim_in,
            #dim_out,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim_in, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim_in)
        self.mlp = Mlp_resize(
            in_features=dim_in,
            out_features=dim_out,
            hidden_features=int(dim_in * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim_out, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
        
        
class Attention_resize(nn.Module):

    def __init__(
            self,
            dim,
            dim_out=None,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        #self.final = nn.Linear(dim, dim_out)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x) # (B, N, dim) dim=C*n_heads
        #x = self.final(x) # (B, N, dim_out)
        return x

        
class Mlp_resize(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            out_features=None,
            hidden_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
        
        
class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma



from itertools import repeat
import collections.abc

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)