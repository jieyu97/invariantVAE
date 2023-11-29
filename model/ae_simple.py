import torch
from torch import nn, Tensor
from torch.nn import functional as F
import math
from utils import device

class AEens(nn.Module):

    def __init__(self,
                 latent_dim,
                 n_nodes1=1024,
                 n_nodes2=1024,
                 n_layers=2,
                 activation='none',
                 num_var=1):
        super(AEens, self).__init__()
        
        #num_var = 2
        self.num_var = num_var
        
        self.latent_dim = latent_dim
        self.n_nodes1 = n_nodes1
        self.n_layers = n_layers
        self.act = activation
        
        if self.n_layers == 2:
            n_nodes2 = n_nodes1
        elif self.n_layers == 1:
            n_nodes2 = num_var*81*81
            
        self.n_nodes2 = n_nodes2
        
        self.flat = nn.Flatten()
        
        self.e_linear1 = nn.Linear(num_var*81*81, n_nodes1)
        self.e_linear0 = nn.Linear(n_nodes1, n_nodes2)
        self.e_linear2 = nn.Linear(n_nodes2, latent_dim)
        
        self.d_linear1 = nn.Linear(latent_dim, n_nodes2)
        self.d_linear0 = nn.Linear(n_nodes2, n_nodes1)
        self.d_linear2 = nn.Linear(n_nodes1, num_var*81*81)
        
        self.unflat = nn.Unflatten(1, (num_var, 81, 81))
        
        if activation == 'none':
            self.act = nn.Identity()
        elif activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'leakyrelu':
            self.act = nn.LeakyReLU(1e-3)
        elif activation == 'gelu':
            self.act = nn.GELU()


    def encoder(self, input: Tensor, **kwargs):
        x = input
        
        x = self.flat(x)
        
        if self.n_layers == 1:
            z = self.e_linear2(x)
        elif self.n_layers == 2:
            x = self.e_linear1(x)
            x = self.act(x)
            z = self.e_linear2(x)
        elif self.n_layers == 3:
            x = self.e_linear1(x)
            x = self.act(x)
            x = self.e_linear0(x)
            x = self.act(x)
            z = self.e_linear2(x)
        
        return z


    def decoder(self, input: Tensor, **kwargs): 
        z = input
                
        if self.n_layers == 1:
            y = self.d_linear1(z)
        elif self.n_layers == 2:
            y = self.d_linear1(z)
            y = self.act(y)
            y = self.d_linear2(y)
        elif self.n_layers == 3:
            y = self.d_linear1(z)
            y = self.act(y)
            y = self.d_linear0(y)
            y = self.act(y)
            y = self.d_linear2(y)
        
        y = self.unflat(y)
        
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
        # input: (bs*50, 32), output: (bs*50, 1, 81, 81)
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
        
