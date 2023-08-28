# remember channels first
import torch
from torch import nn, Tensor

        
class IvaeEncoder(nn.Module):

    def __init__(self):
        super(IvaeEncoder, self).__init__()

        self.encoder_invariant = nn.Sequential(
            # Convolutional layer 1
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(1e-3),
            ResBlockBottleneck(32, 32),
            # Convolutional layer 2: reduce size
            nn.Conv2d(32, 64, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(1e-3),
            ResBlockBottleneck(64, 64),
            # Convolutional layer 3: reduce size
            nn.Conv2d(64, 128, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(1e-3),
            ResBlockBottleneck(128, 128) # (128,9,9)
        )
        self.encoder_shared = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(1e-3),
            ResBlockBottleneck(256, 256), 
            nn.Flatten() # (256, 3, 3) 
        )
        self.latent1 = nn.Linear(256*3*3, 32)
        self.latent2 = nn.Linear(256*3*3, 32)


    def forward(self, x: Tensor):
        x_split = torch.split(x, 1, dim=1)
        pooled_ens = torch.stack([self.encoder_invariant(ens) for ens in x_split], dim=-1)
        mean_ens = torch.sum(pooled_ens, dim=-1) / 50
        x_latent = self.encoder_shared(mean_ens)
        
        # low-dimensional latent representation:
        z_mean = self.latent1(x_latent)
        z_log_var = self.latent2(x_latent)
        return [z_mean, z_log_var]
        

class IvaeDecoder(nn.Module):

    def __init__(self):
        super(IvaeDecoder, self).__init__()
        

        self.decoder = nn.Sequential(
            nn.Linear(32, 256*3*3),
            nn.Unflatten(1, (256, 3, 3)),
            ResBlockBottleneck(256, 256),
            # Deconvolutional layer 1
            nn.ConvTranspose2d(256, 128, (3,3), stride=(3,3)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(1e-3),
            ResBlockBottleneck(128, 128),
            # Deconvolutional layer 2
            nn.ConvTranspose2d(128, 64, (3,3), stride=(3,3)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(1e-3),
            ResBlockBottleneck(64, 64),
            # Deconvolutional layer 2
            nn.ConvTranspose2d(64, 32, (3,3), stride=(3,3)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(1e-3),
            ResBlockBottleneck(32, 32),
            # Convolutional layer
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
        
    def forward(self, z: Tensor):
        y = self.decoder(z)
        return y

  
      
class Ivae_split(nn.Module):
    def __init__(self):
        super(Ivae_split, self).__init__()
        self.encoder = IvaeEncoder()
        self.decoder = IvaeDecoder()
        
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()


    def sample(self, z_para):
        z_mean = z_para[0]
        z_log_var = z_para[1]
        z = z_mean + torch.exp(0.5*z_log_var) * self.N.sample(z_mean.shape)
        return self.decoder(z)
        

    def forward(self, x):
        z_para = self.encoder(x)
        x_hat = torch.cat([self.sample(z_para) for n_ens in range(50)], dim=1)
        return x_hat, z_para

