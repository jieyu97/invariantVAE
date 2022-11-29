### 2nd simpler version
# remember channels first
import torch
from torch import nn, Tensor


class IvaeEncoder(nn.Module):

    def __init__(self, latent_dim: int):
        super(IvaeEncoder, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        # self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=None, padding=0)
        # self.conv2 = nn.Conv2d(32, 16, (3, 3), stride=1, padding=1)
        # self.maxpool2 = nn.MaxPool2d((3, 3), stride=None, padding=0)
        # self.conv3 = nn.Conv2d(16, 8, (3, 3), stride=1, padding=1)
        # self.conv4 = nn.Conv2d(8, 4, (3, 3), stride=1, padding=1)
        # self.maxpool3 = nn.MaxPool2d((3, 3), stride=None, padding=0)
        # self.activation = nn.ReLU()
        # self.flatten = nn.Flatten()
        # self.linear1 = nn.Linear(in_features=4 * 3 * 3, out_features=2 * latent_dim)
        # self.linear2 = nn.Linear(in_features=2 * latent_dim, out_features=latent_dim)
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=None, padding=0),
            nn.Conv2d(32, 64, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), stride=None, padding=0),
            nn.Conv2d(64, 128, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), stride=None, padding=0),
            nn.Flatten(),
            nn.Linear(in_features=256 * 3 * 3, out_features=2 * latent_dim),
            nn.ReLU(),
            nn.Linear(in_features=2 * latent_dim, out_features=latent_dim),
        )

    #         self.kl = 0

    def forward(self, x: Tensor):
        #         x_split = torch.split(x, 1, dim=1)
        #         pooled_ens = torch.stack([self.encoder_invariant(ens) for ens in x_split], dim=-1)
        #         mean_ens = torch.sum(pooled_ens, dim=-1)
        #         x_latent = self.encoder_shared(mean_ens)
        # x = self.conv1(x)
        # x = self.activation(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = self.activation(x)
        # x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = self.activation(x)
        # x = self.conv4(x)
        # x = self.activation(x)
        # x = self.maxpool3(x)
        # x = self.flatten(x)
        # x = self.linear1(x)
        # x = self.activation(x)
        # # low-dimensional latent representation:
        # z = self.linear2(x)
        # #         # K-L divergence
        # #         self.kl = (torch.exp(0.5*z_log_var)**2 + z_mean**2 - 0.5*z_log_var - 0.5).sum()
        # #         # z = mu + sigma*self.N.sample(mu.shape)
        # #         # self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()

        return self.model(x)


class IvaeDecoder(nn.Module):

    def __init__(self, latent_dim: int):
        super(IvaeDecoder, self).__init__()
        # self.linear = nn.Linear(in_features=latent_dim, out_features=256 * 3 * 3)
        # self.activation1 = nn.ReLU()
        # self.convt1 = nn.ConvTranspose2d(256, 128, (9, 9), stride=(3, 3), padding=3)
        # self.convt2 = nn.ConvTranspose2d(128, 64, (9, 9), stride=(3, 3), padding=3)
        # self.convt3 = nn.ConvTranspose2d(64, 32, (9, 9), stride=(3, 3), padding=3)
        # self.conv = nn.Conv2d(32, 1, (3, 3), stride=1, padding=1)
        # self.activation2 = nn.Sigmoid()

        self.model1 = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=256 * 3 * 3),
            nn.ReLU()
        )
        self.model2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, (3, 3), stride=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, (3, 3), stride=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, (3, 3), stride=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, (1, 1)),
            nn.Sigmoid()
        )

    #         self.N = torch.distributions.Normal(0, 1)
    #         self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
    #         self.N.scale = self.N.scale.cuda()

    def forward(self, z: Tensor):
        # y = self.linear(z)
        # y = self.activation1(y)
        # y = torch.reshape(y, (-1, 256, 3, 3))
        # y = self.convt1(y)
        # y = self.activation1(y)
        # y = self.convt2(y)
        # y = self.activation1(y)
        # y = self.convt3(y)
        # y = self.activation1(y)
        # y = self.conv(y)
        # x_hat = self.activation2(y)
        y = self.model1(z)
        y = torch.reshape(y, (-1, 256, 3, 3))
        return self.model2(y)


class Ivae(nn.Module):
    def __init__(self, encoder, decoder):
        super(Ivae, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    #     def reparameterization(self, mean, var):
    #         epsilon = torch.randn_like(var).to(DEVICE)        # sampling epsilon
    #         z = mean + var*epsilon                          # reparameterization trick
    #         return z

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z