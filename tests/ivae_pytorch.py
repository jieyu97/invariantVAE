# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 09:56:59 2022

@author: gm2154
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import time

f = open('/home/chen_jieyu/convae/model_training.txt','w')

# Get Device for Training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


# read data, weather variable: t2m or 500gh
data_gridded = np.load('/Data/jieyu_data/gridded_ens_t2m_complete.npy')
# shape: (3651, 81, 81, 50)

# move the channel (ensemble members) dimension to the first
data_gridded = np.moveaxis(data_gridded, 3, 1)

# split into training, validation, and test sets (training: 2007-2014, validation: 2015, test: 2016)
train_end = 2920
test_start = 3285

# training data
x_train = data_gridded[:train_end, :, :, :]

mins_train = x_train.reshape(x_train.shape[0], -1).min(axis=1)
maxs_train = x_train.reshape(x_train.shape[0], -1).max(axis=1)

mins_train = np.expand_dims(mins_train, axis=(1, 2, 3))
maxs_train = np.expand_dims(maxs_train, axis=(1, 2, 3))

x_train_normalized = (x_train - mins_train)/(maxs_train - mins_train)

# validation data
x_val = data_gridded[train_end:test_start, :, :, :]

mins_val = x_val.reshape(x_val.shape[0], -1).min(axis=1)
maxs_val = x_val.reshape(x_val.shape[0], -1).max(axis=1)

mins_val = np.expand_dims(mins_val, axis=(1, 2, 3))
maxs_val = np.expand_dims(maxs_val, axis=(1, 2, 3))

x_val_normalized = (x_val - mins_val)/(maxs_val - mins_val)

# test data
x_test = data_gridded[test_start:, :, :, :]

mins_test = x_test.reshape(x_test.shape[0], -1).min(axis=1)
maxs_test = x_test.reshape(x_test.shape[0], -1).max(axis=1)

mins_test = np.expand_dims(mins_test, axis=(1, 2, 3))
maxs_test = np.expand_dims(maxs_test, axis=(1, 2, 3))

x_test_normalized = (x_test - mins_test)/(maxs_test - mins_test)

# model inputs:
train_data = torch.from_numpy(x_train_normalized)
val_data = torch.from_numpy(x_val_normalized)
test_data = torch.from_numpy(x_test_normalized)

train_d = torch.unsqueeze(train_data[:, 0, :, :], dim=1)
val_d = torch.unsqueeze(val_data[:, 0, :, :], dim=1)
test_d = torch.unsqueeze(test_data[:, 0, :, :], dim=1)

train = TensorDataset(train_d)
val = TensorDataset(val_d)
test = TensorDataset(test_d)

# model hyper-parameters:
BATCH_SIZE = 16
LEARNING_RATE = 0.01 #1e-3
EPOCHS = 50


train_loader = DataLoader(train, batch_size=BATCH_SIZE)
val_loader = DataLoader(val, batch_size=BATCH_SIZE)
test_loader = DataLoader(test, batch_size=test_data.shape[0])


start_time = time.time()

### 2nd simpler version
# remember channels first
class ivae_encoder(nn.Module):
    def __init__(self, latent_dim):
        super(ivae_encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.maxpool1  = nn.MaxPool2d(kernel_size=3, stride=None, padding=0)
        self.conv2 = nn.Conv2d(32, 16, (3,3), stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d((3,3), stride=None, padding=0)
        self.conv3 = nn.Conv2d(16, 8, (3,3), stride=1, padding=1)
        self.conv4 = nn.Conv2d(8, 4, (3,3), stride=1, padding=1)
        self.maxpool3 = nn.MaxPool2d((3,3), stride=None, padding=0)
        self.activation = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=4*3*3, out_features=2*latent_dim)
        self.linear2 = nn.Linear(in_features=2*latent_dim, out_features=latent_dim)
        
#         self.kl = 0

    def forward(self, x):
#         x_split = torch.split(x, 1, dim=1)
#         pooled_ens = torch.stack([self.encoder_invariant(ens) for ens in x_split], dim=-1)
#         mean_ens = torch.sum(pooled_ens, dim=-1)
#         x_latent = self.encoder_shared(mean_ens)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = self.conv4(x)
        x = self.activation(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.activation(x)
        # low-dimensional latent representation:
        z = self.linear2(x)
#         # K-L divergence
#         self.kl = (torch.exp(0.5*z_log_var)**2 + z_mean**2 - 0.5*z_log_var - 0.5).sum()
#         # z = mu + sigma*self.N.sample(mu.shape)
#         # self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()

        return z
    

class ivae_decoder(nn.Module):
    def __init__(self, latent_dim):
        super(ivae_decoder, self).__init__()
        self.linear = nn.Linear(in_features=latent_dim, out_features=8*3*3)
        self.activation1 = nn.ReLU()
        self.convt1 = nn.ConvTranspose2d(8, 8, (9,9), stride=(3,3), padding=3)
        self.convt2 = nn.ConvTranspose2d(8, 16, (9,9), stride=(3,3), padding=3)
        self.convt3 = nn.ConvTranspose2d(16, 32, (9,9), stride=(3,3), padding=3)
        self.conv = nn.Conv2d(32, 1, (3,3), stride=1, padding=1)
        self.activation2 = nn.Sigmoid()
        
#         self.N = torch.distributions.Normal(0, 1)
#         self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
#         self.N.scale = self.N.scale.cuda()
        
    def forward(self, z):
        y = self.linear(z)
        y = self.activation1(y)
        y = torch.reshape(y, (-1,8,3,3))
        y = self.convt1(y)
        y = self.activation1(y)
        y = self.convt2(y)
        y = self.activation1(y)
        y = self.convt3(y)
        y = self.activation1(y)
        y = self.conv(y)
        x_hat = self.activation2(y)
        
        return x_hat
    
    
class ivae(nn.Module):
    def __init__(self, encoder, decoder):
        super(ivae, self).__init__()
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
    

N_LATENT = 6
encoder = ivae_encoder(latent_dim = N_LATENT)
decoder = ivae_decoder(latent_dim = N_LATENT)

model = ivae(encoder = encoder, decoder = decoder)
model = model.float()
model.to(device)


######### not using
def loss_cramer_mse(output, target):
    """
    Computes (energy distance + mean square error)

    Parameters
    ----------
    y_true : tf tensor of shape (BATCH_SIZE, 50, 81, 81)
        True values: raw ensemble forecast fields.
    y_pred : tf tensor of shape (BATCH_SIZE, N_SAMPLES, 81, 81)
        N_SAMPLES = 50 by default.
        Predictive samples: reconstructed ensemble forecast fields.

    Returns
    -------
    sum of torch tensor of shape (BATCH_SIZE,)
        Scores.

    """
    
    y_pred = output
    y_true = target
    
    beta = 1
    n_samples_x = y_true.shape[1]
    n_samples_y = y_pred.shape[1]
    
    def expected_dist(diff, beta):
        return torch.sum(torch.pow(torch.sqrt(torch.sum(torch.sum(torch.square(diff), axis=-1), axis=-1)
                                              + 1e-7), beta), axis=1)
    
    energy_1 = 0
    energy_2 = 0
    energy_3 = 0
    
    for i in range(n_samples_y):
        y_pred_i = torch.unsqueeze(y_pred[:, i, :, :], dim=1)
        energy_1 = energy_1 + expected_dist(y_pred_i - y_true, beta)
        
    for i in range(n_samples_x):
        y_true_i = torch.unsqueeze(y_true[:, i, :, :], dim=1)
        energy_2 = energy_2 + expected_dist(y_true_i - y_true, beta)
        
    for i in range(n_samples_y):
        y_pred_i = torch.unsqueeze(y_pred[:, i, :, :], dim=1)
        energy_3 = energy_3 + expected_dist(y_pred_i - y_pred, beta)

    energy_distance = energy_1/(n_samples_x*n_samples_y) - energy_2/(2*n_samples_x**2) - energy_3/(2*n_samples_y**2)
    
    y_true_mean = torch.mean(y_true, axis=1)   # (bs,81,81)
    y_pred_mean = torch.mean(y_pred, axis=1)   # (bs,81,81)
    
    rmse = torch.sqrt(torch.mean(torch.square(y_true_mean - y_pred_mean), axis=(-1,-2)) + 1e-7)
    
    combined_loss = energy_distance + rmse

    loss = torch.mean(combined_loss)
    
    return combined_loss



# Initialize the loss function
#loss_fn = loss_cramer_mse
loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train_loop(dataloader, autoencoder, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0
    
    for batch, X in enumerate(dataloader):
        data = X[0].to(device)
        
        # Backpropagation
        optimizer.zero_grad()
        
        # Compute prediction and loss
        pred, z_para = autoencoder(data.float())
        # loss = ((data.float() - pred)**2).sum() #+ autoencoder.encoder.kl
        loss = loss_fn(pred, data.float())
        
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

        if batch % 40 == 0:
            loss, current = loss.item(), batch * len(X[0])
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]", file=f)
#            print("latent variables:")
#            print(z_para)
            
    train_loss /= num_batches
    print(f"Avg training loss: {train_loss:>8f} \n", file=f)
    
    return train_loss, autoencoder


def val_loop(dataloader, autoencoder, loss_fn):
    num_batches = len(dataloader)
    val_loss = 0

    with torch.no_grad():
        for X in dataloader:
            data = X[0].to(device)
            pred, z_para = autoencoder(data.float())
            val_loss += loss_fn(pred, data.float())

    val_loss /= num_batches
    print(f"Avg validation loss: {val_loss:>8f} \n", file=f)
    
    return val_loss.item()


def test_loop(dataloader, autoencoder, loss_fn):
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for X in dataloader:
            data = X[0].to(device)
            pred, z_para = autoencoder(data.float())
            test_loss += loss_fn(pred, data.float())

    test_loss /= num_batches
    print(f"Avg test loss: {test_loss:>8f} \n", file=f)
    
    return test_loss.item(), pred, z_para


### not using
# early stopping function
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
    
# Training the model
train_loss = []
val_loss = []
test_loss = []

early_stopper = EarlyStopper(patience=3, min_delta=0)

EPOCHS = 5
print("Start training VAE...", file=f)
model.train()

for t in range(EPOCHS):
    print(f"Epoch {t+1}\n-------------------------------", file=f)
    
    epoch_train_loss, ivae = train_loop(train_loader, model, loss_fn, optimizer)
    train_loss.append(epoch_train_loss)
    
    epoch_validate_loss = val_loop(val_loader, ivae, loss_fn)
    val_loss.append(epoch_validate_loss)

#    if early_stopper.early_stop(epoch_validate_loss):
#        print("Early stopping at epoch:", t)
#        break
    
print("Training finished!", file=f)

    
epoch_test_loss, recons_test, z_para = test_loop(test_loader, ivae, loss_fn)
test_loss.append(epoch_test_loss)
print('shape of output reconstructed figures: ', recons_test.shape, file=f)

print('reconstructed test data: ', recons_test, file=f)
print("latent variables of test data: ", z_para, file=f)
#print("z mean")
#print(z_para[0])
#print("z log var")
#print(z_para[1])

recons = recons_test.cpu().numpy()
np.save(arr = recons, file = '/Data/jieyu_data/a8_ivae_torch_recons_t2m.npy')
print("Done!", file=f)

# compute time and loss
elapsed_time = time.time() - start_time
print("Processing time in total: ", elapsed_time/60, file=f)
