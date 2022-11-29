# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 09:56:59 2022

@author: gm2154
"""

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import time

f = None #open('./model_training.txt','w')

# Get Device for Training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


def compute_normalization(x_train):
    mins_train = x_train.ravel().min()
    maxs_train = x_train.ravel().max()
    return mins_train, maxs_train


def apply_normalization(x, scales):
    mins_train, maxs_train = scales
    return (x - mins_train) / (maxs_train - mins_train)


def get_random_data(n_members = 1):
    data_gridded = torch.randn(3651, 81, 81, n_members) * torch.exp(torch.randn(3651, 1, 1, n_members) / 10) + torch.randn(3651, 1, 1, n_members)
    data_gridded = data_gridded.data.numpy()
    return data_gridded


def get_ensemble_data():
    return np.load('/Data/jieyu_data/gridded_ens_t2m_complete.npy')


def build_dataset(train_end=2920, test_start=3285):

    # read data, weather variable: t2m or 500gh
    # data_gridded = get_ensemble_data()
    # I commented this out due to lack of data
    data_gridded = get_random_data()

    # shape: (3651, 81, 81, 50)

    # move the channel (ensemble members) dimension to the first
    data_gridded = np.moveaxis(data_gridded, 3, 1)
    # split into training, validation, and test sets (training: 2007-2014, validation: 2015, test: 2016)
    # training data
    x_train = data_gridded[:train_end, :, :, :]
    scales = compute_normalization(x_train)
    x_train_normalized = apply_normalization(x_train, scales)
    # validation data
    x_val = data_gridded[train_end:test_start, :, :, :]
    x_val_normalized = apply_normalization(x_val, scales)
    # test data
    x_test = data_gridded[test_start:, :, :, :]
    x_test_normalized = apply_normalization(x_test, scales)
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
    return train, val, test


### 2nd simpler version
# remember channels first
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


def expected_dist(diff: Tensor, beta: float):
    # 1.e-7 term was removed! Maybe restore if needed for stability
    return torch.mean(torch.pow(torch.norm(diff, p=2, dim=[-1,-2]), beta), dim=1)


def loss_cramer_mse_old(output, target):
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

    energy_distance = energy_1 / (n_samples_y) - energy_2 / (2 * n_samples_x) - energy_3 / (
                2 * n_samples_y)

    y_true_mean = torch.mean(y_true, axis=1)  # (bs,81,81)
    y_pred_mean = torch.mean(y_pred, axis=1)  # (bs,81,81)

    rmse = torch.sqrt(torch.mean(torch.square(y_true_mean - y_pred_mean), axis=(-1, -2)) + 1e-7)

    combined_loss = energy_distance + rmse

    loss = torch.mean(combined_loss)

    return combined_loss


def energy(x, y, beta):
    return torch.mean(expected_dist(x.unsqueeze(1) - y.unsqueeze(2), beta), dim=1)


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
    energy_1 = energy(y_pred, y_true, beta)
    energy_2 = energy(y_true, y_true, beta)
    energy_3 = energy(y_pred, y_pred, beta)
    energy_distance = energy_1 - (energy_2 + energy_3) / 2.
    y_true_mean = torch.mean(y_true, dim=1)  # (bs,81,81)
    y_pred_mean = torch.mean(y_pred, dim=1)  # (bs,81,81)
    rmse = torch.norm(y_true_mean - y_pred_mean, p=2, dim=[-1, -2])
    combined_loss = energy_distance + rmse
    loss = torch.mean(combined_loss)
    return combined_loss


def train_loop(dataloader, autoencoder, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0
    autoencoder.train()
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
    # train_loop does no longer return the model, as there is no need to do that
    return train_loss


def val_loop(dataloader, autoencoder, loss_fn):
    num_batches = len(dataloader)
    val_loss = 0
    autoencoder.eval()
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
    autoencoder.eval()
    with torch.no_grad():
        for X in dataloader:
            data = X[0].to(device)
            pred, z_para = autoencoder(data.float())
            test_loss += loss_fn(pred, data.float())

    test_loss /= num_batches
    print(f"Avg test loss: {test_loss:>8f} \n", file=f)

    return test_loss.item(), pred, z_para, data


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


def main():
    train, val, test = build_dataset()

    # model hyper-parameters:
    BATCH_SIZE = 16
    LEARNING_RATE = 0.0001 #1e-3
    EPOCHS = 20
    N_LATENT = 6


    train_loader = DataLoader(train, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test, batch_size=len(test))

    encoder = IvaeEncoder(latent_dim = N_LATENT)
    decoder = IvaeDecoder(latent_dim = N_LATENT)

    model = Ivae(encoder = encoder, decoder = decoder)
    model = model.float()
    model.to(device)


    # Initialize the loss function
    #loss_fn = loss_cramer_mse
    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training the model
    train_loss = []
    val_loss = []
    test_loss = []

    early_stopper = EarlyStopper(patience=3, min_delta=0)
    print("Start training VAE...", file=f)
    start_time = time.time()

    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------", file=f)

        epoch_train_loss = train_loop(train_loader, model, loss_fn, optimizer)
        train_loss.append(epoch_train_loss)

        epoch_validate_loss = val_loop(val_loader, model, loss_fn)
        val_loss.append(epoch_validate_loss)

    #    if early_stopper.early_stop(epoch_validate_loss):
    #        print("Early stopping at epoch:", t)
    #        break

    print("Training finished!", file=f)

    epoch_test_loss, recons_test, z_para, data_test = test_loop(test_loader, model, loss_fn)
    test_loss.append(epoch_test_loss)
    print('shape of output reconstructed figures: ', recons_test.shape, file=f)

    # The random dataset generates samples, which differ in mean and std, but are statistically homogeneous in space.
    # Expected behavior of the model for the random data is to have spatial_mean(recons_test) == spatial_mean(data_test)
    # whereas expected behavior for the std is spatial_std -> 0 for model predictions,
    # since the model has no reason to learn spatial structure (all the spatial structure is noise)
    print('reconstructed test data (mean): ', recons_test.mean(dim=(2, 3))[:10], file=f)
    print('target data (mean): ', data_test.mean(dim=(2, 3))[:10], file=f)
    print('reconstructed test data (std): ', data_test.std(dim=(2, 3))[:10], file=f)
    print('target data (std): ', recons_test.std(dim=(2, 3))[:10], file=f)

    print("latent variables of test data: ", z_para, file=f)
    #print("z mean")
    #print(z_para[0])
    #print("z log var")
    #print(z_para[1])

    print('[INFO] ')

    recons = recons_test.cpu().numpy()
    np.save(arr = recons, file = './a8_ivae_torch_recons_t2m.npy')
    print("Done!", file=f)

    # compute time and loss
    elapsed_time = time.time() - start_time
    print("Processing time in total: ", elapsed_time/60, file=f)


if __name__ == '__main__':
    # execute main function only if script is called as the __main__ script
    # main is not executed, e.g., when "from ivae_pytorch import EarlyStopper" is called from within a different script
    main()


if f is not None:
    f.close()