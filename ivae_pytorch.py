# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 09:56:59 2022

@author: gm2154
"""

import torch
from torch import nn
from torch.utils.data import DataLoader

import numpy as np
import time

from data import build_dataset
from models.autoencoder import IvaeEncoder, IvaeDecoder, Ivae
from utils import EarlyStopper

f = None #open('./model_training.txt','w')

# Get Device for Training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


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