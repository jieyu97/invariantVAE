import torch
from torch import nn
from torch.utils.data import DataLoader

import numpy as np
import time

from data import build_ens_allvar_dataset
from model.ivae import Ivae_split
from utils import EarlyStopper, device
from model.loss import energy_distance, rmse, energy_rmse_wd


VAR = 't2m' # t2m or z500 or u10 or v10 or q850
test_version = '_ivae_old'
f = open('./model_training_' + VAR + test_version + '.txt','w')

# Get Device for Training
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")



def main():
    # model hyper-parameters:
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    EPOCHS = 6
    N_LATENT = 32

    ws_path = '/hkfs/work/workspace_haic/scratch/gm2154-ae_data/model_pred/'
    train, val, test, _ = build_ens_allvar_dataset(var=VAR)
    
    train_loader = DataLoader(train, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test, batch_size=BATCH_SIZE)

    model = Ivae_split()
    model.to(device)

    # Initialize the loss function
    loss_fn = energy_rmse_wd

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training the model
    train_loss = []
    val_loss = []

    early_stopper = EarlyStopper(patience=10, min_delta=0)
    print("Start training VAE...", file=f)
    start_time = time.time()

    for t in range(EPOCHS):
        print(t+1)
        print(f"Epoch {t+1}\n-------------------------------", file=f)

        epoch_train_loss = train_loop(train_loader, model, loss_fn, optimizer)
        train_loss.append(epoch_train_loss)

        epoch_validate_loss = val_loop(val_loader, model, loss_fn)
        val_loss.append(epoch_validate_loss)

        if early_stopper.early_stop(epoch_validate_loss):
            print("Early stopping at epoch:", t)
            break

    print("Training finished!", file=f)

    test_loss, recons_test, data_test, z_mean, z_log_var = test_loop(test_loader, model, loss_fn)


    print('shape of output reconstructed figures: ', recons_test.shape, file=f)
    print('shape of latent parameters (mean): ', z_mean.shape, file=f)

    print("--------------------- \n", file=f)

    print('reconstructed test data (mean): ', recons_test.mean(dim=(2, 3))[:1], file=f)
    print('target data (mean): ', data_test.mean(dim=(2, 3))[:1], file=f)
    print('reconstructed test data (std): ', recons_test.std(dim=(2, 3))[:1], file=f)
    print('target data (std): ', data_test.std(dim=(2, 3))[:1], file=f)

    print("--------------------- \n", file=f)

    recons = recons_test.cpu().numpy()
    
    latent_mu = z_mean.cpu().numpy()
    latent_log_var = z_log_var.cpu().numpy()
    latent_sigma = np.exp(latent_log_var / 2)
    latent_para = np.concatenate((np.expand_dims(latent_mu, axis=0), 
                                  np.expand_dims(latent_sigma, axis=0)), axis=0)
    
    raw_data = data_test.cpu().numpy()
    
    np.save(arr = raw_data, file = ws_path + 'ae_ens_raw_'+VAR+test_version+'.npy')
    np.save(arr = recons, file = ws_path + 'ae_ens_recons_'+VAR+test_version+'.npy')
    np.save(arr = latent_para, file = './ae_ens_latent_'+VAR+test_version+'.npy')
    
    print(f"Raw data max: {data_test.max()}; Raw data min: {data_test.min()};\nRecons data max: {recons_test.max()}; Recons data min: {recons_test.min()}.", file=f)
    print(f"Latent representations with dimension {N_LATENT}:\n{latent_test}", file=f)
    
    # compute time and loss
    elapsed_time = time.time() - start_time
    print(f"\n Processing time in total: {elapsed_time/60:>3f} \n", file=f)
    

    
def train_loop(dataloader, autoencoder, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0.0
    autoencoder.train()
    for batch, X in enumerate(dataloader):
        data = X[0].to(device)

        # Backpropagation
        optimizer.zero_grad()
        # Compute prediction and loss
        pred, z_para = autoencoder(data.float())
        loss = loss_fn(pred, data.float())
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        if batch % 40 == 0:
            loss, current = loss.item(), batch * len(X[0])
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]", file=f)

    train_loss /= num_batches
    print(f"Avg training loss: {train_loss:>8f} \n", file=f)
    # train_loop does no longer return the model, as there is no need to do that
    return train_loss


def val_loop(dataloader, autoencoder, loss_fn):
    num_batches = len(dataloader)
    val_loss = 0.0
    # for comparison
    val_loss2 = 0.0
    val_loss3 = 0.0
    autoencoder.eval()
    with torch.no_grad():
        for X in dataloader:
            data = X[0].to(device)
            pred, z_para = autoencoder(data.float())
            loss = loss_fn(pred, data.float())
            val_loss += loss.item()
            # for comparison
            loss2 = rmse(pred, data.float())
            val_loss2 += loss2.item()
            loss3 = energy_distance(pred, data.float())
            val_loss3 += loss3.item()

    val_loss /= num_batches
    print(f"Avg validation loss: {val_loss:>8f} \n", file=f)
    # for comparison
    val_loss2 /= num_batches
    val_loss3 /= num_batches
    print(f"Validation data - Mean Ensemble RMSE: {val_loss2:>8f}", file=f)
    print(f"Validation data - Energy Distance: {val_loss3:>8f} \n", file=f)
    return val_loss


def test_loop(dataloader, autoencoder, loss_fn):
    num_batches = len(dataloader)
    test_loss = 0.0
    # for comparison
    test_loss2 = 0.0
    test_loss3 = 0.0
    autoencoder.eval()
    
    pred_all = torch.Tensor().to(device)
    data_all = torch.Tensor().to(device)
    mean_all = torch.Tensor().to(device)
    log_var_all = torch.Tensor().to(device)
    
    with torch.no_grad():
        for X in dataloader:
            data = X[0].to(device)
            pred, z_para = autoencoder(data.float())
            test_loss += loss_fn(pred, data.float())
            # for comparison
            test_loss2 += rmse(pred, data.float())
            test_loss3 += energy_distance(pred, data.float())
            
            pred_all = torch.cat((pred_all, pred), 0)
            data_all = torch.cat((data_all, data), 0)
            mean_all = torch.cat((mean_all, z_para[0]), 0)
            log_var_all = torch.cat((log_var_all, z_para[1]), 0)

    test_loss /= num_batches
    print(f"Avg test loss: {test_loss:>8f} \n", file=f)
    # for comparison
    test_loss2 /= num_batches
    test_loss3 /= num_batches
    print(f"Test data - Mean Ensemble RMSE: {test_loss2:>8f}", file=f)
    print(f"Test data - Energy Distance: {test_loss3:>8f} \n", file=f)

    return test_loss, pred_all, data_all, mean_all, log_var_all



if __name__ == '__main__':
    # execute main function only if script is called as the __main__ script
    # main is not executed, e.g., when "from ivae_pytorch import EarlyStopper" is called from within a different script
    main()


if f is not None:
    f.close()