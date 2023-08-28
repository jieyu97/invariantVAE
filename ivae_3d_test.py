import torch
from torch import nn
from torch.utils.data import DataLoader

import numpy as np
import time

from data import build_ivae3d_dataset
from model.ivae_3d import Ivae
from utils import EarlyStopper, device
from utils.evaluate import evaluation_metric


VAR = 't2m' # t2m or z500 or u10 or v10 or q850
test_version = '_ivae_3d_t4'
f = open('./model_training_' + VAR + test_version + '.txt','w')

# Get Device for Training
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


def main():
    # model hyper-parameters:
    BATCH_SIZE = 6
    LEARNING_RATE = 1e-6
    EPOCHS = 50
    N_LATENT = 128

    print(f"Weather variable: {VAR}; Input data shape (bs, 1, 50, 81, 81);", file=f)
    print(f"Batch size {BATCH_SIZE}, Learning rate {LEARNING_RATE}, N(epochs) = {EPOCHS}, N(latent) = {N_LATENT}. \n", file=f)
    
    ws_path = '/hkfs/work/workspace_haic/scratch/gm2154-ae_data/model_pred/'
    train, val, test, _ = build_ivae3d_dataset(var=VAR)
    
    train_loader = DataLoader(train, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test, batch_size=BATCH_SIZE)

    model = Ivae(dim_latent=N_LATENT, n_samples=50, reg_weight=0.0005, ed_weight=1, wd_weight=0.1) #0.005
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training the model
    train_loss = []
    val_loss = []

    early_stopper = EarlyStopper(patience=16, min_delta=0)
    print("Start training VAE...", file=f)
    start_time = time.time()

    for t in range(EPOCHS):
        print(t+1)
        print(f"Epoch {t+1}\n-------------------------------", file=f)

        epoch_train_loss = train_loop(train_loader, model, optimizer)
        train_loss.append(epoch_train_loss)

        epoch_validate_loss = val_loop(val_loader, model)
        val_loss.append(epoch_validate_loss)

        if early_stopper.early_stop(epoch_validate_loss):
            print("Early stopping at epoch:", t)
            break

    print("Training finished!", file=f)

    test_loss, recons_test, data_test, z_mean, z_log_var = test_loop(test_loader, model)


    print('shape of output reconstructed figures: ', recons_test.shape, file=f)
    print('shape of latent parameters (mean): ', z_mean.shape, file=f)
    
    recons_test = torch.squeeze(recons_test, dim=1)
    data_test = torch.squeeze(data_test, dim=1)

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
    
    np.save(arr = raw_data, file = ws_path + 'ivae_raw_'+VAR+test_version+'.npy')
    np.save(arr = recons, file = ws_path + 'ivae_recons_'+VAR+test_version+'.npy')
#    np.save(arr = latent_para, file = './ae_ens_latent_'+VAR+test_version+'.npy')
    
    print(f"Raw data max: {raw_data.max()}; Raw data min: {raw_data.min()};\nRecons data max: {recons.max()}; Recons data min: {recons.min()}.", file=f)
    print(f"Latent representations with dimension {N_LATENT}:\n{latent_para}", file=f)
    
    # compute time and loss
    elapsed_time = time.time() - start_time
    print(f"\n Processing time in total: {elapsed_time/60:>3f} \n", file=f)
    
#    recons_scaled = reverse_normalization(recons, scales)
#    raw_scaled = reverse_normalization(raw_data, scales)
    
    rmse, energy_distance_all, energy_distance_grid, wasserstein_distance = evaluation_metric(raw_data, recons)
    
    print("Mean RMSE: ", rmse.mean(), file=f) # (366,)
    print("Mean energy distance all grids: ", energy_distance_all.mean(), file=f) # (366,)
    print("Mean energy distance each grid: ", energy_distance_grid.mean(), file=f) # (366,)
    print("Mean Wasserstein distance: ", wasserstein_distance.mean(), file=f) # (366,)
    
    print("RMSE: ", rmse, file=f)
    print("Energy distance all grids: ", energy_distance_all, file=f)
    print("Energy distance each grid: ", energy_distance_grid, file=f)
    print("Wasserstein distance: ", wasserstein_distance, file=f)


def train_loop(dataloader, autoencoder, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0.0
    autoencoder.train()
    for batch, X in enumerate(dataloader):
        data = X[0].to(device)

        # Backpropagation
        optimizer.zero_grad()
        # Compute prediction and loss
        results = autoencoder(data.float())
        loss_all = autoencoder.loss_function(*results)
        loss = loss_all.get('loss')
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        if batch % 40 == 0:
            loss, current = loss.item(), batch * len(X[0])
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]", file=f)
            print(f"MSE: {loss_all.get('MSE'):>7f}; KLD: {loss_all.get('KLD'):>7f}; ED: {loss_all.get('ED'):>7f}; WD: {loss_all.get('WD'):>7f}", file=f)

    train_loss /= num_batches
    print(f"Avg training loss: {train_loss:>8f} \n", file=f)
    
    return train_loss


def val_loop(dataloader, autoencoder):
    num_batches = len(dataloader)
    val_loss = 0.0
    # for comparison
    val_loss2 = 0.0
    val_loss3 = 0.0
    autoencoder.eval()
    with torch.no_grad():
        for X in dataloader:
            data = X[0].to(device)
            results = autoencoder(data.float())
            loss_all = autoencoder.loss_function(*results)
            loss = loss_all.get('loss')
            val_loss += loss.item()

    val_loss /= num_batches
    print(f"Avg validation loss: {val_loss:>8f} \n", file=f)
    
    return val_loss


def test_loop(dataloader, autoencoder):
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
            results = autoencoder(data.float())
            loss_all = autoencoder.loss_function(*results)
            loss = loss_all.get('loss')
            test_loss += loss.item()
            pred = results[0]
            raw = results[1]
            z_mean = results[3]
            z_log_var = results[4]
            
            pred_all = torch.cat((pred_all, pred), 0)
            data_all = torch.cat((data_all, data), 0)
            mean_all = torch.cat((mean_all, z_mean), 0)
            log_var_all = torch.cat((log_var_all, z_log_var), 0)

    test_loss /= num_batches
    print(f"Avg test loss: {test_loss:>8f} \n", file=f)

    return test_loss, pred_all, data_all, mean_all, log_var_all


    
if __name__ == '__main__':
    # execute main function only if script is called as the __main__ script
    # main is not executed, e.g., when "from ivae_pytorch import EarlyStopper" is called from within a different script
    main()


if f is not None:
    f.close()