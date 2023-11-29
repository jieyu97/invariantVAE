# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 09:56:59 2022

@author: gm2154
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
import time

from data.new import build_ens_allvar_dataset
#from data.orog_lsm import orog_lsm_data
from model.ae_simple import AEens
from utils import EarlyStopper, device
from utils.evaluate import evaluation_metric


def main():

    for VAR in ['t2m', 'u10', 'v10']:
    #for VAR in ['z500', 't2m', 'u10', 'v10']:
    
        # model hyper-parameters:
        BATCH_SIZE = 1024
        LEARNING_RATE = 1e-4
        EPOCHS = 200
        N_LATENT = 32 #2
        best_val_loss = 10
        dim_latent = N_LATENT
        model_name = 'ae_simple'
        
        f = open('./drf_'+model_name+'_d'+str(dim_latent)+'_eachens_'+VAR+'.txt','w')
        
        print("Model: simple AE with dense layers; no additional orog-lsm input.", file=f)
        print(f"Weather variable: {VAR}; Input data shape (bs*50, 1, 81, 81);", file=f)
        print(f"Batch size {BATCH_SIZE}, Learning rate {LEARNING_RATE}, N(epochs) = {EPOCHS}, N(latent) = {N_LATENT}. \n", file=f)
        
        path_fields = '/hkfs/work/workspace_haic/scratch/gm2154-ae_data/output_fcst_fields/'
        path_latent = '/hkfs/work/workspace_haic/scratch/gm2154-ae_data/output_latent/'
        path_model = '/hkfs/work/workspace_haic/scratch/gm2154-ae_data/output_model/'
        
        train, val, test, _ = build_ens_allvar_dataset(var=VAR)
    
        train_loader = DataLoader(train, batch_size=BATCH_SIZE)
        val_loader = DataLoader(val, batch_size=BATCH_SIZE)
        test_loader = DataLoader(test, batch_size=BATCH_SIZE)
    
        ACT_FUN = 'leakyrelu' # none, relu, leakyrelu, gelu
        NUM_NODES1 = 4096
        NUM_NODES2 = 4096
        NUM_LAYERS = 2
        print(f"number of layers: {NUM_LAYERS}, number of nodes: {NUM_NODES1}, {NUM_NODES2}, activation function: {ACT_FUN}", file=f)
        model = AEens(latent_dim=N_LATENT, n_nodes1=NUM_NODES1, n_nodes2=NUM_NODES2, n_layers=NUM_LAYERS, activation=ACT_FUN)
        model.to(device)
    
        #optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, threshold=0.001)
        
        # Training the model
        train_loss = []
        val_loss = []
    
        early_stopper = EarlyStopper(patience=20, min_delta=0)
        print("Start training VAE...", file=f)
        start_time = time.time()
    
        for t in range(EPOCHS):
            print(t+1)
            print(f"Epoch {t+1}\n-------------------------------", file=f)
    
            epoch_train_loss, latent_train = train_loop(train_loader, model, optimizer, f)
            train_loss.append(epoch_train_loss)
    
            epoch_validate_loss, latent_val = val_loop(val_loader, model, scheduler, f)
            val_loss.append(epoch_validate_loss)

            if epoch_validate_loss < best_val_loss:
                best_val_loss = epoch_validate_loss
                print('Saving better model \n')
                torch.save(model.state_dict(), path_model+model_name+'_d'+str(dim_latent)+'_eachens_'+VAR+'.pth')
                
            if early_stopper.early_stop(epoch_validate_loss):
                print("Early stopping at epoch:", t)
                break
    
        print("Training finished!", file=f)
    
        model.load_state_dict(torch.load(path_model+model_name+'_d'+str(dim_latent)+'_eachens_'+VAR+'.pth', map_location="cuda:0"))
        test_loss, recons_test, data_test, latent_test = test_loop(test_loader, model, f)
        
        train_loss = np.array(train_loss)
        val_loss = np.array(val_loss)
        losses_train_val = np.concatenate((np.expand_dims(train_loss, axis=0), 
                                      np.expand_dims(val_loss, axis=0)), axis=0)
        np.save(arr = losses_train_val, file = './losses_'+model_name+'_d'+str(dim_latent)+'_eachens_'+VAR+'.npy')
        
        recons_test = recons_test.cpu().numpy()
        latent_test = latent_test.cpu().numpy()
        data_test = data_test.cpu().numpy()
        
        latent_train = latent_train.cpu().detach().numpy()
        latent_val = latent_val.cpu().detach().numpy()
        
        recons = recons_test.reshape((-1, 50, 81, 81), order='F')
        raw = data_test.reshape((-1, 50, 81, 81), order='F')
        
        latent_test = latent_test.reshape((-1, 50, N_LATENT), order='F')
        # shape (n, 50, 32)
        latent_train = latent_train.reshape((-1, 50, N_LATENT), order='F')
        latent_val = latent_val.reshape((-1, 50, N_LATENT), order='F')
        
        latent_mu = np.empty((0, N_LATENT)) # shape (n, 32)
        latent_cov = np.empty((0, N_LATENT, N_LATENT)) # shape (n, 32, 32)
        latent_sample = np.empty((0, 50, N_LATENT)) # shape (n, 50, 32)
        
        for day in range(latent_test.shape[0]):
            latent_day = latent_test[day,:,:]
            # Compute the sample mean
            mu_hat = np.mean(latent_day, axis=0)
            # Compute the sample covariance matrix
            cov_hat = np.cov(latent_day, rowvar=False)
            latent_new = np.random.multivariate_normal(mu_hat, cov_hat, size=50)
            
            latent_mu = np.concatenate((latent_mu, [mu_hat])) 
            latent_cov = np.concatenate((latent_cov, [cov_hat]))
            latent_sample = np.concatenate((latent_sample, [latent_new]))
        
        latent_sample_flat = np.reshape(latent_sample, (-1, dim_latent)) # shape (n*50, 32)
        latent_sample_flat = torch.from_numpy(latent_sample_flat)
        latent_sample_dataset = TensorDataset(latent_sample_flat)
        latent_loader = DataLoader(latent_sample_dataset, batch_size=BATCH_SIZE)
        
        model.eval()
        sample_all = torch.Tensor().to(device)
        
        with torch.no_grad():
            for X in latent_loader:
                data = X[0].to(device)
                new_samples = model.generate(data.float())
                sample_all = torch.cat((sample_all, new_samples), 0)
                
        new_recons = sample_all.cpu().numpy()
        new_recons = new_recons.reshape((-1, 50, 81, 81))
        
        print('shape of output new reconstructed figures: ', new_recons.shape, file=f)
        print('shape of latent representations: ', latent_test.shape, file=f)
        
        print('reconstructed test data - target data (mean): ', recons.mean(axis=(2, 3))[:1] - raw.mean(axis=(2, 3))[:1], file=f)
        print('new reconstructed test data - target data (mean): ', new_recons.mean(axis=(2, 3))[:1] - raw.mean(axis=(2, 3))[:1], file=f)
        print('reconstructed test data - target data (std): ', recons.std(axis=(2, 3))[:1] - raw.std(axis=(2, 3))[:1], file=f)
        print('new reconstructed test data - target data (std): ', new_recons.std(axis=(2, 3))[:1] - raw.std(axis=(2, 3))[:1], file=f)
        
        np.save(arr = raw, file = path_fields+model_name+'_d'+str(dim_latent)+'_eachens_raw_'+VAR+'.npy')
        np.save(arr = new_recons, file = path_fields+model_name+'_d'+str(dim_latent)+'_eachens_recons_'+VAR+'.npy')
        np.save(arr = recons, file = path_fields+model_name+'_d'+str(dim_latent)+'_eachens_ensrecons_'+VAR+'.npy')
        np.save(arr = latent_test, file = path_latent+model_name+'_d'+str(dim_latent)+'_eachens_latent_test_'+VAR+'.npy')
        np.save(arr = latent_val, file = path_latent+model_name+'_d'+str(dim_latent)+'_eachens_latent_val_'+VAR+'.npy')
        np.save(arr = latent_train, file = path_latent+model_name+'_d'+str(dim_latent)+'_eachens_latent_train_'+VAR+'.npy')
        
        print(f"Raw data max: {raw.max()}; Raw data min: {raw.min()};\nRecons data max: {recons.max()}; Recons data min: {recons.min()};\nNew recons data max: {new_recons.max()}; Recons data min: {new_recons.min()}.", file=f)
        
        mse_test = np.mean((raw - recons) ** 2)
        mse_test_new = np.mean((raw - new_recons) ** 2)
        
        print('RMSE of test data: ', np.sqrt(mse_test), file=f)
        print('\n RMSE of new generated test data: ', np.sqrt(mse_test_new), file=f)
    
        mae, std_true, std_pred, ed_1dim, ed_1dim_fun, ed_ndim, wd_1dim_fun, wd_ndim = evaluation_metric(raw, recons)
    
        print("----- Evaluation Metrics (ens recons) -----------------", file=f)
        print("MAE: ", mae, file=f)
        print("std raw and std recons : ", std_true, std_pred, file=f)
        print("1-dim energy distance (my codes and scipy function): ", ed_1dim, ed_1dim_fun, file=f)
        print("81*81-dim energy distance (my codes): ", ed_ndim, file=f)
        print("1-dim Wasserstein distance (scipy function): ", wd_1dim_fun, file=f)
        print("81*81-dim Sinkhorn distance (my codes): ", wd_ndim, file=f)
        
        mae, std_true, std_pred, ed_1dim, ed_1dim_fun, ed_ndim, wd_1dim_fun, wd_ndim = evaluation_metric(raw, new_recons)
        
        print("----- Evaluation Metrics (new generated) -----------------", file=f)
        print("MAE: ", mae, file=f)
        print("std raw and std recons : ", std_true, std_pred, file=f)
        print("1-dim energy distance (my codes and scipy function): ", ed_1dim, ed_1dim_fun, file=f)
        print("81*81-dim energy distance (my codes): ", ed_ndim, file=f)
        print("1-dim Wasserstein distance (scipy function): ", wd_1dim_fun, file=f)
        print("81*81-dim Sinkhorn distance (my codes): ", wd_ndim, file=f)
        
        # compute time and loss
        elapsed_time = time.time() - start_time
        print(f"\n Processing time in total: {elapsed_time/60:>3f} \n", file=f)
        
        f.close()    
    

def train_loop(dataloader, autoencoder, optimizer, f):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0.0
    autoencoder.train()
    
    latent_all = torch.Tensor().to(device)
    
    for batch, X in enumerate(dataloader):
        data = X[0].to(device)

        # Backpropagation
        optimizer.zero_grad()
        # Compute prediction and loss
        results = autoencoder(data.float())#, orog_lsm_input.float())
        loss_all = autoencoder.loss_function(*results)
        loss = loss_all.get('loss')
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        # save latent representations
        latent = results[2]
        latent_all = torch.cat((latent_all, latent), 0)

        if batch % 40 == 0:
            loss, current = loss.item(), batch * len(X[0])
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]", file=f)
            print(f"RMSE: {loss_all.get('RMSE'):>4f}", file=f)

    train_loss /= num_batches
    print(f"Avg training loss: {train_loss:>8f} \n", file=f)
    # train_loop does no longer return the model, as there is no need to do that
    
    return train_loss, latent_all


def val_loop(dataloader, autoencoder, scheduler, f):
    num_batches = len(dataloader)
    val_loss = 0.0
    autoencoder.eval()
    
    latent_all = torch.Tensor().to(device)
    
    with torch.no_grad():
        for X in dataloader:
            data = X[0].to(device)
            results = autoencoder(data.float())#, orog_lsm_input.float())
            loss_all = autoencoder.loss_function(*results)
            loss = loss_all.get('loss')
            val_loss += loss.item()
            # save latent representations
            latent = results[2]
            latent_all = torch.cat((latent_all, latent), 0)

    val_loss /= num_batches
    print(f"Avg validation loss: {val_loss:>8f} \n", file=f)
    
    scheduler.step(val_loss)
    
    return val_loss, latent_all


def test_loop(dataloader, autoencoder, f):
    num_batches = len(dataloader)
    test_loss = 0.0
    autoencoder.eval()
    
    pred_all = torch.Tensor().to(device)
    data_all = torch.Tensor().to(device)
    latent_all = torch.Tensor().to(device)
    
    with torch.no_grad():
        for X in dataloader:
            data = X[0].to(device)
            results = autoencoder(data.float())#, orog_lsm_input.float())
            loss_all = autoencoder.loss_function(*results)
            loss = loss_all.get('loss')
            test_loss += loss.item()
            # for comparison
            pred = results[0]
            raw = results[1]
            latent = results[2]
            
            pred_all = torch.cat((pred_all, pred), 0)
            data_all = torch.cat((data_all, raw), 0)
            latent_all = torch.cat((latent_all, latent), 0)

    test_loss /= num_batches
    print(f"Avg test loss: {test_loss:>8f} \n", file=f)

    return test_loss, pred_all, data_all, latent_all
    
    

if __name__ == '__main__':
    # execute main function only if script is called as the __main__ script
    # main is not executed, e.g., when "from ivae_pytorch import EarlyStopper" is called from within a different script
    main()


#if f is not None:
#    f.close()
    
    