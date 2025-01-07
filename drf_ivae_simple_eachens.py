# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 09:56:59 2022

@author: gm2154
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
import time

from data.new import build_ivae_dataset
from model.ivae_simple import Ivae
from utils import EarlyStopper, device
from utils.evaluate import evaluation_metric


def main():

    for N_LATENT in [2, 4, 8, 16, 32]:
    #for VAR in ['t2m', 'u10', 'v10']:
    #for VAR in ['z500', 't2m', 'u10', 'v10']:
        VAR = 'z500'
        
        # model hyper-parameters:
        BATCH_SIZE = 64
        LEARNING_RATE = 1e-4 # 1e-4
        EPOCHS = 300
        #N_LATENT = 32 #2
        ACT_FUN = 'leakyrelu' # none, relu, leakyrelu, gelu
        NUM_NODES = 4096 #[4096, 2048, 2048]
        reg_w = 0.001
        wd_w = 0.01
        ed_w = 1
        rmse_w = 0
        best_val_loss = 100
        dim_latent = N_LATENT
        model_name = 'ivae_simple_new'
        
        f = open('./drf_'+model_name+'_d'+str(dim_latent)+'_eachens_'+VAR+'.txt','w')
        
        print("Model: simple iVAE with dense layers; no additional orog-lsm input.", file=f)
        print(f"Weather variable: {VAR}; Input data shape (bs, 50, 81, 81);", file=f)
        print(f"Batch size {BATCH_SIZE}, Learning rate {LEARNING_RATE}, N(epochs) = {EPOCHS}, N(latent) = {N_LATENT}.", file=f)
        print(f"Model type: ivae_simple; Activation function: {ACT_FUN}; nodes = {NUM_NODES}.", file=f)
        print(f"Loss: {reg_w} * KLD + {wd_w} * WD + {ed_w} * ED + {rmse_w} * RMSE. \n", file=f)
        
        path_fields = '/hkfs/work/workspace_haic/scratch/gm2154-ae_data/output_fcst_fields/'
        path_latent = '/hkfs/work/workspace_haic/scratch/gm2154-ae_data/output_latent/'
        path_model = '/hkfs/work/workspace_haic/scratch/gm2154-ae_data/output_model/'
        
        train, val, test, scales = build_ivae_dataset(var=VAR)
        print(f"Scales of standardization: {scales}", file=f)
        
        train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test, batch_size=BATCH_SIZE)
        
        #model = Ivae(dim_latent=N_LATENT, n_samples=50, reg_weight=reg_w, wd_weight=wd_w, ed_weight=ed_w, rmse_weight=rmse_w, n_nodes1=4096, activation=ACT_FUN)
        model = Ivae(dim_latent=N_LATENT, n_samples=50, reg_weight=reg_w, wd_weight=wd_w, ed_weight=ed_w, rmse_weight=rmse_w, n_nodes1=4096, n_nodes2=2048, n_nodes3=2048, activation=ACT_FUN)
        model.to(device)
    
        #optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, threshold=0.005)
    
        # Training the model
        train_loss = []
        val_loss = []
    
        early_stopper = EarlyStopper(patience=20, min_delta=0)
        print("Start training VAE...", file=f)
        start_time = time.time()
    
        for t in range(EPOCHS):
            print(t+1)
            print(f"Epoch {t+1}\n-------------------------------", file=f)
    
            epoch_train_loss, z_mean_train, z_log_var_train = train_loop(train_loader, model, optimizer, f)
            train_loss.append(epoch_train_loss)
    
            epoch_validate_loss, z_mean_val, z_log_var_val = val_loop(val_loader, model, scheduler, f)
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
        test_loss, recons_test, data_test, z_mean, z_log_var = test_loop(test_loader, model, f)
    
        train_loss = np.array(train_loss)
        val_loss = np.array(val_loss)
        losses_train_val = np.concatenate((np.expand_dims(train_loss, axis=0), 
                                      np.expand_dims(val_loss, axis=0)), axis=0)
        np.save(arr = losses_train_val, file = './losses_'+model_name+'_d'+str(dim_latent)+'_eachens_'+VAR+'.npy')
        
        # generate new samples
        model.eval()
        
        with torch.no_grad():
            new_samples = model.generate(z_mean, z_log_var)
                
        new_recons = new_samples.cpu().numpy()
        
        raw = data_test.cpu().numpy()
        recons = recons_test.cpu().numpy()
        
        latent_mu_train = z_mean_train.cpu().detach().numpy()
        latent_log_var_train = z_log_var_train.cpu().detach().numpy()
        latent_mu_val = z_mean_val.cpu().detach().numpy()
        latent_log_var_val = z_log_var_val.cpu().detach().numpy()
        
        latent_train = np.concatenate((np.expand_dims(latent_mu_train, axis=0), 
                                       np.expand_dims(latent_log_var_train, axis=0)), axis=0)
        latent_val = np.concatenate((np.expand_dims(latent_mu_val, axis=0), 
                                     np.expand_dims(latent_log_var_val, axis=0)), axis=0)
        
        latent_mu = z_mean.cpu().numpy()
        latent_log_var = z_log_var.cpu().numpy()
        
        latent_test = np.concatenate((np.expand_dims(latent_mu, axis=0), 
                                      np.expand_dims(latent_log_var, axis=0)), axis=0)
        
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
    
    mean_all = torch.Tensor().to(device)
    log_var_all = torch.Tensor().to(device)
    
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
        
        z_mean = results[3]
        z_log_var = results[4]
        mean_all = torch.cat((mean_all, z_mean), 0)
        log_var_all = torch.cat((log_var_all, z_log_var), 0)

        if batch % 40 == 0:
            loss, current = loss.item(), batch * len(X[0])
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]", file=f)
            print(f"RMSE: {loss_all.get('RMSE'):>7f}; KLD: {loss_all.get('KLD'):>7f}; ED: {loss_all.get('ED'):>7f}; WD: {loss_all.get('WD'):>7f}", file=f)

    train_loss /= num_batches
    print(f"Avg training loss: {train_loss:>8f} \n", file=f)
    
    return train_loss, mean_all, log_var_all


def val_loop(dataloader, autoencoder, scheduler, f):
    
    num_batches = len(dataloader)
    val_loss = 0.0
    # for comparison
    val_loss2 = 0.0
    val_loss3 = 0.0
    autoencoder.eval()
    
    mean_all = torch.Tensor().to(device)
    log_var_all = torch.Tensor().to(device)
    
    with torch.no_grad():
        for X in dataloader:
            data = X[0].to(device)
            results = autoencoder(data.float())
            loss_all = autoencoder.loss_function(*results)
            loss = loss_all.get('loss')
            val_loss += loss.item()
            
            z_mean = results[3]
            z_log_var = results[4]
            mean_all = torch.cat((mean_all, z_mean), 0)
            log_var_all = torch.cat((log_var_all, z_log_var), 0)

    val_loss /= num_batches
    print(f"Avg validation loss: {val_loss:>8f} \n", file=f)
    
    scheduler.step(val_loss)

    return val_loss, mean_all, log_var_all


def test_loop(dataloader, autoencoder, f):
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


#if f is not None:
#    f.close()