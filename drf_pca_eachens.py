import numpy as np
import torch
from sklearn.decomposition import PCA
import time

from data.new import build_ens_var_grid
from utils.evaluate import evaluation_metric


#for VAR in ['t2m', 'u10', 'v10']:
for VAR in ['z500']:
    for dim_latent in [2, 4, 8, 16, 32]:

        #dim_latent = 32
        start_time = time.time()
        
        f = open('./drf_pca_d'+str(dim_latent)+'_eachens_'+VAR+'.txt','w')
        
        print('Standard normalization', file=f)
        train_end=2920
        test_start=3285
        
        train_all, val_all, test_all, _ = build_ens_var_grid(var=VAR)
        # shape (n*50, 1, 81, 81)    
    
        # flatten grid data
        input_train = np.reshape(train_all, (-1, 81*81))
        input_val = np.reshape(val_all, (-1, 81*81))
        input_test = np.reshape(test_all, (-1, 81*81))
        # shape (n*50, 81*81) 
        
        # apply PCA
        pca = PCA(n_components=dim_latent).fit(input_train)
        
        # Project data into the reduced-dimensional space
        latent_train = pca.transform(input_train)
        latent_val = pca.transform(input_val)
        latent_test = pca.transform(input_test)
        # shape (n*50, 32) 
        
        # Reconstruct data from the reduced-dimensional space
        recons_train = pca.inverse_transform(latent_train)
        recons_val = pca.inverse_transform(latent_val)
        recons_test = pca.inverse_transform(latent_test)
        # shape (n*50, 81*81)
        
        # compute MSE
        mse_train = np.mean((input_train - recons_train) ** 2)
        mse_val = np.mean((input_val - recons_val) ** 2)
        mse_test = np.mean((input_test - recons_test) ** 2)
        
        # reshape reconstructed data back to images
        raw = np.reshape(input_test, (-1, 1, 81, 81))
        recons = np.reshape(recons_test, (-1, 1, 81, 81))
        # shape (n*50, 81, 81)
        
        raw = raw.reshape((-1, 50, 81, 81), order='F')
        recons = recons.reshape((-1, 50, 81, 81), order='F')
        # shape (n, 50, 81, 81)
        
        latent_test = latent_test.reshape((-1, 50, dim_latent), order='F')
        latent_train = latent_train.reshape((-1, 50, dim_latent), order='F')
        latent_val = latent_val.reshape((-1, 50, dim_latent), order='F')
        # shape (n, 50, 32)
        
        latent_mu = np.empty((0, dim_latent)) # shape (n, 32)
        latent_cov = np.empty((0, dim_latent, dim_latent)) # shape (n, 32, 32)
        latent_sample = np.empty((0, 50, dim_latent)) # shape (n, 50, 32)
        
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
        new_recons = pca.inverse_transform(latent_sample_flat) # shape (n*50, 81*81)
        new_recons = new_recons.reshape((-1, 50, 81, 81)) # shape (n, 50, 81, 81)
        
        print('shape of output new reconstructed figures: ', new_recons.shape, file=f)
        print('shape of latent representations: ', latent_test.shape, file=f)
        
        print('reconstructed test data - target data (mean): ', recons.mean(axis=(2, 3))[:1] - raw.mean(axis=(2, 3))[:1], file=f)
        print('new reconstructed test data - target data (mean): ', new_recons.mean(axis=(2, 3))[:1] - raw.mean(axis=(2, 3))[:1], file=f)
        print('reconstructed test data - target data (std): ', recons.std(axis=(2, 3))[:1] - raw.std(axis=(2, 3))[:1], file=f)
        print('new reconstructed test data - target data (std): ', new_recons.std(axis=(2, 3))[:1] - raw.std(axis=(2, 3))[:1], file=f)
        
        path_fields = '/hkfs/work/workspace_haic/scratch/gm2154-ae_data/output_fcst_fields/'
        path_latent = '/hkfs/work/workspace_haic/scratch/gm2154-ae_data/output_latent/'
        #path_model = '/hkfs/work/workspace_haic/scratch/gm2154-ae_data/output_model/'
        
        np.save(arr = raw, file = path_fields + 'pca_d'+str(dim_latent)+'_eachens_raw_'+VAR+'.npy')
        np.save(arr = new_recons, file = path_fields + 'pca_d'+str(dim_latent)+'_eachens_recons_'+VAR+'.npy')
        np.save(arr = recons, file = path_fields + 'pca_d'+str(dim_latent)+'_eachens_ensrecons_'+VAR+'.npy')
        np.save(arr = latent_test, file = path_latent + 'pca_d'+str(dim_latent)+'_eachens_latent_test_'+VAR+'.npy')
        np.save(arr = latent_val, file = path_latent + 'pca_d'+str(dim_latent)+'_eachens_latent_val_'+VAR+'.npy')
        np.save(arr = latent_train, file = path_latent + 'pca_d'+str(dim_latent)+'_eachens_latent_train_'+VAR+'.npy')
        
        mse_test_new = np.mean((raw - new_recons) ** 2)
        
        print(f"Raw data max: {raw.max()}; Raw data min: {raw.min()};\nRecons data max: {recons.max()}; Recons data min: {recons.min()};\nNew recons data max: {new_recons.max()}; Recons data min: {new_recons.min()}.", file=f)
            
        print('RMSE of training data: ', np.sqrt(mse_train), file=f)
        print('RMSE of validation data: ', np.sqrt(mse_val), file=f)
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


