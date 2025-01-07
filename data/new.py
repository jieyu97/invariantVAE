import numpy as np
import torch
from torch.utils.data import TensorDataset


def compute_normalization(x_train):
    mu = x_train.ravel().mean()
    sigma = x_train.ravel().std()
    return mu, sigma


def apply_normalization(x, scales):
    mu, sigma = scales
    return (x - mu) / sigma


def reverse_normalization(x, scales):
    mu, sigma = scales
    return x * sigma + mu


def get_ensemble_data_allvar(var='t2m'):
    ws_path = '/hkfs/work/workspace_haic/scratch/gm2154-ae_data/'
    if var == 't2m':
        path = ws_path + 'grided_ens_t2m_alldates.npy'
    elif var == 'z500':
        path = ws_path + 'grided_ens_z500_alldates.npy'
    elif var == 'u10':
        path = ws_path + 'grided_ens_u10_alldates.npy'
    elif var == 'v10':
        path = ws_path + 'grided_ens_v10_alldates.npy'
    elif var == 'q850':
        path = ws_path + 'grided_ens_q850_alldates.npy'
    elif var == 'd2m':
        path = ws_path + 'grided_ens_d2m_alldates.npy'
    elif var == 'sp':
        path = ws_path + 'grided_ens_sp_alldates.npy'
    elif var == 'u500':
        path = ws_path + 'grided_ens_u500_alldates.npy'
    elif var == 'v500':
        path = ws_path + 'grided_ens_v500_alldates.npy'
    elif var == 'u850':
        path = ws_path + 'grided_ens_u850_alldates.npy'
    elif var == 'v850':
        path = ws_path + 'grided_ens_v850_alldates.npy'
    elif var == 'ssr':
        path = ws_path + 'grided_ens_ssr_alldates.npy'
    elif var == 'cape':
        path = ws_path + 'grided_ens_cape_alldates.npy'
    elif var == 'slhf':
        path = ws_path + 'grided_ens_slhf_alldates.npy'
    elif var == 'sshf':
        path = ws_path + 'grided_ens_sshf_alldates.npy'
    elif var == 'tcc':
        path = ws_path + 'grided_ens_tcc_alldates.npy'
    
    data = np.load(path)
    data_used = data[:-2, :, :, :]
    return data_used
    
    
def build_ens_allvar_dataset(var='t2m', reorder='F', train_end=2920, test_start=3285):

    # read data, weather variable: t2m or 500gh
    weather_variable = var
    data_gridded = get_ensemble_data_allvar(var=weather_variable)
    # shape: (3651, 81, 81, 50)

    # move the channel (ensemble members) dimension to the first
    data_gridded = np.moveaxis(data_gridded, 3, 1) # (3651, 50, 81, 81)
    # split into training, validation, and test sets (training: 2007-2014, validation: 2015, test: 2016)
    # training data
    x_train = data_gridded[:train_end, :, :, :]
    scales = compute_normalization(x_train)
    x_train_normalized = apply_normalization(x_train, scales)
    x_train_norm_ens = x_train_normalized.reshape((x_train.shape[0]*50, 1, 81, 81), order=reorder)
    
    # validation data
    x_val = data_gridded[train_end:test_start, :, :, :]
    x_val_normalized = apply_normalization(x_val, scales)
    x_val_norm_ens = x_val_normalized.reshape((x_val.shape[0]*50, 1, 81, 81), order=reorder)
    
    # test data
    x_test = data_gridded[test_start:, :, :, :]
    x_test_normalized = apply_normalization(x_test, scales)
    x_test_norm_ens = x_test_normalized.reshape((x_test.shape[0]*50, 1, 81, 81), order=reorder)
    
    # model inputs:
    train_data = torch.from_numpy(x_train_norm_ens)
    val_data = torch.from_numpy(x_val_norm_ens)
    test_data = torch.from_numpy(x_test_norm_ens)
    
    train = TensorDataset(train_data)
    val = TensorDataset(val_data)
    test = TensorDataset(test_data)
    
    return train, val, test, scales


def build_ivae3d_dataset(var='t2m', train_end=2920, test_start=3285):

    # read data, weather variable: t2m or 500gh
    weather_variable = var
    data_gridded = get_ensemble_data_allvar(var=weather_variable)
    # shape: (3651, 81, 81, 50)

    # move the channel (ensemble members) dimension to the first
    data_gridded = np.moveaxis(data_gridded, 3, 1) # (3651, 50, 81, 81)
    # split into training, validation, and test sets (training: 2007-2014, validation: 2015, test: 2016)
    # training data
    x_train = data_gridded[:train_end, :, :, :]
    scales = compute_normalization(x_train)
    x_train_normalized = apply_normalization(x_train, scales)
    x_train_norm_ens = np.expand_dims(x_train_normalized, axis=1)
    
    # validation data
    x_val = data_gridded[train_end:test_start, :, :, :]
    x_val_normalized = apply_normalization(x_val, scales)
    x_val_norm_ens = np.expand_dims(x_val_normalized, axis=1)
    
    # test data
    x_test = data_gridded[test_start:, :, :, :]
    x_test_normalized = apply_normalization(x_test, scales)
    x_test_norm_ens = np.expand_dims(x_test_normalized, axis=1)
    
    # model inputs:
    train_data = torch.from_numpy(x_train_norm_ens)
    val_data = torch.from_numpy(x_val_norm_ens)
    test_data = torch.from_numpy(x_test_norm_ens)
    
    train = TensorDataset(train_data)
    val = TensorDataset(val_data)
    test = TensorDataset(test_data)
    
    return train, val, test, scales
    

def build_ens_var_grid(var='t2m', train_end=2920, test_start=3285):

    # read data, weather variable: t2m or 500gh
    weather_variable = var
    data_gridded = get_ensemble_data_allvar(var=weather_variable)
    # shape: (3651, 81, 81, 50)

    # move the channel (ensemble members) dimension to the first
    data_gridded = np.moveaxis(data_gridded, 3, 1) # (3651, 50, 81, 81)
    # split into training, validation, and test sets (training: 2007-2014, validation: 2015, test: 2016)
    # training data
    x_train = data_gridded[:train_end, :, :, :]
    scales = compute_normalization(x_train)
    x_train_normalized = apply_normalization(x_train, scales)
    x_train_norm_ens = x_train_normalized.reshape((x_train.shape[0]*50, 1, 81, 81), order='F')
    
    # validation data
    x_val = data_gridded[train_end:test_start, :, :, :]
    x_val_normalized = apply_normalization(x_val, scales)
    x_val_norm_ens = x_val_normalized.reshape((x_val.shape[0]*50, 1, 81, 81), order='F')
    
    # test data
    x_test = data_gridded[test_start:, :, :, :]
    x_test_normalized = apply_normalization(x_test, scales)
    x_test_norm_ens = x_test_normalized.reshape((x_test.shape[0]*50, 1, 81, 81), order='F')
    
    return x_train_norm_ens, x_val_norm_ens, x_test_norm_ens, scales
    
    
def build_eachens_var_grid(var='t2m', train_end=2920, test_start=3285):

    # read data, weather variable: t2m or 500gh
    weather_variable = var
    data_gridded = get_ensemble_data_allvar(var=weather_variable)
    # shape: (3651, 81, 81, 50)

    # move the channel (ensemble members) dimension to the first
    data_gridded = np.moveaxis(data_gridded, 3, 1) # (3651, 50, 81, 81)
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
    
    return x_train_normalized, x_val_normalized, x_test_normalized, scales
    
    
def build_ens_nvar_dataset(var='ws10', train_end=2920, test_start=3285):

    if var == 'ws10':
    
        # read data, weather variable: t2m or 500gh
        weather_variable = var
        train1, val1, test1, scale1 = build_ens_var_grid(var='u10')
        train2, val2, test2, scale2 = build_ens_var_grid(var='v10')
        # shape: (3651, 81, 81, 50)
        
        train_all = np.concatenate((train1, train2), axis=1)
        val_all = np.concatenate((val1, val2), axis=1)
        test_all = np.concatenate((test1, test2), axis=1)
        
    elif var == 'ws':
    
        # read data, weather variable: t2m or 500gh
        weather_variable = var
        VAR1 = 'u10'
        VAR2 = 'v10'
        VAR3 = 'u500'
        VAR4 = 'v500'
        VAR5 = 'u850'
        VAR6 = 'v850'
        train1, val1, test1, _ = build_ens_var_grid(var=VAR1)
        train2, val2, test2, _ = build_ens_var_grid(var=VAR2)
        train3, val3, test3, _ = build_ens_var_grid(var=VAR3)
        train4, val4, test4, _ = build_ens_var_grid(var=VAR4)
        train5, val5, test5, _ = build_ens_var_grid(var=VAR5)
        train6, val6, test6, _ = build_ens_var_grid(var=VAR6)
        # shape: (3651, 81, 81, 50)
        
        train_all = np.concatenate((train1, train2, train3, train4, train5, train6), axis=1)
        val_all = np.concatenate((val1, val2, val3, val4, val5, val6), axis=1)
        test_all = np.concatenate((test1, test2, test3, test4, test5, test6), axis=1) 
        
    # model inputs:
    train_data = torch.from_numpy(train_all)
    val_data = torch.from_numpy(val_all)
    test_data = torch.from_numpy(test_all)
    
    train = TensorDataset(train_data)
    val = TensorDataset(val_data)
    test = TensorDataset(test_data)
        
        
    return train, val, test



def build_ivae_dataset(var='t2m', train_end=2920, test_start=3285):

    # read data, weather variable: t2m or 500gh
    weather_variable = var
    data_gridded = get_ensemble_data_allvar(var=weather_variable)
    # shape: (3651, 81, 81, 50)

    # move the channel (ensemble members) dimension to the first
    data_gridded = np.moveaxis(data_gridded, 3, 1) # (3651, 50, 81, 81)
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
    
    train = TensorDataset(train_data)
    val = TensorDataset(val_data)
    test = TensorDataset(test_data)
    
    return train, val, test, scales
