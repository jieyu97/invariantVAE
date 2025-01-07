import numpy as np
import torch
from torch.utils.data import TensorDataset


def compute_normalization(x_train):
    mins_train = x_train.ravel().min()
    maxs_train = x_train.ravel().max()
    return mins_train, maxs_train


def apply_normalization(x, scales):
    mins_train, maxs_train = scales
    return (x - mins_train) / (maxs_train - mins_train)


def reverse_normalization(x, scales):
    mins_train, maxs_train = scales
    return x * (maxs_train - mins_train) + mins_train
    

def get_random_data(n_members = 1):
    data_gridded = torch.randn(3651, 81, 81, n_members) * torch.exp(torch.randn(3651, 1, 1, n_members) / 10) + torch.randn(3651, 1, 1, n_members)
    data_gridded = data_gridded.data.numpy()
    return data_gridded


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
    return np.load(path)
    
    
def get_ensemble_data(var='t2m'):
    if var == 't2m':
        path = '/pfs/work7/workspace/scratch/gm2154-ae_data/grided_ens_t2m_complete.npy'
    elif var == 'z500':
        path = '/pfs/work7/workspace/scratch/gm2154-ae_data/grided_ens_500gh_complete.npy'
    return np.load(path)


def build_dataset(var='t2m', train_end=2920, test_start=3285):

    # read data, weather variable: t2m or 500gh
    weather_variable = var
    data_gridded = get_ensemble_data(var=weather_variable)
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
    
    train = TensorDataset(train_data)
    val = TensorDataset(val_data)
    test = TensorDataset(test_data)
    
    return train, val, test, scales
    
    
def build_ens_dataset(var='t2m', train_end=2920, test_start=3285):

    # read data, weather variable: t2m or 500gh
    weather_variable = var
    data_gridded = get_ensemble_data(var=weather_variable)
    # shape: (3651, 81, 81, 50)

    # move the channel (ensemble members) dimension to the first
    data_gridded = np.moveaxis(data_gridded, 3, 1) # (3651, 50, 81, 81)
    # split into training, validation, and test sets (training: 2007-2014, validation: 2015, test: 2016)
    # training data
    x_train = data_gridded[:train_end, :, :, :]
    scales = compute_normalization(x_train)
    x_train_normalized = apply_normalization(x_train, scales)
    x_train_norm_ens = x_train_normalized.reshape((x_train.shape[0]*50, 1, 81, 81))
    
    # validation data
    x_val = data_gridded[train_end:test_start, :, :, :]
    x_val_normalized = apply_normalization(x_val, scales)
    x_val_norm_ens = x_val_normalized.reshape((x_val.shape[0]*50, 1, 81, 81))
    
    # test data
    x_test = data_gridded[test_start:, :, :, :]
    x_test_normalized = apply_normalization(x_test, scales)
    x_test_norm_ens = x_test_normalized.reshape((x_test.shape[0]*50, 1, 81, 81))
    
    # model inputs:
    train_data = torch.from_numpy(x_train_norm_ens)
    val_data = torch.from_numpy(x_val_norm_ens)
    test_data = torch.from_numpy(x_test_norm_ens)
    
    train = TensorDataset(train_data)
    val = TensorDataset(val_data)
    test = TensorDataset(test_data)
    
    return train, val, test, scales
    
        
def build_ens_f_dataset(var='t2m', train_end=2920, test_start=3285):

    # read data, weather variable: t2m or 500gh
    weather_variable = var
    data_gridded = get_ensemble_data(var=weather_variable)
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
    
    # model inputs:
    train_data = torch.from_numpy(x_train_norm_ens)
    val_data = torch.from_numpy(x_val_norm_ens)
    test_data = torch.from_numpy(x_test_norm_ens)
    
    train = TensorDataset(train_data)
    val = TensorDataset(val_data)
    test = TensorDataset(test_data)
    
    return train, val, test, scales
    
    
def build_ens_allvar_dataset(var='t2m', train_end=2920, test_start=3285):

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
    

