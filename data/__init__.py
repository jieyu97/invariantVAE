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