import numpy as np
import geomloss
import torch


def rmse_of_mean_ensemble(y_true, y_pred):
    """
    Computes RMSE of mean ensemble fields at each day of the test set year 2016

    Parameters
    ----------
    y_true : numpy array of shape (366, 50, 81, 81)
    y_recons : numpy array of shape (366, 50, 81, 81)

    Returns
    -------
    Vector: RMSE of mean ensemble fields at each day

    """
    y_true_mean = np.mean(y_true, axis=1)
    y_pred_mean = np.mean(y_pred, axis=1)
    rmse1 = np.sqrt(np.mean(np.square(y_true_mean - y_pred_mean), axis=0))
    rmse2 = np.sqrt(np.mean(np.square(y_true_mean - y_pred_mean), axis=(-1,-2)))
    return rmse1, rmse2


def mean_std_ensemble(y_true, y_pred):
    """
    Computes RMSE of mean ensemble fields at each day of the test set year 2016

    Parameters
    ----------
    y_true : numpy array of shape (366, 50, 81, 81)
    y_recons : numpy array of shape (366, 50, 81, 81)

    Returns
    -------
    Vector: RMSE of mean ensemble fields at each day

    """
    y_true_mean = np.std(y_true, axis=1)
    y_pred_mean = np.std(y_pred, axis=1)
    std1d = np.mean(y_true_mean, axis=0)
    std1g = np.mean(y_true_mean, axis=(-1,-2))
    std2d = np.mean(y_pred_mean, axis=0)
    std2g = np.mean(y_pred_mean, axis=(-1,-2))
    return std1d, std2d, std1g, std2g
    
    
def expected_dist(diff, beta):
    # 1.e-7 term was removed! Maybe restore if needed for stability
    return np.mean(np.power(np.linalg.norm(diff, axis=(-1,-2)), beta), axis=1)


def energy(x, y, beta):
    return np.mean(expected_dist(np.expand_dims(x, 1) - np.expand_dims(y, 2), beta), axis=1)


def energy_distance_allgrids(y_true, y_pred):
    """
    Computes Cramer/energy distance of full ensemble fields

    Parameters
    ----------
    y_true : numpy array of shape (366, 50, 81, 81)
    y_recons : numpy array of shape (366, 50, 81, 81)

    Returns
    -------
    Vector: Cramer/energy distance of full ensemble fields

    """
    beta = 1
    energy_1 = energy(y_pred, y_true, beta)
    energy_2 = energy(y_true, y_true, beta)
    energy_3 = energy(y_pred, y_pred, beta)
    energy_distance = energy_1 - (energy_2 + energy_3) / 2.
    return energy_distance


def energy_grid(x, y):
    return np.mean(np.abs(np.expand_dims(x, 1) - np.expand_dims(y, 2)), axis=(1,2))


def energy_distance_eachgrid(y_true, y_pred):
    """
    Computes Cramer/energy distance of full ensemble fields

    Parameters
    ----------
    y_true : numpy array of shape (366, 50, 81, 81)
    y_recons : numpy array of shape (366, 50, 81, 81)

    Returns
    -------
    Vector: Cramer/energy distance of full ensemble fields

    """
    energy_1 = energy_grid(y_pred, y_true)
    energy_2 = energy_grid(y_true, y_true)
    energy_3 = energy_grid(y_pred, y_pred)
    energy_distance = energy_1 - (energy_2 + energy_3) / 2.
    return energy_distance


def wasserstein_distance_full_ensemble(y_true, y_pred):
    """
    Computes Wasserstein distance of full ensemble fields

    Parameters
    ----------
    y_true : numpy array of shape (366, 50, 81, 81)
    y_recons : numpy array of shape (366, 50, 81, 81)

    Returns
    -------
    Vector: Wasserstein distance of full ensemble fields

    """
    y_true_flat = np.reshape(y_true, (y_true.shape[0], y_true.shape[1], 81*81))
    y_pred_flat = np.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1], 81*81))
    y_true_torch = torch.from_numpy(y_true_flat)
    y_pred_torch = torch.from_numpy(y_pred_flat)
    y_true_torch = y_true_torch.double()
    y_pred_torch = y_pred_torch.double()
    loss_sinkhorn = geomloss.SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.8)
    wasserstein_distance = loss_sinkhorn(y_true_torch, y_pred_torch)
    return wasserstein_distance


def evaluation_metric2(y_true, y_pred):
    """
    Computes 2 metrics

    Parameters
    ----------
    y_true : numpy array of shape (366, 50, 81, 81)
    y_recons : numpy array of shape (366, 50, 81, 81)

    Returns
    -------
    List

    """
    rmse = rmse_of_mean_ensemble(y_true, y_pred)
    energy_distance = cramer_full_ensemble_grid(y_true, y_pred)
    
    return rmse, energy_distance
    
    
def evaluation_metric(y_true, y_pred):
    """
    Computes all 3 metrics

    Parameters
    ----------
    y_true : numpy array of shape (366, 50, 81, 81)
    y_recons : numpy array of shape (366, 50, 81, 81)

    Returns
    -------
    List

    """
    rmse1, rmse2 = rmse_of_mean_ensemble(y_true, y_pred)
    energy_distance_all = energy_distance_allgrids(y_true, y_pred)
    energy_distance_grid = energy_distance_eachgrid(y_true, y_pred)
    wasserstein_distance = wasserstein_distance_full_ensemble(y_true, y_pred)
    
    return rmse2, energy_distance_all, energy_distance_grid, wasserstein_distance
