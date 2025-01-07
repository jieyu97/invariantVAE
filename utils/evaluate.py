import numpy as np
# import geomloss
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from scipy.stats import wasserstein_distance, energy_distance


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
    mean_diff_abs = np.abs(y_true_mean - y_pred_mean)
    mean_diff = np.mean(mean_diff_abs)
    return mean_diff


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
    std_true = np.mean(y_true_mean)
    std_pred = np.mean(y_pred_mean)
    return std_true, std_pred
    
    
def expected_dist(diff, beta):
    # 1.e-7 term was removed! Maybe restore if needed for stability
    return np.mean(np.power(np.linalg.norm(diff, axis=(-1,-2)), beta), axis=1)


def energy(x, y, beta):
    return np.mean(expected_dist(np.expand_dims(x, 1) - np.expand_dims(y, 2), beta), axis=1)


def energy_distance_allgrids(y_true, y_pred):
    """
    Computes energy distance of full ensemble fields

    Parameters
    ----------
    y_true : numpy array of shape (366, 50, 81, 81)
    y_recons : numpy array of shape (366, 50, 81, 81)

    Returns
    -------
    Vector: Energy distance of full ensemble fields

    """
    beta = 1
    energy_1 = energy(y_pred, y_true, beta)
    energy_2 = energy(y_true, y_true, beta)
    energy_3 = energy(y_pred, y_pred, beta)
    ed = 2 * energy_1 - (energy_2 + energy_3)
    return np.sqrt(ed)



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
    y_true_torch = torch.from_numpy(y_true)
    y_pred_torch = torch.from_numpy(y_pred)
    wasserstein_d = wasserstein(y_true_torch, y_pred_torch)
    
    return wasserstein_d


def scipy_function_wd_ed(y_true, y_pred):
    """
    Parameters
    ----------
    y_true : numpy array of shape (366, 50, 81, 81)
    y_recons : numpy array of shape (366, 50, 81, 81)
    
    """
    y_true_new = np.moveaxis(y_true, 1, 3)
    y_pred_new = np.moveaxis(y_pred, 1, 3)
    y_true_flat = np.reshape(y_true_new, (y_true_new.shape[0]*81*81, 50))
    y_pred_flat = np.reshape(y_pred_new, (y_pred_new.shape[0]*81*81, 50))
    
    wasserstein_1d = map(wasserstein_distance, y_true_flat, y_pred_flat)
    wasserstein_1d_list = list(wasserstein_1d)
    wasserstein_1d_array = np.array(wasserstein_1d_list)
    wd = np.mean(wasserstein_1d_array)
    
    energy_distance_1d = map(energy_distance, y_true_flat, y_pred_flat)
    energy_distance_1d_list = list(energy_distance_1d)
    energy_distance_1d_array = np.array(energy_distance_1d_list)
    ed = np.mean(energy_distance_1d_array)
    
    return wd, ed
    
    
    
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
    mae = rmse_of_mean_ensemble(y_true, y_pred)
    std_true, std_pred = mean_std_ensemble(y_true, y_pred)
    energy_distance_all = energy_distance_allgrids(y_true, y_pred)
    wasserstein_distance = wasserstein_distance_full_ensemble(y_true, y_pred)
    wasserstein_distance = wasserstein_distance.numpy()
    ed_ndim = np.mean(energy_distance_all)
    wd_ndim = np.mean(wasserstein_distance)
    wd_all, ed_all = scipy_function_wd_ed(y_true, y_pred)
    wd_1dim_fun = np.mean(wd_all)
    ed_1dim_fun = np.mean(ed_all)
    
    return mae, std_true, std_pred, ed_1dim_fun, ed_ndim, wd_1dim_fun, wd_ndim


def wasserstein(output, target):
    y_pred = output
    y_true = target
    
    # implicit Sinkhorn
    y_true_flat = torch.reshape(y_true, (y_true.shape[0], y_true.shape[1], 81*81))
    y_pred_flat = torch.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1], 81*81))
    
    y_true_reshaped = y_true_flat.view(y_true.shape[0], y_true.shape[1], 1, -1)
    y_pred_reshaped = y_pred_flat.view(y_pred.shape[0], 1, y_pred.shape[1], -1)

    # Compute the pairwise distances between all rows using torch.norm()
    c = torch.square(torch.norm(y_true_reshaped - y_pred_reshaped, dim=-1))
    # torch Fro norm: RMSE --> cost function: MSE

    a = torch.ones((y_true.shape[0], y_true.shape[1]), dtype=torch.float32) / y_true.shape[1]
    b = torch.ones((y_pred.shape[0], y_pred.shape[1]), dtype=torch.float32) / y_pred.shape[1]
    
    p = Sinkhorn.apply(c, a, b, 100, 1e-2)

    ot_wasserstein = (c*p).sum(dim=(-2,-1))
    
    return ot_wasserstein
    
    

class Sinkhorn(torch.autograd.Function):
    """
    An implementation of a Sinkhorn layer with our custom backward module, based on implicit differentiation
    :param c: input cost matrix, size [*,m,n], where * are arbitrarily many batch dimensions
    :param a: first input marginal, size [*,m]
    :param b: second input marginal, size [*,n]
    :param num_sink: number of Sinkhorn iterations
    :param lambd_sink: entropy regularization weight
    :return: optimized soft permutation matrix
    """
    @staticmethod
    def forward(ctx, c, a, b, num_sink, lambd_sink):
        log_p = -c / lambd_sink
        log_a = torch.log(a).unsqueeze(dim=-1)
        log_b = torch.log(b).unsqueeze(dim=-2)
        for _ in range(num_sink):
            log_p -= (torch.logsumexp(log_p, dim=-2, keepdim=True) - log_b)
            log_p -= (torch.logsumexp(log_p, dim=-1, keepdim=True) - log_a)
        p = torch.exp(log_p)

        ctx.save_for_backward(p, torch.sum(p, dim=-1), torch.sum(p, dim=-2))
        ctx.lambd_sink = lambd_sink
        return p

    @staticmethod
    def backward(ctx, grad_p):
        p, a, b = ctx.saved_tensors

        m, n = p.shape[-2:]
        batch_shape = list(p.shape[:-2])

        grad_p *= -1 / ctx.lambd_sink * p
        K = torch.cat((torch.cat((torch.diag_embed(a), p), dim=-1),
                       torch.cat((p.transpose(-2, -1), torch.diag_embed(b)), dim=-1)), dim=-2)[..., :-1, :-1]
        t = torch.cat((grad_p.sum(dim=-1), grad_p[..., :, :-1].sum(dim=-2)), dim=-1).unsqueeze(-1)
        grad_ab = torch.linalg.solve(K, t)
        grad_a = grad_ab[..., :m, :]
        grad_b = torch.cat((grad_ab[..., m:, :], torch.zeros(batch_shape + [1, 1], device=device, dtype=torch.float32)), dim=-2)
        U = grad_a + grad_b.transpose(-2, -1)
        grad_p -= p * U
        grad_a = -ctx.lambd_sink * grad_a.squeeze(dim=-1)
        grad_b = -ctx.lambd_sink * grad_b.squeeze(dim=-1)
        return grad_p, grad_a, grad_b, None, None, None









