import torch
from torch import Tensor
from torch import nn
from utils import device


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


def rmse(output, target):
    y_pred = output
    y_true = target
    y_true_mean = torch.mean(y_true, dim=1)  # (bs,81,81)
    y_pred_mean = torch.mean(y_pred, dim=1)  # (bs,81,81)
    rmse = torch.sqrt(torch.mean(torch.square(y_true_mean - y_pred_mean), dim=(-1,-2)))
    return torch.mean(rmse)
    
    
def energy(x, y, beta):
    diff = x.unsqueeze(1) - y.unsqueeze(2)
    energy = torch.mean(torch.pow(torch.norm(diff, p=2, dim=[-1,-2]), beta), dim=[-1,-2])
    return energy


def energy_distance(output, target):
    y_pred = output
    y_true = target
    beta = 1
    energy_1 = energy(y_pred, y_true, beta)
    energy_2 = energy(y_true, y_true, beta)
    energy_3 = energy(y_pred, y_pred, beta)
    energy_distance = torch.sqrt(2.0 * energy_1 - (energy_2 + energy_3))
    
    loss = torch.mean(energy_distance)
    return loss
    

def kld(z_para):
    mu = z_para[0]
    log_var = z_para[1]
    mu = torch.flatten(mu, start_dim=1)
    log_var = torch.flatten(log_var, start_dim=1)
    kld = torch.mean(-0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim = 1), dim = 0)
    return kld
    

def rmse_kl(output, target, z_para, w_kl=0.5):
    y_pred = output
    y_true = target
    
    y_true_mean = torch.mean(y_true, dim=1)  # (bs,81,81)
    y_pred_mean = torch.mean(y_pred, dim=1)  # (bs,81,81)
    rmse = torch.sqrt(torch.mean(torch.square(y_true_mean - y_pred_mean), dim=(-1,-2)))
    
    mu = z_para[0]
    log_var = z_para[1]
    mu = torch.flatten(mu, start_dim=1)
    log_var = torch.flatten(log_var, start_dim=1)
    kld = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
    
    combined_loss = rmse + w_kl * kld
    loss = torch.mean(combined_loss)
    return loss
    

def energy_kl(output, target, z_para, w_kl=0.0): #0.05
    y_pred = output
    y_true = target
    beta = 1
    energy_1 = energy(y_pred, y_true, beta)
    energy_2 = energy(y_true, y_true, beta)
    energy_3 = energy(y_pred, y_pred, beta)
    energy_distance = torch.sqrt(2.0 * energy_1 - (energy_2 + energy_3))
    
    mu = z_para[0]
    log_var = z_para[1]
    mu = torch.flatten(mu, start_dim=1)
    log_var = torch.flatten(log_var, start_dim=1)
    kld = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
    
    combined_loss = energy_distance + w_kl * kld
    loss = torch.mean(combined_loss)
    return loss
    
    
def energy_rmse_kl(output, target, z_para, w_ed=0.05, w_kl=0.001):
    y_pred = output
    y_true = target
    beta = 1
    energy_1 = energy(y_pred, y_true, beta)
    energy_2 = energy(y_true, y_true, beta)
    energy_3 = energy(y_pred, y_pred, beta)
    energy_distance = torch.sqrt(2.0 * energy_1 - (energy_2 + energy_3))
    
    y_true_mean = torch.mean(y_true, dim=1)  # (bs,81,81)
    y_pred_mean = torch.mean(y_pred, dim=1)  # (bs,81,81)
    rmse = torch.sqrt(torch.mean(torch.square(y_true_mean - y_pred_mean), dim=(-1,-2)))
    
    mu = z_para[0]
    log_var = z_para[1]
    mu = torch.flatten(mu, start_dim=1)
    log_var = torch.flatten(log_var, start_dim=1)
    kld = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
    
    combined_loss = w_ed * energy_distance + rmse + w_kl * kld
    loss = torch.mean(combined_loss)
    return loss


def wasserstein(output, target):
    y_pred = output
    y_true = target
    
    # implicit Sinkhorn
    y_true_flat = torch.reshape(y_true, (y_true.shape[0], y_true.shape[1], 81*81))
    y_pred_flat = torch.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1], 81*81))
    
    y_true_reshaped = y_true_flat.view(y_true.shape[0], y_true.shape[1], 1, -1)
    y_pred_reshaped = y_pred_flat.view(y_pred.shape[0], 1, y_pred.shape[1], -1)

    # Compute the pairwise distances between all rows using torch.norm()
    c = torch.norm(y_true_reshaped - y_pred_reshaped, dim=-1)

    a = torch.ones((y_true.shape[0], y_true.shape[1]), device=device, dtype=torch.float32) / y_true.shape[1]
    b = torch.ones((y_pred.shape[0], y_pred.shape[1]), device=device, dtype=torch.float32) / y_pred.shape[1]
    
    p = Sinkhorn.apply(c, a, b, 100, 1e-3)

    ot_wasserstein = (c*p).sum(dim=(-2,-1))
    ot_wasserstein_mean = torch.mean(ot_wasserstein)
    
    loss = torch.mean(ot_wasserstein_mean)
    return loss
    

def energy_rmse_wd(output, target, w_ed=0.02, w_wd=0.02):
    y_pred = output
    y_true = target
    beta = 1
    energy_1 = energy(y_pred, y_true, beta)
    energy_2 = energy(y_true, y_true, beta)
    energy_3 = energy(y_pred, y_pred, beta)
    energy_distance = torch.sqrt(2.0 * energy_1 - (energy_2 + energy_3))
    
    y_true_mean = torch.mean(y_true, dim=1)  # (bs,81,81)
    y_pred_mean = torch.mean(y_pred, dim=1)  # (bs,81,81)
    rmse = torch.sqrt(torch.mean(torch.square(y_true_mean - y_pred_mean), dim=(-1,-2)))
    
    # implicit Sinkhorn
    y_true_flat = torch.reshape(y_true, (y_true.shape[0], y_true.shape[1], 81*81))
    y_pred_flat = torch.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1], 81*81))
    
    y_true_reshaped = y_true_flat.view(y_true.shape[0], y_true.shape[1], 1, -1)
    y_pred_reshaped = y_pred_flat.view(y_pred.shape[0], 1, y_pred.shape[1], -1)

    # Compute the pairwise distances between all rows using torch.norm()
    c = torch.norm(y_true_reshaped - y_pred_reshaped, dim=-1)

    a = torch.ones((y_true.shape[0], y_true.shape[1]), device=device, dtype=torch.float32) / y_true.shape[1]
    b = torch.ones((y_pred.shape[0], y_pred.shape[1]), device=device, dtype=torch.float32) / y_pred.shape[1]
    
    p = Sinkhorn.apply(c, a, b, 100, 1e-3)

    ot_wasserstein = (c*p).sum(dim=(-2,-1))
    ot_wasserstein_mean = torch.mean(ot_wasserstein)
    
    combined_loss = (rmse + w_ed * energy_distance + w_wd * ot_wasserstein_mean) / 3
    loss = torch.mean(combined_loss)
    return loss
    

def wasserstein_kl(output, target, z_para, w_kl=1):
    y_pred = output
    y_true = target
    
    mu = z_para[0]
    log_var = z_para[1]
    mu = torch.flatten(mu, start_dim=1)
    log_var = torch.flatten(log_var, start_dim=1)
    kld = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
    
    # implicit Sinkhorn
    y_true_flat = torch.reshape(y_true, (y_true.shape[0], y_true.shape[1], 81*81))
    y_pred_flat = torch.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1], 81*81))
    
    y_true_reshaped = y_true_flat.view(y_true.shape[0], y_true.shape[1], 1, -1)
    y_pred_reshaped = y_pred_flat.view(y_pred.shape[0], 1, y_pred.shape[1], -1)

    # Compute the pairwise distances between all rows using torch.norm()
    c = torch.norm(y_true_reshaped - y_pred_reshaped, dim=-1)

    a = torch.ones((y_true.shape[0], y_true.shape[1]), device=device, dtype=torch.float32) / y_true.shape[1]
    b = torch.ones((y_pred.shape[0], y_pred.shape[1]), device=device, dtype=torch.float32) / y_pred.shape[1]
    
    p = Sinkhorn.apply(c, a, b, 100, 1e-3)

    ot_wasserstein = (c*p).sum(dim=(-2,-1))
    ot_wasserstein_mean = torch.mean(ot_wasserstein)
    
    combined_loss = w_kl * kld + ot_wasserstein_mean
    loss = torch.mean(combined_loss)
    return loss
    
           
def energy_rmse_kl_wd(output, target, z_para, w_ed=1/40, w_kl=0, w_wd = 0.01):
    y_pred = output
    y_true = target
    beta = 1
    energy_1 = energy(y_pred, y_true, beta)
    energy_2 = energy(y_true, y_true, beta)
    energy_3 = energy(y_pred, y_pred, beta)
    energy_distance = torch.sqrt(2.0 * energy_1 - (energy_2 + energy_3))
    
    y_true_mean = torch.mean(y_true, dim=1)  # (bs,81,81)
    y_pred_mean = torch.mean(y_pred, dim=1)  # (bs,81,81)
    rmse = torch.sqrt(torch.mean(torch.square(y_true_mean - y_pred_mean), dim=(-1,-2)))
    
    mu = z_para[0]
    log_var = z_para[1]
    mu = torch.flatten(mu, start_dim=1)
    log_var = torch.flatten(log_var, start_dim=1)
    kld = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
    
    # implicit Sinkhorn
    y_true_flat = torch.reshape(y_true, (y_true.shape[0], y_true.shape[1], 81*81))
    y_pred_flat = torch.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1], 81*81))
    
    y_true_reshaped = y_true_flat.view(y_true.shape[0], y_true.shape[1], 1, -1)
    y_pred_reshaped = y_pred_flat.view(y_pred.shape[0], 1, y_pred.shape[1], -1)

    # Compute the pairwise distances between all rows using torch.norm()
    c = torch.norm(y_true_reshaped - y_pred_reshaped, dim=-1)

    a = torch.ones((y_true.shape[0], y_true.shape[1]), device=device, dtype=torch.float32) / y_true.shape[1]
    b = torch.ones((y_pred.shape[0], y_pred.shape[1]), device=device, dtype=torch.float32) / y_pred.shape[1]
    
    p = Sinkhorn.apply(c, a, b, 100, 1e-3)

    ot_wasserstein = (c*p).sum(dim=(-2,-1))
    ot_wasserstein_mean = torch.mean(ot_wasserstein)
    
    combined_loss = rmse + w_ed * energy_distance + w_kl * kld + w_wd * ot_wasserstein_mean
    loss = torch.mean(combined_loss)
    return loss
    