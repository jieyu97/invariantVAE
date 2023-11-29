# remember channels first
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from utils import device


class Ivae(nn.Module):

    def __init__(self,
                 dim_latent: int = 32,
                 n_samples: int = 50,
                 reg_weight = 0.05,
                 wd_weight = 0.1,
                 ed_weight = 0.1,
                 rmse_weight = 1,
                 n_nodes1: int = 4096,
                 activation = 'gelu'):
        super(Ivae, self).__init__()
        
        self.dim_latent = dim_latent
        self.n_samples = n_samples
        self.reg_weight = reg_weight
        self.wd_weight = wd_weight
        self.ed_weight = ed_weight
        self.rmse_weight = rmse_weight
        
        n_nodes2 = n_nodes1
        
        # input (bs, 50, 81, 81)
        self.flat = nn.Flatten(start_dim=2, end_dim=-1)
        self.unflat = nn.Unflatten(2, (81, 81))
        
        self.e_linear1 = nn.Linear(81*81, n_nodes1)
        self.e_linear2_1 = nn.Linear(n_nodes2, dim_latent)
        self.e_linear2_2 = nn.Linear(n_nodes2, dim_latent)
        
        self.d_linear1 = nn.Linear(dim_latent, n_nodes2)
        self.d_linear2 = nn.Linear(n_nodes1, 81*81)
        
        if activation == 'none':
            self.act = nn.Identity()
        elif activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'leakyrelu':
            self.act = nn.LeakyReLU(1e-3)
        elif activation == 'gelu':
            self.act = nn.GELU()
        
        self.avgpool = nn.AvgPool2d(kernel_size=(50,1), stride=1)
        self.maxpool = nn.MaxPool2d(kernel_size=(50,1), stride=1)
        
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()

    def sample(self, z_mean, z_log_var, **kwargs):
        # input: (bs, 32)
        bs = z_mean.shape[0]
        mean1 = torch.unsqueeze(z_mean, dim=1)
        mean = torch.repeat_interleave(mean1, self.n_samples, dim=1)
        log_var1 = torch.unsqueeze(z_log_var, dim=1)
        log_var = torch.repeat_interleave(log_var1, self.n_samples, dim=1)
        z = mean + torch.exp(0.5*log_var) * self.N.sample(torch.Size([bs, self.n_samples, self.dim_latent]))
        # output: (bs, 50, 32)
        return z
        
    def encoder(self, input: Tensor, **kwargs):
        x = input
        # (, 50, 81, 81)
        x = self.flat(x)
        # (, 50, 81*81)
        x = self.e_linear1(x)
        x = self.act(x)
        # (, 50, 4096)
        x_pool = self.avgpool(x)
        # (, 1, 4096)
        x_pool = torch.squeeze(x_pool, 1)
        # (, 4096)
        mu = self.e_linear2_1(x_pool)
        sigma = self.e_linear2_2(x_pool)
        # (, 1, 32)
        return mu, sigma

    def decoder(self, input: Tensor, **kwargs): 
        z = input
        # (, 50, 32)
        y = self.d_linear1(z)
        y = self.act(y)
        # (, 50, 4096)
        y = self.d_linear2(y)
        # (, 50, 81*81)
        y = self.unflat(y)
        # (, 50, 81, 81)
        return y
        
    def forward(self, input: Tensor, **kwargs):
        # input & recons shape: (bs, 1, 50, 81, 81) (bs, 50, 81, 81)
        x = input
        #bs = x.shape[0]
        z_mean, z_log_var = self.encoder(x)
        z = self.sample(z_mean, z_log_var)
        recons = self.decoder(z)
        return [recons, x, z, z_mean, z_log_var]  
        
    def generate(self, mean: Tensor, log_var: Tensor, **kwargs):
        # decoder: (bs, latent_dim) --> (bs, 1, n_samples, 81, 81)
        z = self.sample(mean, log_var)
        new_samples = self.decoder(z)
        return new_samples

    def loss_function(self, *args, **kwargs): 
        # input & recons: (bs, 50, 81, 81)
        recons = args[0]
        input = args[1]
        z_mean = args[3]
        z_log_var = args[4]
        
        recons_copy = recons # (bs,50,81,81)
        raw_copy = input # (bs,50,81,81)
        
        recons_mean = torch.mean(recons_copy, dim=2) # (bs,81,81)
        raw_mean = torch.mean(raw_copy, dim=2) # (bs,81,81)
        
        rmse = torch.sqrt(torch.mean(torch.square(raw_mean - recons_mean), dim=(-1,-2)))
        rmse = torch.mean(rmse)
    
        recons_loss = F.l1_loss(recons_mean, raw_mean)
        regularizer = self.kld(z_mean, z_log_var)
        
        #ed_grid = self.energy_distance_eachgrid(recons_copy, raw_copy)
        ed_grid = self.energy_distance(recons_copy, raw_copy)
        wd_ens = self.wasserstein(recons_copy, raw_copy)
        
        loss = self.reg_weight * regularizer + self.wd_weight * wd_ens + self.ed_weight * ed_grid + self.rmse_weight * rmse

        return {'loss': loss, 'RMSE': rmse, 'KLD': regularizer, 'ED': ed_grid, 'WD': wd_ens}
        
    def kld(self,
            mean: Tensor,
            log_var: Tensor) -> Tensor:
        mu = mean # (bs, latent_dim)
        mu = torch.flatten(mu, start_dim=1) # (bs, latent_dim)
        log_var = torch.flatten(log_var, start_dim=1) # (bs, latent_dim)
        kld = torch.mean(-0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim = 1), dim = 0)
        return kld
        
    def energy_grid(self, x, y):
        diff = torch.unsqueeze(x, 1) - torch.unsqueeze(y, 2) # (366, 50, 50, 81, 81)
        return torch.mean(torch.abs(diff), dim=(1,2))
    
    def energy_distance_eachgrid(self, output, target):
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
        y_pred = output
        y_true = target
        energy_1 = self.energy_grid(y_pred, y_true)
        energy_2 = self.energy_grid(y_true, y_true)
        energy_3 = self.energy_grid(y_pred, y_pred)
        #energy_distance = energy_1 - (energy_2 + energy_3) / 2.
        energy_distance = torch.sqrt(2.0 * energy_1 - (energy_2 + energy_3))
        
        loss_batch = torch.mean(energy_distance, dim=(1,2))
        loss = torch.mean(loss_batch)
        return loss
        
    def energy(self, x, y):
        diff = x.unsqueeze(1) - y.unsqueeze(2) # (366, 50, 50, 81, 81)
        energy = torch.mean(torch.norm(diff, p=2, dim=(-1,-2)), dim=(1,2))
        return energy
    
    def energy_distance(self, output, target):
        y_pred = output
        y_true = target
        energy_1 = self.energy(y_pred, y_true)
        energy_2 = self.energy(y_true, y_true)
        energy_3 = self.energy(y_pred, y_pred)
        energy_distance = torch.sqrt(2.0 * energy_1 - (energy_2 + energy_3))
        
        loss = torch.mean(energy_distance)
        return loss
    
    def wasserstein(self, output, target):
        y_pred = output
        y_true = target
        
        # implicit Sinkhorn
        y_true_flat = torch.reshape(y_true, (y_true.shape[0], y_true.shape[1], 81*81))
        y_pred_flat = torch.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1], 81*81))
        
        y_true_reshaped = y_true_flat.view(y_true.shape[0], y_true.shape[1], 1, -1)
        y_pred_reshaped = y_pred_flat.view(y_pred.shape[0], 1, y_pred.shape[1], -1)
    
        # Compute the pairwise distances between all rows using torch.norm()
        c = torch.square(torch.norm(y_true_reshaped - y_pred_reshaped, dim=-1))
        c_max = c.max()
        c_min = c.min()
        c_normalized = (c - c_min) / (c_max - c_min)
    
        a = torch.ones((y_true.shape[0], y_true.shape[1]), device=device, dtype=torch.float32) / y_true.shape[1]
        b = torch.ones((y_pred.shape[0], y_pred.shape[1]), device=device, dtype=torch.float32) / y_pred.shape[1]
        
        p = Sinkhorn.apply(c_normalized, a, b, 50, 1e-1)
    
        ot_wasserstein = (c*p).sum(dim=(-2,-1))
        ot_wasserstein_mean = torch.mean(ot_wasserstein)
        
        loss = torch.mean(ot_wasserstein_mean)
        return loss
    

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
