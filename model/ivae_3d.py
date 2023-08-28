# remember channels first
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from utils import device

        
class IvaeEncoder(nn.Module):

    def __init__(self,
                 latent_dim: int, res_type):
        super(IvaeEncoder, self).__init__()

        self.latent_dim = latent_dim
        self.res_type = res_type # bottleneck or basic
        
        self.conv1 = nn.Conv3d(1, 32, (1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.conv2 = nn.Conv3d(32, 64, (1, 3, 3), stride=(1, 3, 3), padding=(0, 0, 0))
        self.conv3 = nn.Conv3d(64, 128, (1, 3, 3), stride=(1, 3, 3), padding=(0, 0, 0))
        self.conv4 = nn.Conv3d(128, 256, (1, 3, 3), stride=(1, 3, 3), padding=(0, 0, 0))
        self.avgpool = nn.AvgPool3d((50, 1, 1), stride=(1, 1, 1))
        self.maxpool = nn.MaxPool3d((50, 1, 1), stride=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(32)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(128)
        self.bn4 = nn.BatchNorm3d(256)
        self.act = nn.LeakyReLU(1e-3)
        self.flat = nn.Flatten()
        self.latent1 = nn.Linear(256*3*3, latent_dim)
        self.latent2 = nn.Linear(256*3*3, latent_dim)
        
        self.resblock_e1 = ResBlock(self.res_type, 32, 32)
        self.resblock_e2 = ResBlock(self.res_type, 64, 64)
        self.resblock_e3 = ResBlock(self.res_type, 128, 128)
        self.resblock_e4 = ResBlock(self.res_type, 256, 256)


    def forward(self, x: Tensor):
        # x shape: (bs, 1, 50, 81, 81)
        x_encode = self.conv1(x)
        x_encode = self.bn1(x_encode)
        x_encode = self.act(x_encode)
        x_encode = self.resblock_e1(x_encode)
        # (, 32, 50, 81, 81)
        x_encode = self.conv2(x_encode)
        x_encode = self.bn2(x_encode)
        x_encode = self.act(x_encode)
        x_encode = self.resblock_e2(x_encode)
        # (, 64, 50, 27, 27)
        x_encode = self.conv3(x_encode)
        x_encode = self.bn3(x_encode)
        x_encode = self.act(x_encode)
        x_encode = self.resblock_e3(x_encode)
        # (, 128, 50, 9, 9)
        x_encode = self.avgpool(x_encode)
        # (, 128, 1, 9, 9)
        x_encode = self.conv4(x_encode)
        x_encode = self.bn4(x_encode)
        x_encode = self.act(x_encode)
        x_encode = self.resblock_e4(x_encode)
        # (, 256, 1, 3, 3)
        x_encode = self.flat(x_encode)
        # low-dimensional latent representation:
        z_mean = self.latent1(x_encode)
        z_log_var = self.latent2(x_encode)
        
        return z_mean, z_log_var
        

class IvaeDecoder(nn.Module):

    def __init__(self,
                 latent_dim: int, res_type):
        super(IvaeDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.res_type = res_type # bottleneck or basic
        
        self.deconv1 = nn.ConvTranspose3d(256, 128, (1, 3, 3), stride=(1, 3, 3), padding=(0, 0, 0))
        self.deconv2 = nn.ConvTranspose3d(128, 64, (1, 3, 3), stride=(1, 3, 3), padding=(0, 0, 0))
        self.deconv3 = nn.ConvTranspose3d(64, 32, (1, 3, 3), stride=(1, 3, 3), padding=(0, 0, 0))
        self.conv1 = nn.Conv3d(256, 256, (1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.conv2 = nn.Conv3d(128, 128, (1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.conv3 = nn.Conv3d(64, 64, (1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.conv4 = nn.Conv3d(32, 32, (1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.conv5 = nn.Conv3d(32, 1, (1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.bn1_1 = nn.BatchNorm3d(256)
        self.bn1_2 = nn.BatchNorm3d(128)
        self.bn2_1 = nn.BatchNorm3d(128)
        self.bn2_2 = nn.BatchNorm3d(64)
        self.bn3_1 = nn.BatchNorm3d(64)
        self.bn3_2 = nn.BatchNorm3d(32)
        self.bn4 = nn.BatchNorm3d(32)
        self.act = nn.LeakyReLU(1e-3)
        self.act_final = nn.Sigmoid()
        self.linear = nn.Linear(latent_dim, 256*3*3)
        self.unflat = nn.Unflatten(2, (256, 3, 3))
        
        self.resblock_d0 = ResBlock(self.res_type, 256, 256)
        self.resblock_d1 = ResBlock(self.res_type, 128, 128)
        self.resblock_d2 = ResBlock(self.res_type, 64, 64)
        self.resblock_d3 = ResBlock(self.res_type, 32, 32)
        

    def forward(self, z: Tensor):
        # z shape: (bs, 50, 32)
        y_decode = self.linear(z)
        # (, 50, 256*3*3)
        y_decode = self.unflat(y_decode)
        # (, 50, 256, 3, 3)
        y_decode = torch.movedim(y_decode, 2, 1)
        # (, 256, 50, 3, 3)
        y_decode = self.resblock_d0(y_decode)
        #y_decode = self.conv1(y_decode)
        #y_decode = self.bn1_1(y_decode)
        y_decode = self.act(y_decode)
        y_decode = self.deconv1(y_decode)
        y_decode = self.bn1_2(y_decode)
        y_decode = self.act(y_decode)
        # (, 128, 50, 9, 9)
        y_decode = self.resblock_d1(y_decode)
        #y_decode = self.conv2(y_decode)
        #y_decode = self.bn2_1(y_decode)
        y_decode = self.act(y_decode)
        y_decode = self.deconv2(y_decode)
        y_decode = self.bn2_2(y_decode)
        y_decode = self.act(y_decode)
        # (, 64, 50, 27, 27)
        y_decode = self.resblock_d2(y_decode)
        #y_decode = self.conv3(y_decode)
        #y_decode = self.bn3_1(y_decode)
        y_decode = self.act(y_decode)
        y_decode = self.deconv3(y_decode)
        y_decode = self.bn3_2(y_decode)
        y_decode = self.act(y_decode)
        # (, 32, 50, 81, 81)
        y_decode = self.resblock_d3(y_decode)
        #y_decode = self.conv4(y_decode)
        #y_decode = self.bn4(y_decode)
        y_decode = self.act(y_decode)
        y_decode = self.conv5(y_decode)
        y = self.act_final(y_decode)
        # (, 1, 50, 81, 81)
        
        return y

        
class Ivae(nn.Module):

    def __init__(self,
                 dim_latent: int = 32,
                 n_samples: int = 50,
                 reg_weight = 0.05,
                 ed_weight = 1,
                 wd_weight = 0.1,
                 res_type = 'basic'):
        super(Ivae, self).__init__()
        
        self.dim_latent = dim_latent
        self.n_samples = n_samples
        self.reg_weight = reg_weight
        self.ed_weight = ed_weight
        self.wd_weight = wd_weight
        self.res_type = res_type # bottleneck or basic
        
        self.encoder = IvaeEncoder(latent_dim = self.dim_latent, res_type = self.res_type)
        self.decoder = IvaeDecoder(latent_dim = self.dim_latent, res_type = self.res_type)
        
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
        
        return z
        

    def forward(self, x):
        # input & recons shape: (bs, 1, 50, 81, 81)
        input = x
        z_mean, z_log_var = self.encoder(x)
        z = self.sample(z_mean, z_log_var)
        recons = self.decoder(z)
        
        return recons, input, z, z_mean, z_log_var


    def generate(self, z_mean, z_log_var, **kwargs):
        # decoder: (bs, latent_dim), (bs, latent_dim)
        #           --> (bs, n_samples, latent_dim)
        #           --> (bs, 1, n_samples, 81, 81)
        z = self.sample(z_mean, z_log_var)
        new_samples = self.decoder(z)

        return new_samples


    def loss_function(self, *args, **kwargs): 
        # input & recons: (bs, 1, 50, 81, 81)
        recons = args[0]
        input = args[1]
        z_mean = args[3]
        z_log_var = args[4]
        
        recons_loss = F.mse_loss(recons, input)
        regularizer = self.kld(z_mean, z_log_var)
        
        recons_copy = torch.squeeze(recons, dim=1)
        raw_copy = torch.squeeze(input, dim=1)
        
        ed_grid = self.energy_distance_eachgrid(recons_copy, raw_copy)
        wd_ens = self.wasserstein(recons_copy, raw_copy)
        
        loss = self.reg_weight * regularizer + self.ed_weight * ed_grid + self.wd_weight * wd_ens# + recons_loss

        return {'loss': loss, 'MSE': recons_loss, 'KLD': regularizer, 'ED': ed_grid, 'WD': wd_ens}
        

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
	# energy distance at each grid: 1 dimension with 50 samples
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
	# energy distance for all grids: 81*81 dimensions with 50 samples
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
        c = torch.norm(y_true_reshaped - y_pred_reshaped, dim=-1)
    
        a = torch.ones((y_true.shape[0], y_true.shape[1]), device=device, dtype=torch.float32) / y_true.shape[1]
        b = torch.ones((y_pred.shape[0], y_pred.shape[1]), device=device, dtype=torch.float32) / y_pred.shape[1]
        
        p = Sinkhorn.apply(c, a, b, 100, 1e-3)
    
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



class ResBlock(nn.Module):

    def __init__(
        self,
        restype: str,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1
    ) -> None:
        super().__init__()
    
        self.restype = restype
            
        self.conv11 = nn.Conv3d(in_channels, out_channels, (1,1,1), stride=(1,1,1), padding=(0, 0, 0), bias=False)
        self.bn11 = nn.BatchNorm3d(out_channels)
        self.conv12 = nn.Conv3d(out_channels, out_channels, (1,3,3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False)
        self.bn12 = nn.BatchNorm3d(out_channels)
        self.conv13 = nn.Conv3d(out_channels, out_channels, (1,1,1), stride=(1,1,1), padding=(0, 0, 0), bias=False)
        self.bn13 = nn.BatchNorm3d(out_channels * expansion)
        
        self.conv21 = nn.Conv3d(in_channels, out_channels, (1,3,3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False)
        self.bn21 = nn.BatchNorm3d(out_channels)
        self.conv22 = nn.Conv3d(out_channels, out_channels, (1,3,3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False)
        self.bn22 = nn.BatchNorm3d(out_channels * expansion)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: Tensor) -> Tensor:
        identity = x

        if self.restype == 'bottleneck':
            out = self.conv11(x)
            out = self.bn11(out)
            out = self.relu(out)
    
            out = self.conv12(out)
            out = self.bn12(out)
            out = self.relu(out)
    
            out = self.conv13(out)
            out = self.bn13(out)
            
        elif self.restype == 'basic':
        
            out = self.conv21(x)
            out = self.bn21(out)
            out = self.relu(out)
    
            out = self.conv22(out)
            out = self.bn22(out)

        out += identity
        out = self.relu(out)

        return out

