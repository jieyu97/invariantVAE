import math
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

from models.sinkhorn import Sinkhorn


class LayerNorm3d(nn.Module):

    def __init__(self, channels: int, eps: float = 1.e-5):
        super().__init__()
        self.eps = float(eps)
        self.register_parameter('bias', nn.Parameter(torch.zeros(1, channels, 1, 1, 1, dtype=torch.float32), requires_grad=True))
        self.register_parameter('scale', nn.Parameter(torch.ones(1, channels, 1, 1, 1, dtype=torch.float32), requires_grad=True))

    def forward(self, x: Tensor) -> Tensor:
        mu = torch.mean(x, dim=1, keepdim=True)
        sigma = torch.std(x, dim=1, keepdim=True, unbiased=False)
        out = self.scale * (x - mu) / (sigma + self.eps) + self.bias
        return out


class _CRPSMetric(nn.Module):

    def __init__(self):
        super().__init__()
        self._log_2pi = math.log(2. * np.pi)
        self._sqrt_2 = math.sqrt(2.)

    def _lower_phi(self, x: Tensor) -> Tensor:
        return torch.exp(- torch.square(x) / 2. - self._log_2pi / 2.)

    def _upper_phi(self, x: Tensor) -> Tensor:
        return (1. + torch.erf(x / self._sqrt_2)) * 0.5


class NormalMatchingCRPSLoss(_CRPSMetric):

    def __init__(self, sinkhorn_iterations: int = 100, entropy_weight: float = 2.e-3):
        super().__init__()
        self.sinkhorn = Sinkhorn(num_iterations=int(sinkhorn_iterations), entropy_weight=float(entropy_weight))
        self._inv_sqrt_pi = 1. / math.sqrt(np.pi)

    def forward(self, samples: Tuple[Tensor, Tensor], targets: Tensor) -> Tensor:
        # samples = (prediction_mean, prediction_log_standard_deviation)
        # shape of tensors: (batch, channels, h, w, member)
        locs, log_scales = samples
        # compute analytical CRPS of normal distribution parameterized by locs and log_scales for each grid location separately
        # formula from http://cran.nexr.com/web/packages/scoringRules/vignettes/crpsformulas.html
        y_red = (targets.unsqueeze(-2) - locs.unsqueeze(-1)) * torch.exp(- log_scales).unsqueeze(-1)
        crps_raw = self._raw_crps(y_red) * torch.exp(log_scales).unsqueeze(-1)
        # use field average as cost matrix
        cost = torch.mean(crps_raw, dim=(1, 2, 3)) 
        # compute normalization for cost matrix, as discussed
        cost_norm = torch.norm(cost, p='fro', dim=(-1, -2), keepdim=True)
        # compute weighting
        batch_size = len(targets)
        num_samples, num_targets = cost.shape[-2:]
        weighting = self.sinkhorn(
            cost / cost_norm,
            torch.ones(batch_size, num_samples, dtype=cost.dtype, device=cost.device) / num_samples,
            torch.ones(batch_size, num_targets, dtype=cost.dtype, device=cost.device) / num_targets
        )
        weighting = weighting.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        crps = torch.sum(crps_raw * weighting, dim=(-1,-2))
        # crps.shape = (batch, channel, h, w)
        return crps

    def _raw_crps(self, y: Tensor) -> Tensor:
        return y * (2. * self._upper_phi(y) - 1) + 2. * self._lower_phi(y) + self._inv_sqrt_pi


class SinkhornEnsembleNLL(nn.Module):

    def __init__(self, p=1., sinkhorn_iterations=100, entropy_weight=2.e-3, use_weight_grads=True):
        super().__init__()
        self.p = float(p)
        self.sinkhorn = Sinkhorn(num_iterations=int(sinkhorn_iterations), entropy_weight=float(entropy_weight))
        self.use_weight_grads = use_weight_grads

    def forward(self, predictions: Tuple[Tensor, Tensor], targets: Tensor):
        # shape same as above, predictions = samples
        batchsize = len(targets)
        num_dims = math.prod(targets.shape[1:4])
        device = targets.device
        mu, log_sigma = predictions
        num_predictions = mu.shape[-1]
        num_targets = targets.shape[-1]
        mu, log_sigma = mu[..., None], log_sigma[..., None]
        targets = targets.unsqueeze(-2)
        # use local standard deviations to rescale the deviations before cost computation
        deviation = torch.norm(torch.abs(targets - mu) * torch.exp(- log_sigma), p='fro', dim=(2, 3)).squeeze(1)
        # note that cost is measured per dimension: ... / num_dims
        cost = torch.pow(deviation, self.p) / num_dims
        cost_norm = torch.norm(cost, p=2, dim=(-1, -2), keepdim=True)
        # add normalization factor to compensate the effect of varying sigma
        log_det = torch.mean(log_sigma, dim=(1, 2, 3))
        # compute weighting from normalized cost
        target_marginals = torch.ones(batchsize, num_targets, dtype=torch.float32, device=device) / num_targets
        prediction_marginals = torch.ones(batchsize, num_predictions, dtype=torch.float32, device=device) / num_predictions
        weighting = self.sinkhorn(cost / cost_norm, target_marginals, prediction_marginals)
        if not self.use_weight_grads:
            weighting = weighting.detach()
        # compute negative log likelihood (per dimension) for multivariate normal distribution as sinkhorn-weighted average
        nll = torch.sum(weighting * (cost + log_det), dim=(1, 2)) / 2. # + const.; const. = math.log(2. * np.pi) / 2.
        # nll.shape = (batch_size,)
        return nll


