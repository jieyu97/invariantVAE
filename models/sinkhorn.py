import torch
from torch import nn


def sinkhorn_unrolled(c, a, b, num_sink, lambd_sink):
    """
    An implementation of a Sinkhorn layer with Automatic Differentiation (AD).
    The format of input parameters and outputs is equivalent to the 'Sinkhorn' module below.
    """
    log_p = -c / lambd_sink
    log_a = torch.log(a).unsqueeze(dim=-1)
    log_b = torch.log(b).unsqueeze(dim=-2)
    for _ in range(num_sink):
        log_p = log_p - (torch.logsumexp(log_p, dim=-2, keepdim=True) - log_b)
        log_p = log_p - (torch.logsumexp(log_p, dim=-1, keepdim=True) - log_a)
    p = torch.exp(log_p)
    return p


class _SinkhornFunction(torch.autograd.Function):
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
        grad_b = torch.cat((grad_ab[..., m:, :], torch.zeros(batch_shape + [1, 1], device=grad_ab.device, dtype=torch.float32)), dim=-2)
        U = grad_a + grad_b.transpose(-2, -1)
        grad_p -= p * U
        grad_a = -ctx.lambd_sink * grad_a.squeeze(dim=-1)
        grad_b = -ctx.lambd_sink * grad_b.squeeze(dim=-1)
        return grad_p, grad_a, grad_b, None, None, None


class Sinkhorn(nn.Module):

    def __init__(self, num_iterations: int = 1, entropy_weight: float = 2.e-3):
        super(Sinkhorn, self).__init__()
        self.num_iterations = num_iterations
        self.entropy_weight = entropy_weight

    def forward(self, c, a, b):
        return _SinkhornFunction.apply(c, a, b, self.num_iterations, self.entropy_weight)


def _test():
    b, n, h, w = 1, 50, 81, 81
    a = torch.randn(b, n, h, w) * 0.01 + torch.randn(n)[None, :, None, None]
    b = torch.randn(b, n, h, w) * 0.01 + torch.randn(n)[None, :, None, None]
    marginals = torch.ones(1, n) / n

    def dist_mat(x1, x2):
        return torch.mean(torch.abs(x1.unsqueeze(1) - x2.unsqueeze(2)) ** 2, dim=(-1, -2))

    c = dist_mat(a, b)

    sinkhorn = Sinkhorn(num_iterations=1000, entropy_weight=0.0001)

    p = sinkhorn(c, marginals, marginals)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.pcolor(n * p[0].data.cpu().numpy(), vmin=0., vmax=1.)
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    _test()
