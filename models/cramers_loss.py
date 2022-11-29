import torch
from torch import Tensor


def expected_dist(diff: Tensor, beta: float):
    # 1.e-7 term was removed! Maybe restore if needed for stability
    return torch.mean(torch.pow(torch.norm(diff, p=2, dim=[-1,-2]), beta), dim=1)


def loss_cramer_mse_old(output: Tensor, target: Tensor):
    """
    Computes (energy distance + mean square error)

    Parameters
    ----------
    y_true : tf tensor of shape (BATCH_SIZE, 50, 81, 81)
        True values: raw ensemble forecast fields.
    y_pred : tf tensor of shape (BATCH_SIZE, N_SAMPLES, 81, 81)
        N_SAMPLES = 50 by default.
        Predictive samples: reconstructed ensemble forecast fields.

    Returns
    -------
    sum of torch tensor of shape (BATCH_SIZE,)
        Scores.

    """

    y_pred = output
    y_true = target

    beta = 1
    n_samples_x = y_true.shape[1]
    n_samples_y = y_pred.shape[1]

    energy_1 = 0
    energy_2 = 0
    energy_3 = 0

    for i in range(n_samples_y):
        y_pred_i = torch.unsqueeze(y_pred[:, i, :, :], dim=1)
        energy_1 = energy_1 + expected_dist(y_pred_i - y_true, beta)

    for i in range(n_samples_x):
        y_true_i = torch.unsqueeze(y_true[:, i, :, :], dim=1)
        energy_2 = energy_2 + expected_dist(y_true_i - y_true, beta)

    for i in range(n_samples_y):
        y_pred_i = torch.unsqueeze(y_pred[:, i, :, :], dim=1)
        energy_3 = energy_3 + expected_dist(y_pred_i - y_pred, beta)

    energy_distance = energy_1 / (n_samples_y) - energy_2 / (2 * n_samples_x) - energy_3 / (
                2 * n_samples_y)

    y_true_mean = torch.mean(y_true, axis=1)  # (bs,81,81)
    y_pred_mean = torch.mean(y_pred, axis=1)  # (bs,81,81)

    rmse = torch.sqrt(torch.mean(torch.square(y_true_mean - y_pred_mean), axis=(-1, -2)) + 1e-7)

    combined_loss = energy_distance + rmse

    loss = torch.mean(combined_loss)

    return combined_loss


def energy(x, y, beta):
    return torch.mean(expected_dist(x.unsqueeze(1) - y.unsqueeze(2), beta), dim=1)


def loss_cramer_mse(output, target):
    """
    Computes (energy distance + mean square error)

    Parameters
    ----------
    y_true : tf tensor of shape (BATCH_SIZE, 50, 81, 81)
        True values: raw ensemble forecast fields.
    y_pred : tf tensor of shape (BATCH_SIZE, N_SAMPLES, 81, 81)
        N_SAMPLES = 50 by default.
        Predictive samples: reconstructed ensemble forecast fields.

    Returns
    -------
    sum of torch tensor of shape (BATCH_SIZE,)
        Scores.

    """

    y_pred = output
    y_true = target
    beta = 1
    energy_1 = energy(y_pred, y_true, beta)
    energy_2 = energy(y_true, y_true, beta)
    energy_3 = energy(y_pred, y_pred, beta)
    energy_distance = energy_1 - (energy_2 + energy_3) / 2.
    y_true_mean = torch.mean(y_true, dim=1)  # (bs,81,81)
    y_pred_mean = torch.mean(y_pred, dim=1)  # (bs,81,81)
    rmse = torch.norm(y_true_mean - y_pred_mean, p=2, dim=[-1, -2])
    combined_loss = energy_distance + rmse
    loss = torch.mean(combined_loss)
    return combined_loss