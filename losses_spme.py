import torch
import torch.nn as nn

from electrolyte_fvm import electrolyte_lithium_inventory


_mse = nn.MSELoss()


def rmse_loss(prediction, target, sample_weights=None):
    if sample_weights is None:
        return torch.sqrt(_mse(prediction, target))
    error_sq = (prediction - target) ** 2
    per_sample = error_sq.reshape(error_sq.shape[0], -1).mean(dim=1)
    weights = sample_weights.to(dtype=per_sample.dtype, device=per_sample.device).reshape(-1)
    denom = weights.sum().clamp_min(torch.finfo(per_sample.dtype).eps)
    return torch.sqrt((per_sample * weights).sum() / denom)


def voltage_loss(prediction, target, sample_weights=None):
    return rmse_loss(prediction, target, sample_weights=sample_weights)


def legacy_rmse_loss(prediction, target):
    return torch.sqrt(_mse(prediction, target))


def solid_lithium_loss(model, c_n_seq, c_p_seq, target_inventory):
    n_li_s = model.calc_solid_lithium(c_n_seq, c_p_seq)
    return rmse_loss(n_li_s, torch.full_like(n_li_s, target_inventory))


def electrolyte_lithium_loss(c_e_seq, p, target_inventory):
    if c_e_seq.dim() != 3:
        raise ValueError('c_e_seq must have shape [batch, n_ce, time].')
    inventories = []
    for time_idx in range(c_e_seq.shape[-1]):
        inventories.append(electrolyte_lithium_inventory(c_e_seq[:, :, time_idx], p))
    n_li_e = torch.stack(inventories, dim=1)
    return rmse_loss(n_li_e, torch.full_like(n_li_e, target_inventory))


def bounds_loss(c_n_seq, c_e_seq, c_p_seq, p):
    loss_n_low = torch.relu(-c_n_seq)
    loss_n_high = torch.relu(c_n_seq - p.c_s_n_max)
    loss_p_low = torch.relu(-c_p_seq)
    loss_p_high = torch.relu(c_p_seq - p.c_s_p_max)
    loss_e_low = torch.relu(p.ce_min - c_e_seq)
    loss_e_high = torch.relu(c_e_seq - p.ce_max)
    penalties = torch.cat([
        loss_n_low.reshape(loss_n_low.shape[0], -1),
        loss_n_high.reshape(loss_n_high.shape[0], -1),
        loss_p_low.reshape(loss_p_low.shape[0], -1),
        loss_p_high.reshape(loss_p_high.shape[0], -1),
        loss_e_low.reshape(loss_e_low.shape[0], -1),
        loss_e_high.reshape(loss_e_high.shape[0], -1),
    ], dim=1)
    return penalties.mean()


def electrolyte_interface_optional_loss(c_e_seq, p):
    # The conservative FVM operator uses shared interface fluxes, so no extra
    # interface continuity penalty is needed by default.
    return torch.zeros(1, dtype=c_e_seq.dtype, device=c_e_seq.device).squeeze()
