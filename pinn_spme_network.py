import numpy as np
import torch
import torch.nn as nn

import custom_lstm


def _fc_layers(cfg, output_size):
    layers = np.array(cfg['hidden_lstm'])
    for _ in range(cfg['layer_fc']):
        layers = np.append(layers, cfg['hidden_fc'])
    return np.append(layers, output_size)


def create_spme_nn_models(cfg, p):
    """Create SPM-style independent SPMe networks: anode, electrolyte, cathode."""
    if cfg.get('network_output', 'raw') != 'raw':
        raise ValueError("SPMe three-network architecture expects network_output: raw.")

    output_sizes = [p.n_cs_n, p.n_ce, p.n_cs_p]
    min_values = [
        torch.full((p.n_cs_n,), p.c_n_soc0),
        torch.full((p.n_ce,), p.ce_min),
        torch.full((p.n_cs_p,), p.c_p_soc100),
    ]
    max_values = [
        torch.full((p.n_cs_n,), p.c_n_soc100),
        torch.full((p.n_ce,), p.ce_max),
        torch.full((p.n_cs_p,), p.c_p_soc0),
    ]

    nn_models = nn.ModuleList()
    for output_size, min_value, max_value in zip(output_sizes, min_values, max_values):
        nn_models.append(
            custom_lstm.CustomLSTM(cfg, _fc_layers(cfg, output_size), p.input_size, max_value, min_value)
        )
    return nn_models


def pack_spme_state(c_s_n, c_e, c_s_p):
    return torch.cat([c_s_n, c_e, c_s_p], dim=1)


def predict_spme_state_raw(nn_models, u_seq):
    """Return raw SPMe concentrations after SPM-style CustomLSTM output scaling.

    Practical min/max bounds are only used inside each CustomLSTM to scale its
    output. The returned tensor is in mol/m3 and can be passed to SPMe physics.
    """
    c_s_n_raw = nn_models[0](u_seq)
    c_e_raw = nn_models[1](u_seq)
    c_s_p_raw = nn_models[2](u_seq)
    return pack_spme_state(c_s_n_raw, c_e_raw, c_s_p_raw)


def solid_concentration_to_theta(c_s_n_raw, c_s_p_raw, p):
    """Convert raw solid concentrations to stoichiometry for OCP/voltage use."""
    theta_n = c_s_n_raw / p.c_s_n_max
    theta_p = c_s_p_raw / p.c_s_p_max
    return theta_n, theta_p


def predict_spme_state(nn_models, u_seq):
    """Backward-compatible alias for code that already expects raw states."""
    return predict_spme_state_raw(nn_models, u_seq)
