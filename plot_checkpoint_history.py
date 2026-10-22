#!/usr/bin/env python3
"""Plot PINN-SPMe training history stored in a checkpoint.

The training checkpoint stores a ``history`` list with one dictionary per epoch.
This script extracts that history and saves standalone figures for losses,
degradation parameters, lithium inventory diagnostics, and timing if present.
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

import init_params as ip
import train_pinn_spme
from pinn_spme_network import create_spme_nn_models
from spme_model import SPMeModel


def _load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _with_config_override(checkpoint, config_path):
    if config_path is None:
        return checkpoint
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}
    merged_config = dict(checkpoint.get("config", {}) or {})
    merged_config.update(config)
    updated = dict(checkpoint)
    updated["config"] = merged_config
    return updated


def _history_frame(checkpoint):
    history = checkpoint.get("history", [])
    if not history:
        raise ValueError("Checkpoint does not contain non-empty history.")
    df = pd.DataFrame(history)
    if "epoch" not in df.columns:
        df.insert(0, "epoch", np.arange(1, len(df) + 1))
    df = df.sort_values("epoch")
    return df


def _scalar(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        return float(value.detach().cpu().reshape(-1)[0])
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _constant_column(df, column, value):
    if value is not None and column not in df.columns:
        df[column] = float(value)


def _add_reference_degradation_columns(df, checkpoint):
    """Add true/reference columns derivable from the checkpoint state.

    The training loop stores true values for delta_sei, epsilon_e_n, and nLi_sei.
    It stores estimates, but not true/reference values, for nLi_s, nLi_e, and
    R_f_n. Those can be reconstructed from the same model buffers used by
    SPMeModel when they are available in the checkpoint.
    """
    cfg = checkpoint.get("config", {}) or {}
    model_state = checkpoint.get("model_state", {}) or {}

    delta_true = None
    if "delta_sei_true_epoch" in df.columns:
        values = pd.to_numeric(df["delta_sei_true_epoch"], errors="coerce").dropna()
        if not values.empty:
            delta_true = float(values.iloc[-1])
    if delta_true is None:
        delta_true = _scalar(cfg.get("delta_sei_true"))
    _constant_column(df, "delta_sei_true_epoch", delta_true)

    epsilon_true = None
    if "epsilon_e_n_true_epoch" in df.columns:
        values = pd.to_numeric(df["epsilon_e_n_true_epoch"], errors="coerce").dropna()
        if not values.empty:
            epsilon_true = float(values.iloc[-1])
    if epsilon_true is None:
        epsilon_true = _scalar(cfg.get("epsilon_e_n_true"))
    _constant_column(df, "epsilon_e_n_true_epoch", epsilon_true)

    c_li_sei = _scalar(model_state.get("C_li_sei_value"))
    nli_total = _scalar(model_state.get("nLi_total"))
    nli_e_init = _scalar(model_state.get("nLi_e_init"))
    nli_e_coeff = _scalar(model_state.get("nLi_e_delta_sei_coeff"))
    r_f_n_base = _scalar(cfg.get("R_f_n_base"))
    if r_f_n_base is None:
        r_f_n_base = 1.0e-3
    kappa_sei = _scalar(cfg.get("kappa_sei"))
    if kappa_sei is None:
        kappa_sei = 5.0e-6

    if delta_true is None:
        return df

    if c_li_sei is not None:
        nli_sei_true = c_li_sei * delta_true
        _constant_column(df, "nLi_sei_true_epoch", nli_sei_true)
        if nli_total is not None:
            _constant_column(df, "nLi_s_true_epoch", nli_total - nli_sei_true)

    if nli_e_init is not None and nli_e_coeff is not None:
        _constant_column(df, "nLi_e_true_epoch", nli_e_init - nli_e_coeff * delta_true)

    _constant_column(df, "R_f_n_true_epoch", r_f_n_base + delta_true / kappa_sei)
    return df


def _numeric_series(df, column):
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    return np.where(np.isfinite(values), values, np.nan)


def _existing(df, columns):
    return [column for column in columns if column in df.columns]


def _save(fig, output_dir, stem, formats):
    for fmt in formats:
        fig.savefig(output_dir / f"{stem}.{fmt}", dpi=180, bbox_inches="tight")


def _plot_log_losses(df, output_dir, formats):
    import matplotlib.pyplot as plt

    groups = {
        "nn_state_supervision": [
            ("loss_total", "train batch total"),
            ("train_loss_total_eval", "train eval total"),
            ("train_loss_v", "train voltage"),
            ("train_loss_c_s_n", "train c_s_n norm"),
            ("train_loss_c_e", "train c_e norm"),
            ("train_loss_c_s_p", "train c_s_p norm"),
            ("val_loss_state", "val total"),
            ("val_loss_v", "val voltage"),
            ("val_loss_c_s_n", "val c_s_n norm"),
            ("val_loss_c_e", "val c_e norm"),
            ("val_loss_c_s_p", "val c_s_p norm"),
        ],
        "nn_raw_state_rmse": [
            ("train_loss_c_s_n_raw", "train c_s_n raw"),
            ("train_loss_c_e_raw", "train c_e raw"),
            ("train_loss_c_s_p_raw", "train c_s_p raw"),
            ("val_loss_c_s_n_raw", "val c_s_n raw"),
            ("val_loss_c_e_raw", "val c_e raw"),
            ("val_loss_c_s_p_raw", "val c_s_p raw"),
        ],
        "train_batch": [
            ("loss", "total"),
            ("train_batch_loss_v", "voltage"),
            ("train_batch_loss_li_s", "Li solid"),
            ("train_batch_loss_li_e", "Li electrolyte"),
            ("train_batch_loss_integ", "integration"),
            ("train_batch_loss_integ_n", "integration n"),
            ("train_batch_loss_integ_e", "integration e"),
            ("train_batch_loss_integ_p", "integration p"),
        ],
        "validation": [
            ("val_subset_loss", "val subset total"),
            ("val_subset_loss_v", "val subset voltage"),
            ("val_subset_loss_li_s", "val subset Li solid"),
            ("val_subset_loss_li_e", "val subset Li electrolyte"),
            ("val_subset_loss_integ", "val subset integration"),
            ("train_full_loss", "train full total"),
            ("val_full_loss", "val full total"),
            ("val_full_loss_v", "val full voltage"),
            ("val_full_loss_li_s", "val full Li solid"),
            ("val_full_loss_li_e", "val full Li electrolyte"),
            ("val_full_loss_integ", "val full integration"),
        ],
    }

    epochs = _numeric_series(df, "epoch")
    for stem, columns in groups.items():
        present = [(column, label) for column, label in columns if column in df.columns]
        if not present:
            continue

        fig, ax = plt.subplots(figsize=(10.5, 6.0))
        for column, label in present:
            values = _numeric_series(df, column)
            mask = np.isfinite(values) & (values > 0)
            if mask.any():
                ax.plot(epochs[mask], np.log10(values[mask]), linewidth=1.5, label=label)

        ax.set_xlabel("epoch")
        ax.set_ylabel("log10(loss)")
        ax.set_title(stem.replace("_", " ").title())
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8, ncol=2)
        _save(fig, output_dir, f"{stem}_log_losses", formats)
        plt.close(fig)


def _plot_degradation(df, output_dir, formats):
    import matplotlib.pyplot as plt

    groups = [
        ("delta_sei", "delta_sei [m]", "delta_sei_est_epoch", "delta_sei_true_epoch"),
        ("epsilon_e_n", "epsilon_e_n", "epsilon_e_n_est_epoch", "epsilon_e_n_true_epoch"),
        ("nLi_sei", "nLi_sei [mol]", "nLi_sei_est_epoch", "nLi_sei_true_epoch"),
        ("nLi_s", "nLi_s [mol]", "nLi_s_est_epoch", "nLi_s_true_epoch"),
        ("nLi_e", "nLi_e [mol]", "nLi_e_est_epoch", "nLi_e_true_epoch"),
        ("R_f_n", "R_f_n", "R_f_n_est_epoch", "R_f_n_true_epoch"),
    ]
    present = [(name, ylabel, est, true) for name, ylabel, est, true in groups if est in df.columns]
    if not present:
        return

    epochs = _numeric_series(df, "epoch")
    fig, axes = plt.subplots(len(present), 1, figsize=(10.5, 2.6 * len(present)), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, (name, ylabel, est_col, true_col) in zip(axes, present):
        est = _numeric_series(df, est_col)
        ax.plot(epochs, est, linestyle="-", linewidth=1.6, label=f"{name} estimate")
        if true_col and true_col in df.columns:
            true = _numeric_series(df, true_col)
            ax.plot(epochs, true, linestyle=":", linewidth=1.8, color="#111111", label=f"{name} true")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("epoch")
    fig.suptitle("Estimated Degradation Parameters", y=0.995)
    fig.tight_layout()
    _save(fig, output_dir, "degradation_parameters", formats)
    plt.close(fig)


def _plot_lithium_diagnostics(df, output_dir, formats):
    import matplotlib.pyplot as plt

    groups = [
        ("solid lithium inventory", [
            ("train_batch_nLi_s_mean", "batch mean prediction"),
            ("nLi_s_est_epoch", "estimated target"),
            ("train_batch_std_pred_nLi_s", "batch std"),
            ("train_batch_nLi_s_rel_std", "relative std"),
        ]),
        ("electrolyte lithium inventory", [
            ("train_batch_nLi_e_mean", "batch mean prediction"),
            ("nLi_e_est_epoch", "estimated target"),
            ("train_batch_std_pred_nLi_e", "batch std"),
            ("train_batch_nLi_e_rel_std", "relative std"),
        ]),
    ]
    epochs = _numeric_series(df, "epoch")

    for title, columns in groups:
        present = [(column, label) for column, label in columns if column in df.columns]
        if not present:
            continue
        fig, ax = plt.subplots(figsize=(10.5, 5.5))
        for column, label in present:
            values = _numeric_series(df, column)
            ax.plot(epochs, values, linewidth=1.5, label=label)
        ax.set_xlabel("epoch")
        ax.set_title(title.title())
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        _save(fig, output_dir, title.replace(" ", "_"), formats)
        plt.close(fig)


def _plot_timing(df, output_dir, formats):
    import matplotlib.pyplot as plt

    timing_cols = sorted(column for column in df.columns if column.startswith("timing_") and column.endswith("_s"))
    if not timing_cols:
        return
    epochs = _numeric_series(df, "epoch")

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    for column in timing_cols:
        values = _numeric_series(df, column)
        mask = np.isfinite(values) & (values > 0)
        if mask.any():
            label = column.removeprefix("timing_").removesuffix("_s")
            ax.plot(epochs[mask], values[mask], linewidth=1.3, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel("seconds")
    ax.set_title("Timing Profile")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8, ncol=2)
    _save(fig, output_dir, "timing_profile", formats)
    plt.close(fig)


def _device_from_arg(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _restore_checkpoint_models(checkpoint, device):
    cfg = dict(checkpoint.get("config", {}) or {})
    if not cfg:
        raise ValueError("Checkpoint does not contain a config; cannot rebuild models for profile plots.")

    p = ip.InitParamsSPMe(cfg)
    nn_models = create_spme_nn_models(cfg, p).to(device)
    model = SPMeModel(p, cfg).to(device)

    for nn_model, state_dict in zip(nn_models, checkpoint["nn_models"]):
        nn_model.load_state_dict(state_dict)
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"], strict=False)

    nn_models.eval()
    model.eval()
    return cfg, p, nn_models, model


def _safe_name(value):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))


def _plot_pair(ax, x, true_y, pred_y, label, color_idx=0):
    actual_colors = [
        "#111111", "#8c564b", "#7f7f7f", "#bcbd22", "#17becf",
        "#9467bd", "#e377c2", "#393b79", "#637939", "#8c6d31",
    ]
    pred_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    ax.plot(x, true_y, linestyle=":", linewidth=1.8, color=actual_colors[color_idx % len(actual_colors)], label=f"{label} true")
    ax.plot(x, pred_y, linestyle="--", linewidth=1.4, color=pred_colors[color_idx % len(pred_colors)], label=f"{label} predicted")


def _plot_profile_spatial(result, model, output_dir, stem, kind, formats, n_samples):
    import matplotlib.pyplot as plt

    if result.get(f"c_n_{kind}") is None:
        return

    if kind == "roll":
        valid = (
            np.isfinite(result["c_n_roll"]).all(axis=1)
            & np.isfinite(result["c_ex_roll"]).all(axis=1)
            & np.isfinite(result["c_p_roll"]).all(axis=1)
        )
        valid_idx = np.flatnonzero(valid)
    else:
        valid_idx = np.arange(len(result["time"]))
    if len(valid_idx) == 0:
        return

    count = min(n_samples, len(valid_idx))
    sample_idx = valid_idx[np.linspace(0, len(valid_idx) - 1, count, dtype=int)]
    r_axis = np.linspace(0.0, 1.0, model.n_r + 1)
    x_n = np.arange(0, model.p.Nxn + 1)
    x_s = np.arange(model.p.Nxn, model.p.Nxn + model.p.Nxs + 1)
    x_p = np.arange(model.p.Nxn + model.p.Nxs, model.p.Nx + 1)
    suffix = "rolled_spme" if kind == "roll" else "bare_nn"

    fig, axes = plt.subplots(5, count, figsize=(3.8 * count, 12.0), squeeze=False)
    for col, idx in enumerate(sample_idx):
        _plot_pair(axes[0, col], r_axis, result["c_n_true"][idx], result[f"c_n_{kind}"][idx], suffix, 0)
        axes[0, col].set_title(f't={result["time"][idx]:.0f}s c_s_n')
        _plot_pair(
            axes[1, col],
            x_n,
            result["c_ex_true"][idx, :model.p.Nxn + 1],
            result[f"c_ex_{kind}"][idx, :model.p.Nxn + 1],
            suffix,
            1,
        )
        axes[1, col].set_title("c_e negative")
        _plot_pair(
            axes[2, col],
            x_s,
            result["c_ex_true"][idx, model.p.Nxn:model.p.Nxn + model.p.Nxs + 1],
            result[f"c_ex_{kind}"][idx, model.p.Nxn:model.p.Nxn + model.p.Nxs + 1],
            suffix,
            2,
        )
        axes[2, col].set_title("c_e separator")
        _plot_pair(
            axes[3, col],
            x_p,
            result["c_ex_true"][idx, model.p.Nxn + model.p.Nxs:],
            result[f"c_ex_{kind}"][idx, model.p.Nxn + model.p.Nxs:],
            suffix,
            3,
        )
        axes[3, col].set_title("c_e positive")
        _plot_pair(axes[4, col], r_axis, result["c_p_true"][idx], result[f"c_p_{kind}"][idx], suffix, 4)
        axes[4, col].set_title("c_s_p")
        for row in range(5):
            axes[row, col].grid(True, alpha=0.25)
            if col == count - 1:
                axes[row, col].legend(loc="best", fontsize=8)
    fig.tight_layout()
    _save(fig, output_dir, f"{stem}_spatial_profiles_{suffix}", formats)
    plt.close(fig)


def _plot_profile_time_series(result, output_dir, stem, kind, formats):
    import matplotlib.pyplot as plt

    suffix = "rolled_spme" if kind == "roll" else "bare_nn"
    c_n_key = f"c_n_{kind}"
    c_ex_key = f"c_ex_{kind}"
    c_p_key = f"c_p_{kind}"
    v_key = "v_roll" if kind == "roll" else "v_pred"
    n_li_s_key = "n_li_s_roll" if kind == "roll" else "n_li_s_pred"
    n_li_e_key = "n_li_e_roll" if kind == "roll" else "n_li_e_pred"
    n_li_sei_key = "n_li_sei_roll" if kind == "roll" else "n_li_sei_pred"

    if result.get(c_n_key) is None:
        return
    mask = (
        np.isfinite(result[c_n_key]).all(axis=1)
        & np.isfinite(result[c_ex_key]).all(axis=1)
        & np.isfinite(result[c_p_key]).all(axis=1)
        & np.isfinite(result[v_key])
        & np.isfinite(result[n_li_s_key])
        & np.isfinite(result[n_li_e_key])
        & np.isfinite(result[n_li_sei_key])
    )
    if not np.any(mask):
        return

    t = result["time"][mask]
    solid_locs = {"center": 0, "mid": result[c_n_key].shape[1] // 2, "surface": -1}
    n_ce = result[c_ex_key].shape[1]
    e_locs = {
        "neg": [0, max(0, n_ce // 6), max(0, n_ce // 3)],
        "sep": [max(0, n_ce // 3), max(0, n_ce // 2), min(n_ce - 1, 2 * n_ce // 3)],
        "pos": [min(n_ce - 1, 2 * n_ce // 3), min(n_ce - 1, 5 * n_ce // 6), -1],
    }

    fig, axes = plt.subplots(10, 1, figsize=(12, 24), sharex=True)
    for color_idx, (label, idx) in enumerate(solid_locs.items()):
        _plot_pair(axes[0], t, result["c_n_true"][mask, idx], result[c_n_key][mask, idx], label, color_idx)
        _plot_pair(axes[1], t, result["c_p_true"][mask, idx], result[c_p_key][mask, idx], label, color_idx)
    for row, region in enumerate(["neg", "sep", "pos"], start=2):
        for color_idx, idx in enumerate(e_locs[region]):
            _plot_pair(axes[row], t, result["c_ex_true"][mask, idx], result[c_ex_key][mask, idx], f"c_e {region} {idx}", color_idx)
    _plot_pair(axes[5], t, result["v_true"][mask], result[v_key][mask], "voltage", 0)
    axes[6].plot(t, result[v_key][mask] - result["v_true"][mask], linestyle="--", linewidth=1.3, label=f"{suffix} voltage error")
    _plot_pair(axes[7], t, result["n_li_s_true"][mask], result[n_li_s_key][mask], "solid Li inventory", 2)
    _plot_pair(axes[8], t, result["n_li_e_true"][mask], result[n_li_e_key][mask], "electrolyte Li inventory", 3)
    _plot_pair(axes[9], t, result["n_li_sei_true"][mask], result[n_li_sei_key][mask], "SEI Li inventory", 4)

    labels = ["c_s_n", "c_s_p", "c_e neg", "c_e sep", "c_e pos", "V", "V error", "nLi solid", "nLi electrolyte", "nLi SEI"]
    for ax, ylabel in zip(axes, labels):
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8, ncol=2)
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    _save(fig, output_dir, f"{stem}_time_series_{suffix}", formats)
    plt.close(fig)


def _plot_profile_regional_lithium(result, output_dir, stem, kind, formats):
    import matplotlib.pyplot as plt

    suffix = "rolled_spme" if kind == "roll" else "bare_nn"
    solid_key = "n_li_s_regions_roll" if kind == "roll" else "n_li_s_regions_pred"
    electrolyte_key = "n_li_e_regions_roll" if kind == "roll" else "n_li_e_regions_pred"
    solid_pred = result.get(solid_key)
    electrolyte_pred = result.get(electrolyte_key)
    solid_true = result.get("n_li_s_regions_true")
    electrolyte_true = result.get("n_li_e_regions_true")
    if solid_pred is None or electrolyte_pred is None or solid_true is None or electrolyte_true is None:
        return

    rows = [
        ("solid", "anode", "solid anode nLi [mol]"),
        ("solid", "cathode", "solid cathode nLi [mol]"),
        ("electrolyte", "neg", "electrolyte neg nLi [mol]"),
        ("electrolyte", "sep", "electrolyte sep nLi [mol]"),
        ("electrolyte", "pos", "electrolyte pos nLi [mol]"),
    ]
    mask = np.ones(len(result["time"]), dtype=bool)
    for phase, name, _ in rows:
        true_group = solid_true if phase == "solid" else electrolyte_true
        pred_group = solid_pred if phase == "solid" else electrolyte_pred
        if name not in true_group or name not in pred_group:
            return
        mask &= np.isfinite(true_group[name]) & np.isfinite(pred_group[name])
    if not np.any(mask):
        return

    t = result["time"][mask]
    fig, axes = plt.subplots(len(rows), 1, figsize=(12, 12.5), sharex=True)
    axes = np.atleast_1d(axes)
    for row_idx, (phase, name, ylabel) in enumerate(rows):
        true_group = solid_true if phase == "solid" else electrolyte_true
        pred_group = solid_pred if phase == "solid" else electrolyte_pred
        _plot_pair(axes[row_idx], t, true_group[name][mask], pred_group[name][mask], name, row_idx)
        axes[row_idx].set_ylabel(ylabel)
        axes[row_idx].grid(True, alpha=0.25)
        axes[row_idx].legend(loc="best", fontsize=8)
    axes[0].set_title(f"{suffix} regional lithium inventory")
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    _save(fig, output_dir, f"{stem}_regional_lithium_{suffix}", formats)
    plt.close(fig)


def _plot_profile_voltage_components(result, output_dir, stem, kind, formats):
    import matplotlib.pyplot as plt

    suffix = "rolled_spme" if kind == "roll" else "bare_nn"
    components_key = "components_roll" if kind == "roll" else "components_pred"
    components = result.get(components_key)
    if components is None:
        return

    rows = [
        ("U_p", "U_p"),
        ("U_n", "U_n"),
        ("eta_p", "eta_p"),
        ("eta_n", "eta_n"),
        ("electrolyte_ohmic", "elyte ohmic"),
        ("electrolyte_concentration", "elyte conc"),
        ("film_drop", "film drop"),
    ]
    true_components = result["components_true"]
    mask = np.ones(len(result["time"]), dtype=bool)
    for key, _ in rows:
        mask &= np.isfinite(components[key]) & np.isfinite(true_components[key])
    if not np.any(mask):
        return

    t = result["time"][mask]
    fig, axes = plt.subplots(len(rows), 1, figsize=(12, 18), sharex=True)
    axes = np.atleast_1d(axes)
    for row_idx, (key, label) in enumerate(rows):
        value_ax = axes[row_idx]
        true_y = true_components[key][mask]
        pred_y = components[key][mask]
        if key == "film_drop":
            value_ax.plot(t, true_y, linestyle=":", linewidth=1.8, color="#000000", label=f"{label} true")
            value_ax.plot(t, pred_y, linestyle="--", linewidth=1.5, color="#d62728", label=f"{label} predicted")
        else:
            _plot_pair(value_ax, t, true_y, pred_y, label, row_idx)
        value_ax.set_ylabel(f"{label} [V]")
        value_ax.grid(True, alpha=0.25)
        value_ax.legend(loc="best", fontsize=8)
    axes[0].set_title(f"{suffix} voltage components")
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    _save(fig, output_dir, f"{stem}_voltage_components_{suffix}", formats)
    plt.close(fig)

    contribution_errors = [
        ("voltage", "total voltage", lambda comp: comp["voltage"]),
        ("ocp", "U_p - U_n", lambda comp: comp["U_p"] - comp["U_n"]),
        ("kinetic", "eta_p - eta_n", lambda comp: comp["eta_p"] - comp["eta_n"]),
        ("electrolyte_ohmic", "elyte ohmic", lambda comp: comp["electrolyte_ohmic"]),
        ("electrolyte_concentration", "elyte conc", lambda comp: comp["electrolyte_concentration"]),
        ("film_drop", "-film drop", lambda comp: -comp["film_drop"]),
    ]
    fig, ax = plt.subplots(figsize=(12, 6.5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for row_idx, (key, label, contribution) in enumerate(contribution_errors):
        if key == "film_drop":
            color = "#d62728"
            linewidth = 1.7
        elif key == "voltage":
            color = "#000000"
            linewidth = 2.0
        else:
            color = colors[(row_idx - 1) % len(colors)]
            linewidth = 1.3
        true_y = contribution(true_components)[mask]
        pred_y = contribution(components)[mask]
        ax.plot(t, pred_y - true_y, linewidth=linewidth, color=color, label=f"{label} error")
    ax.axhline(0.0, color="#111111", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("error [V]")
    ax.set_title(f"{suffix} voltage/component prediction errors")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    _save(fig, output_dir, f"{stem}_voltage_component_errors_{suffix}", formats)
    plt.close(fig)


def _plot_profile_predictions(checkpoint, output_dir, formats, device, splits, max_profiles, plot_samples):
    cfg, p, nn_models, model = _restore_checkpoint_models(checkpoint, device)
    cfg = dict(cfg)
    cfg["use_wandb"] = False
    cfg["wandb_plot_samples"] = int(plot_samples)
    datasets = {}
    if "train" in splits or "val" in splits:
        if cfg.get("method", "").lower() == "nn":
            train_data, val_data = train_pinn_spme._load_prepared_nn_data(cfg, device, p)
        else:
            train_data, val_data = train_pinn_spme._load_prepared_pinn_data(cfg, device, p)
        datasets["train"] = train_data
        datasets["val"] = val_data

    profile_dir = output_dir / "profile_predictions"
    profile_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for split_name in splits:
            dataset = datasets.get(split_name)
            if dataset is None:
                continue
            profiles = dataset.get("profiles", [])
            if max_profiles is not None:
                profiles = profiles[:max_profiles]
            for profile_idx, profile in enumerate(profiles):
                result = train_pinn_spme._predict_profile_windows(nn_models, model, profile, cfg, device)
                if result is None:
                    continue
                profile_stem = f"{split_name}_{profile_idx:02d}_{_safe_name(Path(profile['name']).stem)}"
                _plot_profile_spatial(result, model, profile_dir, profile_stem, "pred", formats, int(plot_samples))
                _plot_profile_time_series(result, profile_dir, profile_stem, "pred", formats)
                _plot_profile_regional_lithium(result, profile_dir, profile_stem, "pred", formats)
                _plot_profile_voltage_components(result, profile_dir, profile_stem, "pred", formats)
                if cfg.get("method", "").lower() == "pinn":
                    _plot_profile_spatial(result, model, profile_dir, profile_stem, "roll", formats, int(plot_samples))
                    _plot_profile_time_series(result, profile_dir, profile_stem, "roll", formats)
                    _plot_profile_regional_lithium(result, profile_dir, profile_stem, "roll", formats)
                    _plot_profile_voltage_components(result, profile_dir, profile_stem, "roll", formats)
                print(f"Saved profile plots for {split_name}/{profile['name']}")


def _print_summary(df, checkpoint_path, output_dir):
    last = df.iloc[-1]
    print(f"Checkpoint: {checkpoint_path}")
    print(f"History rows: {len(df)}")
    print(f"Epoch range: {df['epoch'].iloc[0]} -> {df['epoch'].iloc[-1]}")
    for column in [
        "loss",
        "loss_total",
        "train_loss_total_eval",
        "val_loss_state",
        "train_loss_v",
        "val_loss_v",
        "train_batch_loss_v",
        "train_batch_loss_li_s",
        "train_batch_loss_li_e",
        "train_batch_loss_integ",
        "delta_sei_est_epoch",
        "epsilon_e_n_est_epoch",
        "nLi_sei_est_epoch",
        "nLi_s_est_epoch",
        "nLi_e_est_epoch",
    ]:
        if column in df.columns:
            value = pd.to_numeric(pd.Series([last[column]]), errors="coerce").iloc[0]
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                print(f"Latest {column}: {float(value):.8g}")
    print(f"Saved plots to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="results/spme_pinn_from_nn_2000/checkpoint_latest.pth",
        help="Path to a PINN-SPMe checkpoint containing a history list.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for plots. Defaults to <checkpoint directory>/history_plots.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config to overlay onto the checkpoint config, e.g. config_spme_nn.yaml.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        choices=["png", "pdf", "svg"],
        help="Figure formats to save.",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also save the extracted history as CSV in the output directory.",
    )
    parser.add_argument(
        "--no-profile-plots",
        action="store_true",
        help="Skip dataset prediction plots and only plot checkpoint history.",
    )
    parser.add_argument(
        "--profile-splits",
        nargs="+",
        default=["train", "val"],
        choices=["train", "val"],
        help="Dataset splits to use for bare-NN and rolled-SPMe prediction plots.",
    )
    parser.add_argument(
        "--max-profiles",
        type=int,
        default=None,
        help="Maximum number of profiles per split to plot. Defaults to all profiles.",
    )
    parser.add_argument(
        "--plot-samples",
        type=int,
        default=None,
        help="Number of spatial profile snapshots. Defaults to checkpoint/config wandb_plot_samples.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for dataset prediction plots: auto, cpu, cuda, cuda:0, etc.",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / "history_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = _with_config_override(_load_checkpoint(checkpoint_path), args.config)
    df = _history_frame(checkpoint)
    df = _add_reference_degradation_columns(df, checkpoint)
    if args.csv:
        df.to_csv(output_dir / "checkpoint_history.csv", index=False)

    import matplotlib
    matplotlib.use("Agg", force=True)

    _plot_log_losses(df, output_dir, args.formats)
    _plot_degradation(df, output_dir, args.formats)
    _plot_lithium_diagnostics(df, output_dir, args.formats)
    _plot_timing(df, output_dir, args.formats)
    if not args.no_profile_plots:
        cfg = checkpoint.get("config", {}) or {}
        plot_samples = args.plot_samples
        if plot_samples is None:
            plot_samples = int(cfg.get("plot_samples", cfg.get("wandb_plot_samples", 2)))
        _plot_profile_predictions(
            checkpoint,
            output_dir,
            args.formats,
            _device_from_arg(args.device),
            args.profile_splits,
            args.max_profiles,
            plot_samples,
        )
    _print_summary(df, checkpoint_path, output_dir)


if __name__ == "__main__":
    main()
