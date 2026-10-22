#!/usr/bin/env python3
"""Generate publication-ready SPMe PINN/NN figures.

The script creates figures for:
1. Training current/voltage profiles.
2. Validation current/voltage profiles.
3. Total loss trajectory.
4. Degradation/parameter estimation trajectories.
5. Bare-NN surface solid concentration predictions.
6. Bare-NN average solid concentration predictions.
7. Bare-NN electrolyte boundary concentration predictions.
8. Bare-NN voltage prediction.

It also writes numerical error reports for final parameter estimates and for
per-trajectory dynamic-profile state estimates.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

import matplotlib
matplotlib.use("Agg", force=True)

import init_params as ip
import train_pinn_spme
from pinn_spme_network import create_spme_nn_models
from spme_model import SPMeModel


def _load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _device_from_arg(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _safe_name(value):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))


def _save(fig, output_dir, stem, formats):
    for fmt in formats:
        fig.savefig(output_dir / f"{stem}.{fmt}", dpi=300, bbox_inches="tight")


def _resolve_files(data_dir, names):
    return [Path(data_dir) / name for name in names]


def _profile_label(path):
    stem = Path(path).stem
    for prefix in ("spme_train_", "spme_val_", "id_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
    return stem.replace("_soh90", "").replace("_", " ")


def _load_profile_signal(path):
    df = pd.read_csv(path)
    if "time_s" in df.columns:
        time = df["time_s"].to_numpy(dtype=float)
    elif "Test_Time(s)" in df.columns:
        time = df["Test_Time(s)"].to_numpy(dtype=float)
    else:
        time = np.arange(len(df), dtype=float)
    return {
        "path": path,
        "label": _profile_label(path),
        "time": time,
        "current": df["current_A"].to_numpy(dtype=float),
        "voltage": df["voltage_V"].to_numpy(dtype=float),
    }


def _plot_one_signal_profile(profile, output_dir, stem, title, formats):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 5.8), sharex=True)
    time = profile["time"]
    axes[0].plot(time, profile["current"], linewidth=1.1, color="#1f77b4")
    axes[1].plot(time, profile["voltage"], linewidth=1.1, color="#d62728")
    axes[0].set_ylabel("current [A]")
    axes[1].set_ylabel("voltage [V]")
    axes[1].set_xlabel("time [s]")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    axes[0].set_title(title)
    fig.tight_layout()
    _save(fig, output_dir, stem, formats)
    plt.close(fig)


def _plot_validation_signal_profiles(files, output_dir, formats):
    for path in files:
        profile = _load_profile_signal(path)
        label = profile["label"]
        _plot_one_signal_profile(
            profile,
            output_dir,
            f"fig02_validation_data_current_voltage_{_safe_name(label)}",
            f"Validation Data: {label}",
            formats,
        )


def _plot_training_signal_profiles(files, output_dir, formats):
    import matplotlib.pyplot as plt

    profiles = [_load_profile_signal(path) for path in files]
    if not profiles:
        return

    dst = next((profile for profile in profiles if "DST" in profile["label"].upper()), profiles[0])
    hppc = [profile for profile in profiles if profile is not dst]
    if not hppc:
        _plot_one_signal_profile(
            dst,
            output_dir,
            "fig01_training_data_current_voltage",
            f"Training Data: {dst['label']}",
            formats,
        )
        return

    n_hppc = len(hppc)
    fig = plt.figure(figsize=(max(11.0, 2.35 * n_hppc), 8.2))
    gs = fig.add_gridspec(4, n_hppc, height_ratios=[1.0, 1.0, 0.9, 0.9], hspace=0.42, wspace=0.34)

    dst_current_ax = fig.add_subplot(gs[0, :])
    dst_voltage_ax = fig.add_subplot(gs[1, :], sharex=dst_current_ax)
    dst_current_ax.plot(dst["time"], dst["current"], linewidth=1.1, color="#1f77b4")
    dst_voltage_ax.plot(dst["time"], dst["voltage"], linewidth=1.1, color="#d62728")
    dst_current_ax.set_title(f"DST: {dst['label']}")
    dst_current_ax.set_ylabel("current [A]")
    dst_voltage_ax.set_ylabel("voltage [V]")
    dst_voltage_ax.set_xlabel("time [s]")

    hppc_current_axes = []
    hppc_voltage_axes = []
    hppc_current_values = np.concatenate([profile["current"] for profile in hppc])
    hppc_voltage_values = np.concatenate([profile["voltage"] for profile in hppc])
    for col, profile in enumerate(hppc):
        current_ax = fig.add_subplot(gs[2, col])
        voltage_ax = fig.add_subplot(gs[3, col], sharex=current_ax)
        current_ax.plot(profile["time"], profile["current"], linewidth=1.0, color="#1f77b4")
        voltage_ax.plot(profile["time"], profile["voltage"], linewidth=1.0, color="#d62728")
        voltage_ax.set_xlabel("time [s]")
        if col == 0:
            current_ax.set_ylabel("current [A]")
            voltage_ax.set_ylabel("voltage [V]")
        hppc_current_axes.append(current_ax)
        hppc_voltage_axes.append(voltage_ax)

    current_pad = 0.04 * max(float(np.ptp(hppc_current_values)), 1e-12)
    voltage_pad = 0.04 * max(float(np.ptp(hppc_voltage_values)), 1e-12)
    for ax in hppc_current_axes:
        ax.set_ylim(float(np.min(hppc_current_values)) - current_pad, float(np.max(hppc_current_values)) + current_pad)
    for ax in hppc_voltage_axes:
        ax.set_ylim(float(np.min(hppc_voltage_values)) - voltage_pad, float(np.max(hppc_voltage_values)) + voltage_pad)

    for ax in [dst_current_ax, dst_voltage_ax, *hppc_current_axes, *hppc_voltage_axes]:
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="both", labelsize=8)

    fig.subplots_adjust(left=0.075, right=0.985, top=0.94, bottom=0.08)
    _save(fig, output_dir, "fig01_training_data_current_voltage", formats)
    plt.close(fig)


def _history_frame(checkpoint):
    history = checkpoint.get("history", [])
    if not history:
        raise ValueError("Checkpoint does not contain non-empty history.")
    df = pd.DataFrame(history)
    if "epoch" not in df.columns:
        df.insert(0, "epoch", np.arange(1, len(df) + 1))
    return df.sort_values("epoch")


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


def _add_reference_columns(df, checkpoint):
    cfg = checkpoint.get("config", {}) or {}
    model_state = checkpoint.get("model_state", {}) or {}

    delta_true = _scalar(cfg.get("delta_sei_true"))
    if "delta_sei_true_epoch" in df.columns:
        values = pd.to_numeric(df["delta_sei_true_epoch"], errors="coerce").dropna()
        if not values.empty:
            delta_true = float(values.iloc[-1])
    _constant_column(df, "delta_sei_true_epoch", delta_true)

    epsilon_true = _scalar(cfg.get("epsilon_e_n_true"))
    if "epsilon_e_n_true_epoch" in df.columns:
        values = pd.to_numeric(df["epsilon_e_n_true_epoch"], errors="coerce").dropna()
        if not values.empty:
            epsilon_true = float(values.iloc[-1])
    _constant_column(df, "epsilon_e_n_true_epoch", epsilon_true)

    c_li_sei = _scalar(model_state.get("C_li_sei_value"))
    nli_total = _scalar(model_state.get("nLi_total"))
    if delta_true is not None and c_li_sei is not None:
        nli_sei_true = c_li_sei * delta_true
        _constant_column(df, "nLi_sei_true_epoch", nli_sei_true)
        if nli_total is not None:
            _constant_column(df, "nLi_s_true_epoch", nli_total - nli_sei_true)
    return df


def _last_finite(values):
    arr = np.asarray(values, dtype=float).reshape(-1)
    mask = np.isfinite(arr)
    if not mask.any():
        return None
    return float(arr[np.flatnonzero(mask)[-1]])


def _relative_error_pct(error, reference):
    if reference is None or not np.isfinite(reference) or abs(reference) <= 0.0:
        return np.nan
    return 100.0 * error / abs(reference)


def _error_metrics(true, pred):
    true = np.asarray(true, dtype=float).reshape(-1)
    pred = np.asarray(pred, dtype=float).reshape(-1)
    mask = np.isfinite(true) & np.isfinite(pred)
    if not mask.any():
        return None
    err = pred[mask] - true[mask]
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    max_abs = float(np.max(np.abs(err)))
    true_rms = float(np.sqrt(np.mean(true[mask] ** 2)))
    rel_rmse_pct = np.nan if true_rms <= 0.0 else 100.0 * rmse / true_rms
    return {
        "n_samples": int(mask.sum()),
        "rmse": rmse,
        "mae": mae,
        "max_abs": max_abs,
        "bias": float(np.mean(err)),
        "relative_rmse_pct": rel_rmse_pct,
    }


def _write_parameter_error_report(checkpoint, output_dir):
    df = _add_reference_columns(_history_frame(checkpoint), checkpoint)
    parameter_specs = [
        ("delta_sei", "SEI thickness", "m", "delta_sei_est_epoch", "delta_sei_true_epoch"),
        ("epsilon_e_n", "negative electrolyte porosity", "1", "epsilon_e_n_est_epoch", "epsilon_e_n_true_epoch"),
        ("nLi_s", "cyclable lithium in solid", "mol", "nLi_s_est_epoch", "nLi_s_true_epoch"),
        ("nLi_sei", "lithium in SEI", "mol", "nLi_sei_est_epoch", "nLi_sei_true_epoch"),
        ("nLi_e", "lithium in electrolyte", "mol", "nLi_e_est_epoch", "nLi_e_true_epoch"),
    ]
    rows = []
    for key, label, unit, est_col, true_col in parameter_specs:
        if est_col not in df.columns or true_col not in df.columns:
            continue
        est = _last_finite(pd.to_numeric(df[est_col], errors="coerce").to_numpy(dtype=float))
        true = _last_finite(pd.to_numeric(df[true_col], errors="coerce").to_numpy(dtype=float))
        if est is None or true is None:
            continue
        error = est - true
        rows.append({
            "parameter": key,
            "label": label,
            "unit": unit,
            "estimate": est,
            "true": true,
            "error": error,
            "absolute_error": abs(error),
            "relative_error_pct": _relative_error_pct(error, true),
        })

    frame = pd.DataFrame(rows)
    path = output_dir / "parameter_estimation_error.csv"
    frame.to_csv(path, index=False)
    return path, frame


def _append_metric_row(rows, profile_label, estimator, quantity, unit, true, pred):
    metrics = _error_metrics(true, pred)
    if metrics is None:
        return
    rows.append({
        "trajectory": profile_label,
        "estimator": estimator,
        "quantity": quantity,
        "unit": unit,
        **metrics,
    })


def _write_state_error_report(results, model, output_dir):
    rows = []
    for item in results:
        result = item["result"]
        label = item["label"]
        true_avg_n, true_avg_p = _solid_averages(result, model, "true")

        for estimator, suffix in (("bare_nn", "pred"), ("rolled_spme", "roll")):
            c_n_key = f"c_n_{suffix}"
            c_p_key = f"c_p_{suffix}"
            v_key = f"v_{suffix}"
            if c_n_key not in result or result[c_n_key] is None:
                continue

            pred_avg_n, pred_avg_p = _solid_averages(result, model, suffix)
            _append_metric_row(
                rows, label, estimator, "solid_anode_surface_concentration", "mol/m3",
                result["c_n_true"][:, -1], result[c_n_key][:, -1],
            )
            _append_metric_row(
                rows, label, estimator, "solid_cathode_surface_concentration", "mol/m3",
                result["c_p_true"][:, -1], result[c_p_key][:, -1],
            )
            _append_metric_row(
                rows, label, estimator, "solid_anode_average_concentration", "mol/m3",
                true_avg_n, pred_avg_n,
            )
            _append_metric_row(
                rows, label, estimator, "solid_cathode_average_concentration", "mol/m3",
                true_avg_p, pred_avg_p,
            )
            _append_metric_row(
                rows, label, estimator, "voltage", "V", result["v_true"], result[v_key],
            )

    frame = pd.DataFrame(rows)
    path = output_dir / "trajectory_state_estimation_errors.csv"
    frame.to_csv(path, index=False)
    return path, frame


def _write_text_error_summary(parameter_frame, state_frame, output_dir):
    lines = ["SPMe PINN numerical error summary", ""]

    if parameter_frame.empty:
        lines.append("Final parameter-estimation errors: no parameter rows available.")
    else:
        lines.append("Final parameter-estimation errors:")
        for row in parameter_frame.itertuples(index=False):
            rel = ""
            if isinstance(row.relative_error_pct, float) and math.isfinite(row.relative_error_pct):
                rel = f", rel={row.relative_error_pct:.6g}%"
            lines.append(
                f"- {row.parameter}: estimate={row.estimate:.12g} {row.unit}, "
                f"true={row.true:.12g} {row.unit}, error={row.error:.12g} {row.unit}{rel}"
            )

    lines.append("")
    if state_frame.empty:
        lines.append("Per-trajectory state-estimation errors: no prediction rows available.")
    else:
        lines.append("Per-trajectory state-estimation errors (RMSE):")
        for row in state_frame.itertuples(index=False):
            rel = ""
            if isinstance(row.relative_rmse_pct, float) and math.isfinite(row.relative_rmse_pct):
                rel = f", rel_rmse={row.relative_rmse_pct:.6g}%"
            lines.append(
                f"- {row.trajectory}, {row.estimator}, {row.quantity}: "
                f"rmse={row.rmse:.12g} {row.unit}, mae={row.mae:.12g} {row.unit}, "
                f"max_abs={row.max_abs:.12g} {row.unit}{rel}"
            )

    path = output_dir / "numerical_error_summary.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


def _series(df, column):
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    return np.where(np.isfinite(values), values, np.nan)


def _plot_parameter_trajectory(checkpoint, output_dir, formats):
    import matplotlib.pyplot as plt

    df = _add_reference_columns(_history_frame(checkpoint), checkpoint)
    epochs = _series(df, "epoch")
    groups = [
        ("delta_sei", "SEI thickness [m]", "delta_sei_est_epoch", "delta_sei_true_epoch"),
        ("epsilon_e_n", "negative electrolyte porosity", "epsilon_e_n_est_epoch", "epsilon_e_n_true_epoch"),
        ("nLi_s", "cyclable lithium in solid [mol]", "nLi_s_est_epoch", "nLi_s_true_epoch"),
    ]
    present = [(name, ylabel, est, true) for name, ylabel, est, true in groups if est in df.columns]
    if not present:
        return

    fig, axes = plt.subplots(len(present), 1, figsize=(10.5, 2.8 * len(present)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (name, ylabel, est_col, true_col) in zip(axes, present):
        est = _series(df, est_col)
        ax.plot(epochs, est, linewidth=1.6, label=f"{name} estimate")
        if true_col in df.columns:
            true = _series(df, true_col)
            ax.plot(epochs, true, linestyle=":", linewidth=2.0, color="#111111", label=f"{name} true")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("epoch")
    fig.tight_layout()
    _save(fig, output_dir, "fig04_parameter_estimation_trajectory", formats)
    plt.close(fig)


def _plot_total_loss(checkpoint, output_dir, formats):
    import matplotlib.pyplot as plt

    df = _history_frame(checkpoint)
    epochs = _series(df, "epoch")
    candidates = [
        ("loss", "PINN train total"),
        ("loss_total", "NN train total"),
        ("val_subset_loss", "validation subset"),
        ("val_full_loss", "validation full"),
        ("val_loss_state", "NN validation total"),
    ]
    present = [(col, label) for col, label in candidates if col in df.columns]
    if not present:
        return

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for col, label in present:
        values = _series(df, col)
        mask = np.isfinite(values) & (values > 0)
        if mask.any():
            ax.plot(epochs[mask], np.log10(values[mask]), linewidth=1.5, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel("log10(total loss)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    _save(fig, output_dir, "fig03_total_loss_trajectory", formats)
    plt.close(fig)


def _restore_models(checkpoint, device):
    cfg = dict(checkpoint.get("config", {}) or {})
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


def _load_profiles_for_prediction(files, cfg, p):
    dataset = train_pinn_spme._load_spme_csv_dataset([str(path) for path in files], p)
    return dataset.get("profiles", [])


def _solid_averages(result, model, kind):
    weights = model.radial_average_weights.detach().cpu().numpy()
    c_n = result[f"c_n_{kind}"]
    c_p = result[f"c_p_{kind}"]
    return c_n @ weights, c_p @ weights


def _plot_prediction_surface(results, output_dir, formats):
    import matplotlib.pyplot as plt

    for item in results:
        result = item["result"]
        label = item["label"]
        t = result["time"]
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 3.8), sharex=False)
        axes[0].plot(t, result["c_n_true"][:, -1], color="#111111", linestyle=":", linewidth=1.8, label="true")
        axes[0].plot(t, result["c_n_pred"][:, -1], color="#1f77b4", linestyle="--", linewidth=1.2, label="bare NN")
        axes[1].plot(t, result["c_p_true"][:, -1], color="#111111", linestyle=":", linewidth=1.8, label="true")
        axes[1].plot(t, result["c_p_pred"][:, -1], color="#d62728", linestyle="--", linewidth=1.2, label="bare NN")
        axes[0].set_ylabel("c_s,n surf [mol/m3]")
        axes[1].set_ylabel("c_s,p surf [mol/m3]")
        axes[0].set_title("Anode surface concentration")
        axes[1].set_title("Cathode surface concentration")
        for ax in axes:
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=8)
            ax.set_xlabel("time [s]")
        fig.suptitle(label)
        fig.tight_layout()
        _save(fig, output_dir, f"fig05_nn_surface_solid_concentration_{_safe_name(label)}", formats)
        plt.close(fig)


def _plot_prediction_average(results, model, output_dir, formats):
    import matplotlib.pyplot as plt

    for item in results:
        result = item["result"]
        label = item["label"]
        t = result["time"]
        true_n, true_p = _solid_averages(result, model, "true")
        pred_n, pred_p = _solid_averages(result, model, "pred")
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 3.8), sharex=False)
        axes[0].plot(t, true_n, color="#111111", linestyle=":", linewidth=1.8, label="true")
        axes[0].plot(t, pred_n, color="#1f77b4", linestyle="--", linewidth=1.2, label="bare NN")
        axes[1].plot(t, true_p, color="#111111", linestyle=":", linewidth=1.8, label="true")
        axes[1].plot(t, pred_p, color="#d62728", linestyle="--", linewidth=1.2, label="bare NN")
        axes[0].set_ylabel("avg c_s,n [mol/m3]")
        axes[1].set_ylabel("avg c_s,p [mol/m3]")
        axes[0].set_title("Anode average concentration")
        axes[1].set_title("Cathode average concentration")
        for ax in axes:
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=8)
            ax.set_xlabel("time [s]")
        fig.suptitle(label)
        fig.tight_layout()
        _save(fig, output_dir, f"fig06_nn_average_solid_concentration_{_safe_name(label)}", formats)
        plt.close(fig)


def _plot_prediction_electrolyte_boundaries(results, output_dir, formats):
    import matplotlib.pyplot as plt

    for item in results:
        result = item["result"]
        label = item["label"]
        t = result["time"]
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 3.8), sharex=False)
        axes[0].plot(t, result["c_ex_true"][:, 0], color="#111111", linestyle=":", linewidth=1.8, label="true")
        axes[0].plot(t, result["c_ex_pred"][:, 0], color="#1f77b4", linestyle="--", linewidth=1.2, label="bare NN")
        axes[1].plot(t, result["c_ex_true"][:, -1], color="#111111", linestyle=":", linewidth=1.8, label="true")
        axes[1].plot(t, result["c_ex_pred"][:, -1], color="#d62728", linestyle="--", linewidth=1.2, label="bare NN")
        axes[0].set_ylabel("c_e anode cc [mol/m3]")
        axes[1].set_ylabel("c_e cathode cc [mol/m3]")
        axes[0].set_title("Electrolyte concentration at anode current collector")
        axes[1].set_title("Electrolyte concentration at cathode current collector")
        for ax in axes:
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=8)
            ax.set_xlabel("time [s]")
        fig.suptitle(label)
        fig.tight_layout()
        _save(fig, output_dir, f"fig07_nn_electrolyte_boundary_concentration_{_safe_name(label)}", formats)
        plt.close(fig)


def _plot_prediction_voltage(results, output_dir, formats):
    import matplotlib.pyplot as plt

    for item in results:
        result = item["result"]
        label = item["label"]
        t = result["time"]
        v_true = result["v_true"]
        v_pred = result["v_pred"]
        fig, ax = plt.subplots(figsize=(10.5, 4.8))
        ax.plot(t, v_true, color="#111111", linestyle=":", linewidth=1.8, label="true")
        ax.plot(t, v_pred, color="#2ca02c", linestyle="--", linewidth=1.2, label="bare NN calculated")
        ax.set_title(f"Voltage from bare NN states: {label}")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("voltage [V]")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        _save(fig, output_dir, f"fig08_nn_voltage_prediction_{_safe_name(label)}", formats)
        plt.close(fig)


def _prediction_results(files, checkpoint, device, max_profiles=None):
    cfg, p, nn_models, model = _restore_models(checkpoint, device)
    profiles = _load_profiles_for_prediction(files, cfg, p)
    if max_profiles is not None:
        profiles = profiles[:max_profiles]

    results = []
    with torch.no_grad():
        for profile in profiles:
            result = train_pinn_spme._predict_profile_windows(nn_models, model, profile, cfg, device)
            if result is None:
                continue
            results.append({
                "label": _profile_label(profile["name"]),
                "profile": profile,
                "result": result,
            })
    return results, model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML used for data-file/output defaults. The trained model config comes from the checkpoint.",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--formats", nargs="+", default=["png"], choices=["png", "pdf", "svg"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-files", nargs="*", default=None)
    parser.add_argument("--validation-files", nargs="*", default=None)
    parser.add_argument("--prediction-files", nargs="*", default=None)
    parser.add_argument("--max-prediction-profiles", type=int, default=None)
    args = parser.parse_args()

    config = _load_config(args.config) if args.config else {}
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    elif "save_dir" in config:
        checkpoint_path = Path(config["save_dir"]) / "checkpoint_latest.pth"
    else:
        parser.error("provide --checkpoint, or provide --config with save_dir")

    checkpoint = _load_checkpoint(checkpoint_path)
    checkpoint_config = dict(checkpoint.get("config", {}) or {})
    if not checkpoint_config:
        checkpoint_config = dict(config)
    if not checkpoint_config:
        parser.error("checkpoint does not contain config; provide --config for this older checkpoint")
    checkpoint["config"] = checkpoint_config
    checkpoint["config"]["use_wandb"] = False

    plot_config = dict(checkpoint_config)
    plot_config.update(config)

    data_dir = Path(plot_config.get("data_dir", "data"))
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / "result_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_files = args.train_files if args.train_files is not None else plot_config.get("train_data_files", [])
    validation_files = args.validation_files if args.validation_files is not None else plot_config.get("val_data_files", [])
    prediction_files = args.prediction_files if args.prediction_files is not None else validation_files

    _plot_training_signal_profiles(_resolve_files(data_dir, train_files), output_dir, args.formats)
    _plot_validation_signal_profiles(_resolve_files(data_dir, validation_files), output_dir, args.formats)

    prediction_results, model = _prediction_results(
        _resolve_files(data_dir, prediction_files),
        checkpoint,
        _device_from_arg(args.device),
        max_profiles=args.max_prediction_profiles,
    )
    if prediction_results:
        _plot_prediction_surface(prediction_results, output_dir, args.formats)
        _plot_prediction_average(prediction_results, model, output_dir, args.formats)
        _plot_prediction_electrolyte_boundaries(prediction_results, output_dir, args.formats)
        _plot_prediction_voltage(prediction_results, output_dir, args.formats)

    _plot_parameter_trajectory(checkpoint, output_dir, args.formats)
    _plot_total_loss(checkpoint, output_dir, args.formats)
    parameter_report_path, parameter_frame = _write_parameter_error_report(checkpoint, output_dir)
    state_report_path, state_frame = _write_state_error_report(prediction_results, model, output_dir)
    summary_path = _write_text_error_summary(parameter_frame, state_frame, output_dir)
    print(f"Saved result figures to: {output_dir}")
    print(f"Saved numerical parameter errors to: {parameter_report_path}")
    print(f"Saved numerical trajectory errors to: {state_report_path}")
    print(f"Saved numerical error summary to: {summary_path}")


if __name__ == "__main__":
    main()
