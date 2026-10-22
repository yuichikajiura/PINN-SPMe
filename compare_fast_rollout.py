#!/usr/bin/env python3
"""Compare full h=1 SPMe rollout against sparse-target fast rollout.

The fast rollout advances only to requested target offsets. For a jump from
state x(a) to x(b), b > a, it uses one or more coarse averaged-current steps
over I[a+1], ..., I[b-1], capped by --max-fast-h, then one h=1 step with I[b].
For adjacent targets, it uses only the final h=1 step.
"""

import argparse
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

import matplotlib
matplotlib.use("Agg", force=True)

from init_params import InitParamsSPMe
from spme_model import SPMeModel
from train_pinn_spme import _spme_state_columns

DEFAULT_DYNAMIC_PROFILES = [
    "data/spme_val_DST_soh90.csv",
    "data/spme_train_BJDST_soh90.csv",
    "data/spme_train_FUDS_soh90.csv",
    "data/spme_train_US06_soh90.csv",
]


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _build_model(config_path, checkpoint_path, device):
    cfg = _load_yaml(config_path)
    if checkpoint_path:
        checkpoint = _load_checkpoint(checkpoint_path, device)
        checkpoint_cfg = checkpoint.get("config", {}) or {}
        if checkpoint_cfg:
            cfg = dict(checkpoint_cfg)
        cfg["use_wandb"] = False

    cfg.setdefault("h", 1.0)
    p = InitParamsSPMe(cfg)
    model = SPMeModel(p, cfg).to(device=device, dtype=torch.float64)
    if checkpoint_path and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()
    return cfg, p, model


def _load_profile(path, p, device):
    df = pd.read_csv(path)
    c_s_n_cols, c_e_cols, c_s_p_cols = _spme_state_columns(p)
    state_cols = c_s_n_cols + c_e_cols + c_s_p_cols
    missing = [col for col in ["current_A", "voltage_V", *state_cols] if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    if "time_s" in df.columns:
        time = df["time_s"].to_numpy(dtype=np.float64)
    elif "Test_Time(s)" in df.columns:
        time = df["Test_Time(s)"].to_numpy(dtype=np.float64)
    else:
        time = np.arange(len(df), dtype=np.float64)

    x0 = torch.tensor(df.loc[0, state_cols].to_numpy(dtype=np.float64), device=device).view(1, -1)
    current = torch.tensor(df["current_A"].to_numpy(dtype=np.float64), device=device).view(-1, 1)
    voltage = df["voltage_V"].to_numpy(dtype=np.float64)
    return time, x0, current, voltage


def _profile_label(path):
    stem = Path(path).stem
    for prefix in ("spme_train_", "spme_val_", "id_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
    return stem.replace("_soh90", "").replace("_", " ")


@contextmanager
def _temporary_h(model, h):
    old_h = model.h
    model.h = float(h)
    try:
        yield
    finally:
        model.h = old_h


def _voltage(model, x, current):
    c_n, c_p = model.calc_css_and_cs0(x, current)
    _, c_e, _ = model.unpack_state(x)
    return model.calc_voltage(c_n[:, -1], c_p[:, -1], c_e, current).reshape(())


def _step_with_h(model, x, current, h):
    with _temporary_h(model, h):
        return model.step(x, current)


def _linear_step_exact(x, current, matA, matB, h):
    n = matA.shape[0]
    aug = torch.zeros(n + 1, n + 1, dtype=x.dtype, device=x.device)
    aug[:n, :n] = matA.to(dtype=x.dtype, device=x.device)
    aug[:n, n:n + 1] = matB.to(dtype=x.dtype, device=x.device)
    exp_aug = torch.matrix_exp(float(h) * aug)
    exp_a = exp_aug[:n, :n]
    exp_b = exp_aug[:n, n:n + 1]
    return x @ exp_a.T + current.to(dtype=x.dtype, device=x.device) @ exp_b.T


def _linear_step_exact_batched_h(x, current, matA, matB, h_values):
    n = matA.shape[0]
    h_values = h_values.to(dtype=x.dtype, device=x.device).reshape(-1, 1, 1)
    aug = torch.zeros(x.shape[0], n + 1, n + 1, dtype=x.dtype, device=x.device)
    aug[:, :n, :n] = matA.to(dtype=x.dtype, device=x.device).unsqueeze(0)
    aug[:, :n, n:n + 1] = matB.to(dtype=x.dtype, device=x.device).unsqueeze(0)
    exp_aug = torch.matrix_exp(h_values * aug)
    exp_a = exp_aug[:, :n, :n]
    exp_b = exp_aug[:, :n, n:n + 1]
    return torch.bmm(exp_a, x.unsqueeze(-1)).squeeze(-1) + torch.bmm(exp_b, current.unsqueeze(-1)).squeeze(-1)


def _step_exact_variable_h(model, x, current, h_values):
    current = current.to(dtype=x.dtype, device=x.device)
    h_values = h_values.to(dtype=x.dtype, device=x.device).reshape(-1)
    c_s_n, c_e, c_s_p = model.unpack_state(x)
    c_s_n_next = _linear_step_exact_batched_h(c_s_n, current, model.matAn, model.matBn, h_values)
    c_s_p_next = _linear_step_exact_batched_h(c_s_p, current, model.matAp, model.matBp, h_values)

    c_e_next = torch.empty_like(c_e)
    jn = current / (model.p.Faraday * model.p.a_s_n * model.p.Area_n * model.p.L_n)
    jp = -current / (model.p.Faraday * model.p.a_s_p * model.p.Area_p * model.p.L_p)
    for h in torch.unique(h_values.detach().cpu()).tolist():
        mask = h_values == h
        with _temporary_h(model, float(h)):
            c_e_next[mask] = model._electrolyte_step(c_e[mask], jn[mask], jp[mask])
    return model.pack_state(c_s_n_next, c_e_next, c_s_p_next)


def _step_with_h_fast(model, x, current, h, solid_step):
    if solid_step == "rk4":
        return _step_with_h(model, x, current, h)
    if solid_step != "exact":
        raise ValueError("solid_step must be 'rk4' or 'exact'.")

    current = current.to(dtype=x.dtype, device=x.device)
    c_s_n, c_e, c_s_p = model.unpack_state(x)
    jn = current / (model.p.Faraday * model.p.a_s_n * model.p.Area_n * model.p.L_n)
    jp = -current / (model.p.Faraday * model.p.a_s_p * model.p.Area_p * model.p.L_p)
    c_s_n_next = _linear_step_exact(c_s_n, current, model.matAn, model.matBn, h)
    c_s_p_next = _linear_step_exact(c_s_p, current, model.matAp, model.matBp, h)
    with _temporary_h(model, h):
        c_e_next = model._electrolyte_step(c_e, jn, jp)
    return model.pack_state(c_s_n_next, c_e_next, c_s_p_next)


def _roll_h1(model, x0, current, targets):
    max_target = max(targets)
    x = x0.clone()
    states = {0: x.clone()}
    voltages = {0: _voltage(model, x, current[0:1]).detach().clone()}
    with torch.no_grad():
        for step in range(1, max_target + 1):
            x = _step_with_h(model, x, current[step:step + 1], 1)
            states[step] = x.detach().clone()
            voltages[step] = _voltage(model, x, current[step:step + 1]).detach().clone()
    return states, voltages


def _roll_fast(model, x0, current, targets, max_fast_h=5, solid_step="exact"):
    if max_fast_h < 1:
        raise ValueError("max_fast_h must be >= 1.")

    x = x0.clone()
    states = {0: x.clone()}
    voltages = {0: _voltage(model, x, current[0:1]).detach().clone()}
    effective_current = torch.full(
        (max(targets) + 1, 1),
        torch.nan,
        dtype=current.dtype,
        device=current.device,
    )
    effective_current[0:1] = current[0:1]
    previous = 0
    with torch.no_grad():
        for target in targets:
            gap = target - previous
            if gap <= 0:
                raise ValueError("Target offsets must be strictly increasing positive integers.")
            current_step = previous
            while target - current_step > 1:
                chunk_h = min(max_fast_h, target - current_step - 1)
                next_step = current_step + chunk_h
                avg_current = current[current_step + 1:next_step + 1].mean(dim=0, keepdim=True)
                effective_current[current_step + 1:next_step + 1] = avg_current
                x = _step_with_h_fast(model, x, avg_current, chunk_h, solid_step)
                current_step = next_step
            effective_current[target:target + 1] = current[target:target + 1]
            x = _step_with_h_fast(model, x, current[target:target + 1], 1, solid_step)
            states[target] = x.detach().clone()
            voltages[target] = _voltage(model, x, current[target:target + 1]).detach().clone()
            previous = target
    return states, voltages, effective_current


def _roll_direct_exact_targets(model, x0, current, targets):
    target_tensor = torch.as_tensor(targets, dtype=torch.long, device=current.device)
    batch = len(targets)
    x_start = x0.expand(batch, -1).clone()
    x_before_final = x_start.clone()

    coarse_mask = target_tensor > 1
    if bool(coarse_mask.any()):
        coarse_targets = target_tensor[coarse_mask]
        h_values = (coarse_targets - 1).to(dtype=x_start.dtype)
        prefix_current = torch.cat([torch.zeros(1, 1, dtype=current.dtype, device=current.device), current.cumsum(dim=0)], dim=0)
        avg_current = (prefix_current[coarse_targets] - prefix_current[1]) / h_values.reshape(-1, 1)
        x_before_final[coarse_mask] = _step_exact_variable_h(
            model,
            x_start[coarse_mask],
            avg_current,
            h_values,
        )

    final_current = current[target_tensor]
    x_target = _step_with_h_fast(model, x_before_final, final_current, 1, solid_step="exact")
    with torch.no_grad():
        c_n, c_p = model.calc_css_and_cs0(x_target, final_current)
        _, c_e, _ = model.unpack_state(x_target)
        voltages_batch = model.calc_voltage(c_n[:, -1], c_p[:, -1], c_e, final_current).reshape(-1).detach()

    states = {0: x0.clone()}
    voltages = {0: _voltage(model, x0, current[0:1]).detach().clone()}
    for row, target in enumerate(targets):
        states[target] = x_target[row:row + 1].detach().clone()
        voltages[target] = voltages_batch[row].reshape(())

    # A direct-from-x0 prediction has no single time-ordered effective-current path.
    effective_current = current[:max(targets) + 1].detach().clone()
    return states, voltages, effective_current


def _region_rmse_pct(model, reference_state, test_state):
    ref_n, ref_e, ref_p = model.unpack_state(reference_state)
    test_n, test_e, test_p = model.unpack_state(test_state)

    def rel_rmse_pct(ref, test):
        err_rmse = torch.sqrt(torch.mean((test - ref) ** 2))
        ref_rms = torch.sqrt(torch.mean(ref ** 2)).clamp_min(torch.finfo(ref.dtype).eps)
        return float((100.0 * err_rmse / ref_rms).detach().cpu())

    return {
        "solid_anode_rmse_pct": rel_rmse_pct(ref_n, test_n),
        "electrolyte_rmse_pct": rel_rmse_pct(ref_e, test_e),
        "solid_cathode_rmse_pct": rel_rmse_pct(ref_p, test_p),
    }


def _compare(model, time, h1_states, h1_voltages, fast_states, fast_voltages, targets, csv_path=None):
    rows = []
    for target in targets:
        voltage_err = float((fast_voltages[target] - h1_voltages[target]).detach().cpu())
        row = {
            "offset": target,
            "time_s": float(time[target]),
            "voltage_h1_V": float(h1_voltages[target].detach().cpu()),
            "voltage_fast_V": float(fast_voltages[target].detach().cpu()),
            "voltage_error_V": voltage_err,
        }
        row.update(_region_rmse_pct(model, h1_states[target], fast_states[target]))
        rows.append(row)

    df = pd.DataFrame(rows)
    if csv_path:
        df.to_csv(csv_path, index=False)
    return df


def _current_total_check(current, fast_effective_current, targets):
    max_target = max(targets)
    original_plot_total = current[:max_target + 1].sum()
    fast_plot_total = fast_effective_current[:max_target + 1].sum()
    original_step_total = current[1:max_target + 1].sum()
    fast_step_total = fast_effective_current[1:max_target + 1].sum()

    segment_rows = []
    previous = 0
    for target in targets:
        original_segment = current[previous + 1:target + 1].sum()
        fast_segment = fast_effective_current[previous + 1:target + 1].sum()
        segment_rows.append({
            "from": previous,
            "to": target,
            "original_A_s": float(original_segment.detach().cpu()),
            "fast_A_s": float(fast_segment.detach().cpu()),
            "diff_A_s": float((fast_segment - original_segment).detach().cpu()),
        })
        previous = target

    segment_df = pd.DataFrame(segment_rows)
    return {
        "plot_original_A_s": float(original_plot_total.detach().cpu()),
        "plot_fast_A_s": float(fast_plot_total.detach().cpu()),
        "plot_diff_A_s": float((fast_plot_total - original_plot_total).detach().cpu()),
        "step_original_A_s": float(original_step_total.detach().cpu()),
        "step_fast_A_s": float(fast_step_total.detach().cpu()),
        "step_diff_A_s": float((fast_step_total - original_step_total).detach().cpu()),
        "segment_max_abs_diff_A_s": float(segment_df["diff_A_s"].abs().max()) if not segment_df.empty else 0.0,
        "segments": segment_df,
    }


def _stack_states(states, indices):
    return torch.cat([states[int(idx)] for idx in indices], dim=0)


def _stack_voltages(voltages, indices):
    return torch.stack([voltages[int(idx)] for idx in indices]).detach().cpu().numpy()


def _state_observables(model, states, current, indices):
    x = _stack_states(states, indices)
    i = current[indices]
    with torch.no_grad():
        c_n, c_p = model.calc_css_and_cs0(x, i)
        _, c_e, _ = model.unpack_state(x)
        weights = model.radial_average_weights.to(device=c_n.device, dtype=c_n.dtype)
        out = {
            "c_s_n_surf": c_n[:, -1],
            "c_s_p_surf": c_p[:, -1],
            "c_s_n_avg": c_n @ weights,
            "c_s_p_avg": c_p @ weights,
            "c_e_anode_cc": c_e[:, 0],
            "c_e_cathode_cc": c_e[:, -1],
        }
    return {key: value.detach().cpu().numpy() for key, value in out.items()}


def _plot_rollout_comparison(
    model,
    time,
    current,
    fast_effective_current,
    h1_states,
    h1_voltages,
    fast_states,
    fast_voltages,
    targets,
    output_dir,
    stem,
):
    import matplotlib.pyplot as plt

    max_target = max(targets)
    all_idx = np.arange(max_target + 1, dtype=int)
    target_idx = np.asarray(targets, dtype=int)
    t_all = time[all_idx]
    t_fast = time[target_idx]

    current_np = current[:max_target + 1, 0].detach().cpu().numpy()
    fast_current_np = fast_effective_current[:max_target + 1, 0].detach().cpu().numpy()
    h1_obs = _state_observables(model, h1_states, current, all_idx)
    fast_obs = _state_observables(model, fast_states, current, target_idx)
    h1_voltage = _stack_voltages(h1_voltages, all_idx)
    fast_voltage = _stack_voltages(fast_voltages, target_idx)

    fig, axes = plt.subplots(5, 1, figsize=(11.5, 13.0), sharex=True)
    axes[0].step(t_all, current_np, where="post", linewidth=1.4, color="#111111", label="h=1 current")
    axes[0].step(t_all, fast_current_np, where="post", linewidth=1.1, linestyle="--", color="#ff7f0e", label="fast effective current")
    axes[0].set_ylabel("current [A]")

    axes[1].plot(t_all, h1_voltage, linewidth=1.4, color="#111111", label="h=1")
    axes[1].plot(t_fast, fast_voltage, linestyle="None", marker="o", markersize=4, color="#2ca02c", label="fast targets")
    axes[1].set_ylabel("voltage [V]")

    axes[2].plot(t_all, h1_obs["c_s_n_surf"], linewidth=1.4, color="#1f77b4", label="h=1 anode")
    axes[2].plot(t_all, h1_obs["c_s_p_surf"], linewidth=1.4, color="#d62728", label="h=1 cathode")
    axes[2].plot(t_fast, fast_obs["c_s_n_surf"], linestyle="None", marker="o", markersize=4, color="#1f77b4", label="fast anode")
    axes[2].plot(t_fast, fast_obs["c_s_p_surf"], linestyle="None", marker="s", markersize=4, color="#d62728", label="fast cathode")
    axes[2].set_ylabel("surface c_s [mol/m3]")

    axes[3].plot(t_all, h1_obs["c_s_n_avg"], linewidth=1.4, color="#1f77b4", label="h=1 anode")
    axes[3].plot(t_all, h1_obs["c_s_p_avg"], linewidth=1.4, color="#d62728", label="h=1 cathode")
    axes[3].plot(t_fast, fast_obs["c_s_n_avg"], linestyle="None", marker="o", markersize=4, color="#1f77b4", label="fast anode")
    axes[3].plot(t_fast, fast_obs["c_s_p_avg"], linestyle="None", marker="s", markersize=4, color="#d62728", label="fast cathode")
    axes[3].set_ylabel("average c_s [mol/m3]")

    axes[4].plot(t_all, h1_obs["c_e_anode_cc"], linewidth=1.4, color="#1f77b4", label="h=1 anode cc")
    axes[4].plot(t_all, h1_obs["c_e_cathode_cc"], linewidth=1.4, color="#d62728", label="h=1 cathode cc")
    axes[4].plot(t_fast, fast_obs["c_e_anode_cc"], linestyle="None", marker="o", markersize=4, color="#1f77b4", label="fast anode cc")
    axes[4].plot(t_fast, fast_obs["c_e_cathode_cc"], linestyle="None", marker="s", markersize=4, color="#d62728", label="fast cathode cc")
    axes[4].set_ylabel("c_e [mol/m3]")
    axes[4].set_xlabel("time [s]")

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}_trajectories.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    target_current = current[target_idx, 0].detach().cpu().numpy()
    fast_current_target = fast_effective_current[target_idx, 0].detach().cpu().numpy()
    h1_target_obs = {key: value[target_idx] for key, value in h1_obs.items()}

    fig, axes = plt.subplots(5, 1, figsize=(11.5, 13.0), sharex=True)
    axes[0].step(t_all, fast_current_np - current_np, where="post", linewidth=1.2, color="#ff7f0e")
    axes[0].set_ylabel("current diff [A]")

    axes[1].plot(t_fast, fast_voltage - h1_voltage[target_idx], marker="o", linewidth=1.2, color="#2ca02c")
    axes[1].set_ylabel("voltage diff [V]")

    axes[2].plot(t_fast, fast_obs["c_s_n_surf"] - h1_target_obs["c_s_n_surf"], marker="o", linewidth=1.2, color="#1f77b4", label="anode")
    axes[2].plot(t_fast, fast_obs["c_s_p_surf"] - h1_target_obs["c_s_p_surf"], marker="s", linewidth=1.2, color="#d62728", label="cathode")
    axes[2].set_ylabel("surface c_s diff [mol/m3]")

    axes[3].plot(t_fast, fast_obs["c_s_n_avg"] - h1_target_obs["c_s_n_avg"], marker="o", linewidth=1.2, color="#1f77b4", label="anode")
    axes[3].plot(t_fast, fast_obs["c_s_p_avg"] - h1_target_obs["c_s_p_avg"], marker="s", linewidth=1.2, color="#d62728", label="cathode")
    axes[3].set_ylabel("average c_s diff [mol/m3]")

    axes[4].plot(t_fast, fast_obs["c_e_anode_cc"] - h1_target_obs["c_e_anode_cc"], marker="o", linewidth=1.2, color="#1f77b4", label="anode cc")
    axes[4].plot(t_fast, fast_obs["c_e_cathode_cc"] - h1_target_obs["c_e_cathode_cc"], marker="s", linewidth=1.2, color="#d62728", label="cathode cc")
    axes[4].set_ylabel("c_e diff [mol/m3]")
    axes[4].set_xlabel("time [s]")

    for ax in axes:
        ax.axhline(0.0, color="#111111", linewidth=0.8, alpha=0.4)
        ax.grid(True, alpha=0.25)
    for ax in axes[2:]:
        ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}_differences.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _roll_selected_fast_mode(model, x0, current, targets, max_fast_h, solid_step, fast_mode):
    if fast_mode == "chained":
        return _roll_fast(model, x0, current, targets, max_fast_h, solid_step)
    if fast_mode == "direct-exact":
        return _roll_direct_exact_targets(model, x0, current, targets)
    raise ValueError(f"Unknown fast_mode: {fast_mode}")


def _plot_dynamic_profile_trajectories(model, p, device, profile_paths, targets, output_dir, max_fast_h, solid_step, fast_mode):
    checks = []
    for profile_path in profile_paths:
        path = Path(profile_path)
        time, x0, current, _ = _load_profile(path, p, device)
        if targets[-1] >= len(current):
            raise ValueError(f"largest requested offset {targets[-1]} exceeds profile length {len(current)} for {profile_path}")

        h1_states, h1_voltages = _roll_h1(model, x0, current, targets)
        fast_states, fast_voltages, fast_effective_current = _roll_selected_fast_mode(
            model,
            x0,
            current,
            targets,
            max_fast_h,
            solid_step,
            fast_mode,
        )
        checks.append((str(profile_path), _current_total_check(current, fast_effective_current, targets)))
        _plot_rollout_comparison(
            model,
            time,
            current,
            fast_effective_current,
            h1_states,
            h1_voltages,
            fast_states,
            fast_voltages,
            targets,
            output_dir,
            path.stem,
        )
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_spme_pinn.yaml")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint whose model_state/config should be used.")
    parser.add_argument("--profile", default="data/spme_val_DST_soh90.csv")
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 2, 5, 10, 20, 30, 40, 50, 60])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dynamic-profiles", nargs="+", default=DEFAULT_DYNAMIC_PROFILES)
    parser.add_argument("--no-dynamic-profiles", action="store_true")
    parser.add_argument("--max-fast-h", type=int, default=5, help="Maximum h used by each averaged-current fast step.")
    parser.add_argument(
        "--fast-solid-step",
        choices=["exact", "rk4"],
        default="exact",
        help="Solid update used for fast averaged-current steps. 'exact' avoids RK4 instability for coarse h.",
    )
    parser.add_argument(
        "--fast-mode",
        choices=["chained", "direct-exact"],
        default="chained",
        help="Use chained sparse rollout, or direct exact one-shot target predictions from x0.",
    )
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    device = _device(args.device)
    cfg, p, model = _build_model(args.config, args.checkpoint, device)
    time, x0, current, voltage_data = _load_profile(Path(args.profile), p, device)

    targets = sorted(set(args.steps))
    if targets[0] < 1:
        raise ValueError("--steps must contain positive offsets.")
    if targets[-1] >= len(current):
        raise ValueError(f"largest requested offset {targets[-1]} exceeds profile length {len(current)}")

    h1_states, h1_voltages = _roll_h1(model, x0, current, targets)
    fast_states, fast_voltages, fast_effective_current = _roll_selected_fast_mode(
        model,
        x0,
        current,
        targets,
        args.max_fast_h,
        args.fast_solid_step,
        args.fast_mode,
    )
    df = _compare(model, time, h1_states, h1_voltages, fast_states, fast_voltages, targets, args.output_csv)
    current_check = _current_total_check(current, fast_effective_current, targets)

    output_dir = Path(args.output_dir) if args.output_dir else Path("fast_rollout_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.profile).stem
    _plot_rollout_comparison(
        model,
        time,
        current,
        fast_effective_current,
        h1_states,
        h1_voltages,
        fast_states,
        fast_voltages,
        targets,
        output_dir,
        stem,
    )
    dynamic_current_checks = []
    if not args.no_dynamic_profiles:
        dynamic_current_checks = _plot_dynamic_profile_trajectories(
            model,
            p,
            device,
            args.dynamic_profiles,
            targets,
            output_dir,
            args.max_fast_h,
            args.fast_solid_step,
            args.fast_mode,
        )

    print(f"profile: {args.profile}")
    print(f"config h: {cfg.get('h', 1.0)}; comparison forces h=1 for reference single steps")
    print(f"max fast averaged-step h: {args.max_fast_h}")
    print(f"fast solid step: {args.fast_solid_step}")
    print(f"fast mode: {args.fast_mode}")
    if args.checkpoint:
        print(f"checkpoint: {args.checkpoint}")
    print()
    print(df.to_string(index=False, float_format=lambda x: f"{x:.8e}"))
    print()
    print(f"max |voltage_fast - voltage_h1|: {df['voltage_error_V'].abs().max():.8e} V")
    print(f"max solid anode RMSE difference: {df['solid_anode_rmse_pct'].max():.8e} %")
    print(f"max electrolyte RMSE difference: {df['electrolyte_rmse_pct'].max():.8e} %")
    print(f"max solid cathode RMSE difference: {df['solid_cathode_rmse_pct'].max():.8e} %")
    print()
    print("Current integral check, h=1 s:")
    print(
        "  plotted samples 0..{target}: original={orig:.12e} A*s, fast={fast:.12e} A*s, diff={diff:.3e} A*s".format(
            target=max(targets),
            orig=current_check["plot_original_A_s"],
            fast=current_check["plot_fast_A_s"],
            diff=current_check["plot_diff_A_s"],
        )
    )
    print(
        "  rollout steps 1..{target}:  original={orig:.12e} A*s, fast={fast:.12e} A*s, diff={diff:.3e} A*s".format(
            target=max(targets),
            orig=current_check["step_original_A_s"],
            fast=current_check["step_fast_A_s"],
            diff=current_check["step_diff_A_s"],
        )
    )
    print(f"  max per-fast-segment diff: {current_check['segment_max_abs_diff_A_s']:.3e} A*s")
    if dynamic_current_checks:
        print()
        print("Dynamic profile current integral checks:")
        for profile_path, check in dynamic_current_checks:
            print(
                "  {name}: step diff={step:.3e} A*s, plot diff={plot:.3e} A*s, max segment diff={seg:.3e} A*s".format(
                    name=Path(profile_path).stem,
                    step=check["step_diff_A_s"],
                    plot=check["plot_diff_A_s"],
                    seg=check["segment_max_abs_diff_A_s"],
                )
            )
    print()
    print("Note: voltage_data is not used as truth here; this compares two rollout schemes from the same x(0).")
    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()
