#!/usr/bin/env python3
"""Benchmark fast-rollout solid integrators against full h=1 rollout.

This compares:
  - Euler solid update with max coarse h=2
  - RK4 solid update with max coarse h=3
  - exact matrix-exponential solid update with configurable max coarse h

The electrolyte update still uses the model's configured electrolyte integrator.
Errors are reported against the full h=1 rollout at the requested target offsets.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from compare_fast_rollout import (
    _build_model,
    _current_total_check,
    _device,
    _load_profile,
    _region_rmse_pct,
    _roll_h1,
    _state_observables,
    _stack_voltages,
    _step_with_h_fast,
    _temporary_h,
    _voltage,
)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _step_builtin_with_integrator(model, x, current, h, integ_type):
    old_type = model.integ_type
    model.integ_type = int(integ_type)
    try:
        with _temporary_h(model, h):
            return model.step(x, current)
    finally:
        model.integ_type = old_type


def _roll_mode(model, x0, current, targets, mode, max_h):
    if max_h < 1:
        raise ValueError("max_h must be >= 1")

    x = x0.clone()
    states = {0: x.detach().clone()}
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
            cursor = previous
            while target - cursor > 1:
                chunk_h = min(max_h, target - cursor - 1)
                next_cursor = cursor + chunk_h
                avg_current = current[cursor + 1:next_cursor + 1].mean(dim=0, keepdim=True)
                effective_current[cursor + 1:next_cursor + 1] = avg_current
                if mode == "euler":
                    x = _step_builtin_with_integrator(model, x, avg_current, chunk_h, integ_type=1)
                elif mode == "rk4":
                    x = _step_builtin_with_integrator(model, x, avg_current, chunk_h, integ_type=0)
                elif mode == "exact":
                    x = _step_with_h_fast(model, x, avg_current, chunk_h, solid_step="exact")
                else:
                    raise ValueError(f"Unknown mode: {mode}")
                cursor = next_cursor

            effective_current[target:target + 1] = current[target:target + 1]
            if mode == "euler":
                x = _step_builtin_with_integrator(model, x, current[target:target + 1], 1, integ_type=1)
            elif mode == "rk4":
                x = _step_builtin_with_integrator(model, x, current[target:target + 1], 1, integ_type=0)
            else:
                x = _step_with_h_fast(model, x, current[target:target + 1], 1, solid_step="exact")
            states[target] = x.detach().clone()
            voltages[target] = _voltage(model, x, current[target:target + 1]).detach().clone()
            previous = target

    return states, voltages, effective_current


def _time_rollout(fn, repeats, device):
    timings = []
    result = None
    for _ in range(repeats):
        _sync(device)
        t0 = time.perf_counter()
        result = fn()
        _sync(device)
        timings.append(time.perf_counter() - t0)
    return result, float(np.median(timings)), float(np.mean(timings))


def _variable_errors(model, current, targets, ref_states, ref_voltages, states, voltages):
    idx = np.asarray(targets, dtype=int)
    ref_obs = _state_observables(model, ref_states, current, idx)
    obs = _state_observables(model, states, current, idx)
    ref_v = _stack_voltages(ref_voltages, idx)
    v = _stack_voltages(voltages, idx)

    rows = []

    def add(name, pred, ref, unit):
        err = np.asarray(pred) - np.asarray(ref)
        rows.append({
            "variable": name,
            "unit": unit,
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "max_abs": float(np.max(np.abs(err))),
        })

    add("voltage", v, ref_v, "V")
    add("solid_anode_surface", obs["c_s_n_surf"], ref_obs["c_s_n_surf"], "mol/m3")
    add("solid_cathode_surface", obs["c_s_p_surf"], ref_obs["c_s_p_surf"], "mol/m3")
    add("solid_anode_average", obs["c_s_n_avg"], ref_obs["c_s_n_avg"], "mol/m3")
    add("solid_cathode_average", obs["c_s_p_avg"], ref_obs["c_s_p_avg"], "mol/m3")
    add("electrolyte_anode_cc", obs["c_e_anode_cc"], ref_obs["c_e_anode_cc"], "mol/m3")
    add("electrolyte_cathode_cc", obs["c_e_cathode_cc"], ref_obs["c_e_cathode_cc"], "mol/m3")
    return rows


def _target_state_rows(model, time_values, targets, ref_states, states, mode_name):
    rows = []
    for target in targets:
        row = {
            "mode": mode_name,
            "offset": target,
            "time_s": float(time_values[target]),
        }
        row.update(_region_rmse_pct(model, ref_states[target], states[target]))
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_spme_pinn.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--profile", default="data/spme_val_DST_soh90.csv")
    parser.add_argument("--steps", nargs="+", type=int, default=[120])
    parser.add_argument("--euler-h", type=int, default=2)
    parser.add_argument("--rk4-h", type=int, default=3)
    parser.add_argument("--exact-h", type=int, default=None, help="Default: largest requested target.")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="rollout_integrator_benchmark")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    device = _device(args.device)
    _, p, model = _build_model(args.config, args.checkpoint, device)
    time_values, x0, current, _ = _load_profile(Path(args.profile), p, device)

    targets = sorted(set(args.steps))
    if targets[0] < 1:
        raise ValueError("--steps must contain positive offsets.")
    if targets[-1] >= len(current):
        raise ValueError(f"largest requested offset {targets[-1]} exceeds profile length {len(current)}")
    exact_h = args.exact_h if args.exact_h is not None else max(targets)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_result, ref_median, ref_mean = _time_rollout(
        lambda: _roll_h1(model, x0, current, targets),
        args.repeats,
        device,
    )
    ref_states, ref_voltages = ref_result

    modes = [
        ("euler_h2", "euler", args.euler_h),
        ("rk4_h3", "rk4", args.rk4_h),
        (f"exact_h{exact_h}", "exact", exact_h),
    ]

    summary_rows = [{
        "mode": "reference_h1_rk4",
        "max_h": 1,
        "steps_to_target": max(targets),
        "median_sec": ref_median,
        "mean_sec": ref_mean,
        "speedup_vs_reference": 1.0,
        "max_voltage_abs_error_V": 0.0,
        "max_solid_anode_rmse_pct": 0.0,
        "max_electrolyte_rmse_pct": 0.0,
        "max_solid_cathode_rmse_pct": 0.0,
        "current_step_diff_A_s": 0.0,
        "current_segment_max_abs_diff_A_s": 0.0,
    }]
    state_rows = []
    variable_rows = []

    for mode_name, mode, max_h in modes:
        result, median_sec, mean_sec = _time_rollout(
            lambda mode=mode, max_h=max_h: _roll_mode(model, x0, current, targets, mode, max_h),
            args.repeats,
            device,
        )
        states, voltages, effective_current = result
        state_rows.extend(_target_state_rows(model, time_values, targets, ref_states, states, mode_name))
        for row in _variable_errors(model, current, targets, ref_states, ref_voltages, states, voltages):
            row["mode"] = mode_name
            variable_rows.append(row)

        state_df_for_mode = pd.DataFrame([row for row in state_rows if row["mode"] == mode_name])
        var_df_for_mode = pd.DataFrame([row for row in variable_rows if row["mode"] == mode_name])
        current_check = _current_total_check(current, effective_current, targets)
        summary_rows.append({
            "mode": mode_name,
            "max_h": max_h,
            "steps_to_target": int(np.ceil(max(targets) / max_h)),
            "median_sec": median_sec,
            "mean_sec": mean_sec,
            "speedup_vs_reference": ref_median / median_sec if median_sec > 0 else np.nan,
            "max_voltage_abs_error_V": float(var_df_for_mode.loc[var_df_for_mode["variable"] == "voltage", "max_abs"].max()),
            "max_solid_anode_rmse_pct": float(state_df_for_mode["solid_anode_rmse_pct"].max()),
            "max_electrolyte_rmse_pct": float(state_df_for_mode["electrolyte_rmse_pct"].max()),
            "max_solid_cathode_rmse_pct": float(state_df_for_mode["solid_cathode_rmse_pct"].max()),
            "current_step_diff_A_s": current_check["step_diff_A_s"],
            "current_segment_max_abs_diff_A_s": current_check["segment_max_abs_diff_A_s"],
        })

    summary_df = pd.DataFrame(summary_rows)
    state_df = pd.DataFrame(state_rows)
    variable_df = pd.DataFrame(variable_rows)

    stem = Path(args.profile).stem
    summary_path = output_dir / f"{stem}_integrator_summary.csv"
    state_path = output_dir / f"{stem}_integrator_state_errors.csv"
    variable_path = output_dir / f"{stem}_integrator_variable_errors.csv"
    summary_df.to_csv(summary_path, index=False)
    state_df.to_csv(state_path, index=False)
    variable_df.to_csv(variable_path, index=False)

    print(f"profile: {args.profile}")
    print(f"targets: {targets}")
    print(f"repeats: {args.repeats}")
    print()
    print("Speed and max error summary:")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.8e}"))
    print()
    print("Variable errors:")
    print(variable_df.to_string(index=False, float_format=lambda x: f"{x:.8e}"))
    print()
    print("State percent RMSE at targets:")
    print(state_df.to_string(index=False, float_format=lambda x: f"{x:.8e}"))
    print()
    print(f"Saved CSV files to: {output_dir}")


if __name__ == "__main__":
    main()
