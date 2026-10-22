import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from electrolyte_fvm import (
    electrolyte_activity,
    electrolyte_boundary_concentrations,
    electrolyte_conductivity,
    electrolyte_region_means,
)
from init_params import InitParamsSPMe
from spme_model import SPMeModel, ref_potential_anode, ref_potential_cathode


LGM50_DIR = ROOT / "LGM50"
DIAGNOSTIC_DIR = ROOT / "results" / "spme_matlab_cross_run"
MPL_CACHE_DIR = ROOT / ".matplotlib-cache"


def _base_cfg():
    return {
        "n_r": 15,
        "n_x_n": 10,
        "n_x_s": 5,
        "n_x_p": 10,
        "h": 1.0,
        "integ_type": 0,
        "ce_min": 200.0,
        "ce_max": 2000.0,
        "temperature_K": 298.15,
        "use_electrolyte_activity": False,
        "electrolyte_integrator": "implicit",
        "electrolyte_substeps": 1,
        "estimate_delta_sei": False,
        "nLi_s_fresh": 0.2840,
        "nLi_s_init": 0.2840,
        "nLi_e_init": 0.0053677182,
        "delta_sei_init": 0.0,
        "delta_sei_lower": 0.0,
        "delta_sei_upper": 2.0e-6,
        "delta_sei_true": 0.0,
        "epsilon_e_n_true": 0.25,
        "OneC": 5.0,
    }


def _init_cs_from_voltage(p, voltage):
    x_low = 0.2 * p.c_s_p_max
    x_high = p.c_s_p_max
    csp = 0.6 * p.c_s_p_max
    csn = p.csn0
    for _ in range(50):
        delta_n_li_p = (csp - p.csp0) * p.epsilon_s_p * p.L_p * p.Area_p
        csn = p.csn0 - delta_n_li_p / (p.epsilon_s_n * p.L_n * p.Area_n)
        theta_p = torch.tensor([[csp / p.c_s_p_max]], dtype=torch.float64)
        theta_n = torch.tensor([[csn / p.c_s_n_max]], dtype=torch.float64)
        residual = (ref_potential_cathode(theta_p) - ref_potential_anode(theta_n)).item() - voltage
        if abs(residual) <= 1e-7:
            break
        if residual <= 0:
            x_high = csp
        else:
            x_low = csp
        csp = 0.5 * (x_low + x_high)
    return csn, csp


def _solid_concentrations_from_soc(p, soc):
    csn0, csp0 = _init_cs_from_voltage(p, p.volt_min)
    csn100, csp100 = _init_cs_from_voltage(p, p.volt_max)
    return (
        csn0 + soc * (csn100 - csn0),
        csp0 + soc * (csp100 - csp0),
    )


def _cc_rest_cc_profile(one_c_current, dt):
    discharge_steps = int(round(3600 / dt))
    rest_steps = int(round(10 * 60 / dt))
    charge_steps = int(round(3600 / dt))
    current = np.empty(discharge_steps + rest_steps + charge_steps + 1, dtype=np.float64)
    current[:discharge_steps] = one_c_current
    current[discharge_steps:discharge_steps + rest_steps] = 0.0
    current[discharge_steps + rest_steps:] = -one_c_current
    return current


def _find_matlab_executable():
    explicit = os.environ.get("MATLAB_EXECUTABLE")
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    on_path = shutil.which("matlab")
    if on_path:
        candidates.append(Path(on_path))
    candidates.extend(sorted(Path("/Applications").glob("MATLAB_*.app/bin/matlab"), reverse=True))
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    searched = ", ".join(str(path) for path in candidates) if candidates else "no candidates"
    raise AssertionError(
        "Matlab executable is required for this test. "
        "Set MATLAB_EXECUTABLE, add matlab to PATH, or install Matlab under /Applications. "
        f"Searched: {searched}"
    )


def _python_to_matlab_state(x_py, p):
    c_n = x_py[: p.n_cs_n]
    c_e = x_py[p.n_cs_n : p.n_cs_n + p.n_ce]
    c_p = x_py[p.n_cs_n + p.n_ce :]
    return np.concatenate([c_n, c_p, c_e])


def _matlab_to_python_state(x_matlab, p):
    c_n = x_matlab[: p.n_cs_n]
    c_p = x_matlab[p.n_cs_n : p.n_cs_n + p.n_cs_p]
    c_e = x_matlab[p.n_cs_n + p.n_cs_p :]
    return np.concatenate([c_n, c_e, c_p])


def _run_python_spme(model, x0, current_profile):
    x = x0.clone()
    current_tensor = torch.as_tensor(current_profile, dtype=torch.float64).view(-1, 1)
    voltage = []
    states = [x.detach().clone()]
    for k in range(current_tensor.shape[0]):
        current = current_tensor[k:k + 1]
        c_s_n, c_e, c_s_p = model.unpack_state(x)
        c_n, c_p = model.calc_css_and_cs0(x, current)

        if not torch.isfinite(c_e).all():
            raise RuntimeError(f"Non-finite c_e before voltage at k={k}")

        if torch.any(c_e <= 0):
            print("k =", k)
            print("current =", current.squeeze().item())
            print("min c_e =", c_e.min().item())
            print("max c_e =", c_e.max().item())
            raise RuntimeError("Non-positive electrolyte concentration before voltage")

        voltage.append(model.calc_voltage(c_n[:, -1], c_p[:, -1], c_e, current).reshape(1).detach())
        if k < len(current_profile) - 1:
            x_next = model.step(x, current)
            _, c_e_next, _ = model.unpack_state(x_next)

            if torch.any(c_e_next <= 0) or not torch.isfinite(c_e_next).all():
                print("bad step k =", k)
                print("current =", current.squeeze().item())
                print("min c_e_next =", c_e_next.min().item())
                print("max c_e_next =", c_e_next.max().item())
                print("dc_e/dt approx min =", ((c_e_next - c_e) / model.h).min().item())
                print("dc_e/dt approx max =", ((c_e_next - c_e) / model.h).max().item())
                raise RuntimeError("Electrolyte integration produced invalid c_e")

            x = x_next

            states.append(x.detach().clone())
    return torch.cat(voltage).cpu().numpy(), torch.cat(states, dim=0).cpu().numpy()


def _torch_col(value, dtype=torch.float64):
    return torch.tensor([[value]], dtype=dtype)


def _python_voltage_components(model, css_n, css_p, c_e, current):
    p = model.p
    css_n = css_n.reshape(-1, 1).clamp(1.0, p.c_s_n_max * 0.999)
    css_p = css_p.reshape(-1, 1).clamp(1.0, p.c_s_p_max * 0.999)
    current = current.reshape(-1, 1)

    ce0n, cens, cesp, ce0p, _ = electrolyte_boundary_concentrations(c_e, p)
    cen_bar, ces_bar, cep_bar = electrolyte_region_means(c_e, p)

    arrh_kn = torch.exp(torch.as_tensor(p.E_kn / p.R * (1 / p.T_ref - 1 / p.T_amb), dtype=css_n.dtype))
    arrh_kp = torch.exp(torch.as_tensor(p.E_kp / p.R * (1 / p.T_ref - 1 / p.T_amb), dtype=css_n.dtype))
    i_0n = p.k_n0 * arrh_kn * ((p.c_s_n_max - css_n) * css_n * cen_bar) ** p.alph
    i_0p = p.k_p0 * arrh_kp * ((p.c_s_p_max - css_p) * css_p * cep_bar) ** p.alph

    rtaf = (p.R * p.T_amb) / (p.alph * p.Faraday)
    theta_n = css_n / p.c_s_n_max
    theta_p = css_p / p.c_s_p_max
    v_no_vce = (
        rtaf * torch.arcsinh(-current / (2 * p.a_s_p * p.Area_p * p.L_p * i_0p))
        - rtaf * torch.arcsinh(current / (2 * p.a_s_n * p.Area_n * p.L_n * i_0n))
        + ref_potential_cathode(theta_p)
        - ref_potential_anode(theta_n)
        - (p.R_f_n / (p.a_s_n * p.L_n * p.Area_n) + p.R_f_p / (p.a_s_p * p.L_p * p.Area_p)) * current
    )

    kap_n_eff = electrolyte_conductivity(cen_bar, p) * (p.epsilon_e_n ** p.brug)
    kap_s_eff = electrolyte_conductivity(ces_bar, p) * (p.epsilon_e_s ** p.brug)
    kap_p_eff = electrolyte_conductivity(cep_bar, p) * (p.epsilon_e_p ** p.brug)
    v_cond = (p.L_n / (2 * kap_n_eff) + p.L_s / kap_s_eff + p.L_p / (2 * kap_p_eff)) * current

    if model.cfg.get("use_electrolyte_activity", False):
        dfca_n = electrolyte_activity(cen_bar, p)
        dfca_s = electrolyte_activity(ces_bar, p)
        dfca_p = electrolyte_activity(cep_bar, p)
    else:
        dfca_n = torch.zeros_like(cen_bar)
        dfca_s = torch.zeros_like(ces_bar)
        dfca_p = torch.zeros_like(cep_bar)

    v_polar = (2 * p.R * p.T_amb / p.Faraday) * (
        (1 + dfca_n) * (torch.log(cens.clamp_min(1.0)) - torch.log(ce0n.clamp_min(1.0)))
        + (1 + dfca_s) * (torch.log(cesp.clamp_min(1.0)) - torch.log(cens.clamp_min(1.0)))
        + (1 + dfca_p) * (torch.log(ce0p.clamp_min(1.0)) - torch.log(cesp.clamp_min(1.0)))
    )
    return v_no_vce, v_cond, v_polar


def _python_cex_region_means(c_e, p):
    _, _, _, _, c_ex = electrolyte_boundary_concentrations(c_e, p)
    cen_bar = c_ex[:, : p.Nxn + 1].mean(dim=1, keepdim=True)
    ces_bar = c_ex[:, p.Nxn : p.Nxn + p.Nxs + 1].mean(dim=1, keepdim=True)
    cep_bar = c_ex[:, p.Nxn + p.Nxs : p.Nxn + p.Nxs + p.Nxp + 1].mean(dim=1, keepdim=True)
    return cen_bar, ces_bar, cep_bar


def _python_diagnostics(model, states, current_profile):
    names = [
        "c_ss_n", "c_ss_p", "cen_bar", "ces_bar", "cep_bar",
        "ce0n", "cens", "cesp", "ce0p",
        "V_noVCE", "V_electrolyteCond", "V_electrolytePolar",
    ]
    rows = []
    c_ex_rows = []
    for x_np, current_value in zip(states, current_profile):
        x = torch.tensor(x_np.reshape(1, -1), dtype=torch.float64)
        current = _torch_col(current_value)
        _, c_e, _ = model.unpack_state(x)
        c_n, c_p = model.calc_css_and_cs0(x, current)
        ce0n, cens, cesp, ce0p, c_ex = electrolyte_boundary_concentrations(c_e, model.p)
        cen_bar, ces_bar, cep_bar = _python_cex_region_means(c_e, model.p)
        v_no_vce, v_cond, v_polar = _python_voltage_components(model, c_n[:, -1], c_p[:, -1], c_e, current)
        rows.append([
            c_n[:, -1].item(), c_p[:, -1].item(),
            cen_bar.item(), ces_bar.item(), cep_bar.item(),
            ce0n.item(), cens.item(), cesp.item(), ce0p.item(),
            v_no_vce.item(), v_cond.item(), v_polar.item(),
        ])
        c_ex_rows.append(c_ex.squeeze(0).detach().cpu().numpy())
    return names, np.asarray(rows), np.asarray(c_ex_rows)


def _write_csv(path, header, values):
    values = np.asarray(values)
    fmt = "%s" if values.dtype.kind in {"O", "U", "S"} else "%.18e"
    np.savetxt(path, values, delimiter=",", header=",".join(header), comments="", fmt=fmt)


def _write_matplotlib_report(
    output_base, current, soc_est, v_py, v_matlab, x_py, x_matlab,
    diag_names, diag_py, diag_matlab, cex_py, cex_matlab, delta_t, timing,
):
    MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(len(current), dtype=float) * delta_t
    idx = {name: i for i, name in enumerate(diag_names)}
    terminal_state_diff = x_py[-1] - x_matlab[-1]
    x_state = np.arange(x_py.shape[1], dtype=float)
    x_ce = np.arange(cex_py.shape[1], dtype=float)
    voltage_diff_mv = 1e3 * (v_py - v_matlab)

    fig, axes = plt.subplots(7, 2, figsize=(15, 24), constrained_layout=True)
    axes = axes.ravel()
    fig.suptitle("SPMe Python vs Matlab Cross-Run Diagnostics", fontsize=16)

    axes[0].step(t, current, where="post", color="black")
    axes[0].set_title("Current Profile")
    axes[0].set_ylabel("Current [A]")
    ax_soc = axes[0].twinx()
    ax_soc.plot(t, soc_est, color="tab:green", alpha=0.7, label="estimated SoC")
    ax_soc.set_ylabel("Estimated SoC")
    timing_ax = axes[0].inset_axes([0.55, 0.12, 0.4, 0.36])
    timing_labels = ["Python\nrollout", "Matlab\nloop", "Matlab\nprocess"]
    timing_values = [
        timing["python_rollout_seconds"],
        timing["matlab_sim_seconds"],
        timing["matlab_subprocess_seconds"],
    ]
    timing_x = np.arange(len(timing_labels))
    timing_ax.bar(timing_x, timing_values, color=["tab:blue", "tab:red", "0.45"])
    timing_ax.set_xticks(timing_x)
    timing_ax.set_xticklabels(timing_labels)
    timing_ax.set_title("runtime [s]", fontsize=8)
    timing_ax.tick_params(axis="both", labelsize=7)

    axes[1].plot(t, voltage_diff_mv, color="tab:red")
    axes[1].axhline(0.0, color="0.4", linewidth=0.8)
    axes[1].set_title("Voltage Difference")
    axes[1].set_ylabel("Python - Matlab [mV]")

    axes[2].plot(t, v_py, color="tab:blue")
    axes[2].set_title("Python Terminal Voltage")
    axes[2].set_ylabel("Voltage [V]")

    axes[3].plot(t, v_matlab, color="tab:red")
    axes[3].set_title("Matlab Terminal Voltage")
    axes[3].set_ylabel("Voltage [V]")

    axes[4].plot(t, diag_py[:, idx["c_ss_n"]], label="c_ss_n")
    axes[4].plot(t, diag_py[:, idx["c_ss_p"]], label="c_ss_p")
    axes[4].set_title("Python Surface Solid Concentrations")
    axes[4].set_ylabel("mol/m3")
    axes[4].legend(fontsize=8)

    axes[5].plot(t, diag_matlab[:, idx["c_ss_n"]], label="c_ss_n")
    axes[5].plot(t, diag_matlab[:, idx["c_ss_p"]], label="c_ss_p")
    axes[5].set_title("Matlab Surface Solid Concentrations")
    axes[5].set_ylabel("mol/m3")
    axes[5].legend(fontsize=8)

    for name in ["cen_bar", "ces_bar", "cep_bar"]:
        axes[6].plot(t, diag_py[:, idx[name]], label=name)
    axes[6].set_title("Python Electrolyte Region Means")
    axes[6].set_ylabel("mol/m3")
    axes[6].legend(fontsize=8)

    for name in ["cen_bar", "ces_bar", "cep_bar"]:
        axes[7].plot(t, diag_matlab[:, idx[name]], label=name)
    axes[7].set_title("Matlab Electrolyte Region Means")
    axes[7].set_ylabel("mol/m3")
    axes[7].legend(fontsize=8)

    for name in ["ce0n", "cens", "cesp", "ce0p"]:
        axes[8].plot(t, diag_py[:, idx[name]], label=name)
    axes[8].set_title("Python Electrolyte Boundaries")
    axes[8].set_ylabel("mol/m3")
    axes[8].legend(fontsize=8)

    for name in ["ce0n", "cens", "cesp", "ce0p"]:
        axes[9].plot(t, diag_matlab[:, idx[name]], label=name)
    axes[9].set_title("Matlab Electrolyte Boundaries")
    axes[9].set_ylabel("mol/m3")
    axes[9].legend(fontsize=8)

    for name in ["V_noVCE", "V_electrolyteCond", "V_electrolytePolar"]:
        axes[10].plot(t, diag_py[:, idx[name]], label=name)
    axes[10].set_title("Python Voltage Components")
    axes[10].set_ylabel("Voltage [V]")
    axes[10].legend(fontsize=8)

    for name in ["V_noVCE", "V_electrolyteCond", "V_electrolytePolar"]:
        axes[11].plot(t, diag_matlab[:, idx[name]], label=name)
    axes[11].set_title("Matlab Voltage Components")
    axes[11].set_ylabel("Voltage [V]")
    axes[11].legend(fontsize=8)

    axes[12].plot(x_ce, cex_py[-1], color="tab:blue")
    axes[12].set_title("Python Terminal Electrolyte Profile")
    axes[12].set_xlabel("Full electrolyte node index")
    axes[12].set_ylabel("mol/m3")

    axes[13].plot(x_ce, cex_matlab[-1], color="tab:red")
    axes[13].set_title("Matlab Terminal Electrolyte Profile")
    axes[13].set_xlabel("Full electrolyte node index")
    axes[13].set_ylabel("mol/m3")

    inset = axes[1].inset_axes([0.55, 0.48, 0.4, 0.42])
    inset.plot(x_state, terminal_state_diff, color="tab:purple", linewidth=0.9)
    inset.axhline(0.0, color="0.4", linewidth=0.7)
    inset.set_title("terminal dx", fontsize=8)
    inset.tick_params(labelsize=7)

    # Keep a compact numeric summary in the difference panel.
    err_text = (
        f"max |dV| = {np.nanmax(np.abs(voltage_diff_mv)):.4g} mV\n"
        f"max |dx| = {np.nanmax(np.abs(terminal_state_diff)):.4g}\n"
        f"L2 dx = {np.linalg.norm(np.nan_to_num(terminal_state_diff, nan=0.0)):.4g}\n"
        f"Python rollout = {timing['python_rollout_seconds']:.3f} s\n"
        f"Matlab loop = {timing['matlab_sim_seconds']:.3f} s"
    )
    axes[1].text(0.02, 0.95, err_text, transform=axes[1].transAxes, va="top", fontsize=9,
                 bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"})

    for ax in axes:
        ax.grid(True, alpha=0.3)
        if ax not in (axes[12], axes[13]):
            ax.set_xlabel("Time [s]")

    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


class TestSPMeMatlabCrossRun(unittest.TestCase):
    """Run the Python SPMe and Matlab SPMe from the same state/current data."""

    def setUp(self):
        if not LGM50_DIR.exists():
            raise AssertionError(f"LGM50 Matlab folder is required for this test: {LGM50_DIR}")
        self.matlab = _find_matlab_executable()

    def test_python_spme_matches_matlab_spme_same_initial_state_and_current(self):
        torch.set_default_dtype(torch.float64)
        cfg = _base_cfg()
        p = InitParamsSPMe(cfg)
        p.csn0, p.csp0 = _solid_concentrations_from_soc(p, 1.0)
        model = SPMeModel(p, cfg)

        current_profile = _cc_rest_cc_profile(p.OneC, p.delta_t)
        soc_est = 1.0 - np.concatenate([[0.0], np.cumsum(current_profile[:-1])]) * p.delta_t / (p.OneC * 3600.0)
        x0_py = model.initial_state(batch_size=1, dtype=torch.float64)
        python_start = time.perf_counter()
        v_py, x_py = _run_python_spme(model, x0_py, current_profile)
        python_rollout_seconds = time.perf_counter() - python_start

        with tempfile.TemporaryDirectory(prefix="spme_matlab_cross_run_") as tmp:
            tmp_path = Path(tmp)
            np.savetxt(tmp_path / "current.csv", current_profile.reshape(-1, 1), delimiter=",")
            np.savetxt(
                tmp_path / "x0_matlab.csv",
                _python_to_matlab_state(x0_py.squeeze(0).detach().cpu().numpy(), p).reshape(-1, 1),
                delimiter=",",
            )

            runner_path = tmp_path / "run_matlab_spme_cross_run.m"
            runner_path.write_text(
                textwrap.dedent(
                    f"""
                    clear; clc;
                    repo_dir = '{ROOT.as_posix()}';
                    lgm50_dir = fullfile(repo_dir, 'LGM50');
                    addpath(lgm50_dir);
                    addpath(fullfile(lgm50_dir, 'SPMe'));
                    addpath(fullfile(lgm50_dir, 'param'));
                    startup;

                    current = readmatrix(fullfile('{tmp_path.as_posix()}', 'current.csv'));
                    x0 = readmatrix(fullfile('{tmp_path.as_posix()}', 'x0_matlab.csv'));
                    current = current(:);
                    x0 = x0(:);

                    init_soc = 1.0;
                    p = load_SPMe_params(init_soc, 25, {p.Nr}, {p.delta_t:.17g}, {p.Nxn}, {p.Nxs}, {p.Nxp}, ...
                        true, 'none', 'fvm', false, false, false);
                    p.csn0 = {p.csn0:.17g};
                    p.csp0 = {p.csp0:.17g};
                    p.x0 = x0;

                    model = SPMe(p, 'idas');
                    n_steps = numel(current);
                    x = x0;
                    voltage = zeros(n_steps, 1);
                    states = zeros(numel(x0), n_steps);
                    diagnostics = zeros(n_steps, 12);
                    c_ex_hist = zeros(p.Nx + 1, n_steps);
                    states(:, 1) = x0;
                    matlab_sim_tic = tic;
                    for k = 1:n_steps
                        y = model.h('x', x, 'p', current(k));
                        y = full(y.y);
                        voltage(k) = y(1);
                        ss = model.f_ss('x', x, 'u', current(k));
                        ce = model.f_cex('x', x);
                        c_ex = full(ce.c_ex);
                        c_ex_hist(:, k) = c_ex;
                        cen_bar = sum(c_ex(1:p.Nxn+1)) / (p.Nxn+1);
                        ces_bar = sum(c_ex((p.Nxn+1):(p.Nxn+p.Nxs+1))) / (p.Nxs+1);
                        cep_bar = sum(c_ex((p.Nxn+p.Nxs+1):(p.Nxn+p.Nxs+p.Nxp+1))) / (p.Nxp+1);
                        [~, ~, ~, ~, V_noVCE, V_electrolyteCond, V_electrolytePolar] = ...
                            model.potentials(x, current(k));
                        diagnostics(k, :) = [full(ss.c_ss_n), full(ss.c_ss_p), ...
                            cen_bar, ces_bar, cep_bar, ...
                            c_ex(1), c_ex(p.Nxn+1), c_ex(p.Nxn+p.Nxs+1), c_ex(end), ...
                            V_noVCE, V_electrolyteCond, V_electrolytePolar];
                        if k < n_steps
                            [x, ~] = model.step(x, current(k));
                            states(:, k + 1) = x;
                        end
                    end
                    matlab_sim_seconds = toc(matlab_sim_tic);

                    writematrix(voltage, fullfile('{tmp_path.as_posix()}', 'voltage_matlab.csv'));
                    writematrix(states.', fullfile('{tmp_path.as_posix()}', 'states_matlab.csv'));
                    writematrix(diagnostics, fullfile('{tmp_path.as_posix()}', 'diagnostics_matlab.csv'));
                    writematrix(c_ex_hist.', fullfile('{tmp_path.as_posix()}', 'cex_matlab.csv'));
                    writematrix(matlab_sim_seconds, fullfile('{tmp_path.as_posix()}', 'matlab_sim_seconds.csv'));
                    """
                ),
                encoding="utf-8",
            )

            matlab_subprocess_start = time.perf_counter()
            result = subprocess.run(
                [self.matlab, "-nodisplay", "-nosplash", "-nodesktop", "-batch", f"run('{runner_path.as_posix()}')"],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=3600,
            )
            matlab_subprocess_seconds = time.perf_counter() - matlab_subprocess_start
            if result.returncode != 0:
                self.fail(
                    "Matlab SPMe run failed.\n"
                    f"Executable: {self.matlab}\n"
                    f"Return code: {result.returncode}\n"
                    f"STDOUT:\n{result.stdout}\n"
                    f"STDERR:\n{result.stderr}"
                )

            v_matlab = np.loadtxt(tmp_path / "voltage_matlab.csv", delimiter=",").reshape(-1)
            x_matlab_raw = np.loadtxt(tmp_path / "states_matlab.csv", delimiter=",")
            x_matlab = np.vstack([_matlab_to_python_state(row, p) for row in np.atleast_2d(x_matlab_raw)])
            diag_matlab = np.atleast_2d(np.loadtxt(tmp_path / "diagnostics_matlab.csv", delimiter=","))
            cex_matlab = np.atleast_2d(np.loadtxt(tmp_path / "cex_matlab.csv", delimiter=","))
            matlab_sim_seconds = float(np.loadtxt(tmp_path / "matlab_sim_seconds.csv", delimiter=","))

        diag_names, diag_py, cex_py = _python_diagnostics(model, x_py, current_profile)
        timing = {
            "python_rollout_seconds": python_rollout_seconds,
            "matlab_sim_seconds": matlab_sim_seconds,
            "matlab_subprocess_seconds": matlab_subprocess_seconds,
            "python_steps_per_second": len(current_profile) / python_rollout_seconds,
            "matlab_sim_steps_per_second": len(current_profile) / matlab_sim_seconds,
            "matlab_subprocess_steps_per_second": len(current_profile) / matlab_subprocess_seconds,
            "matlab_sim_over_python_speed_ratio": matlab_sim_seconds / python_rollout_seconds,
            "matlab_subprocess_over_python_speed_ratio": matlab_subprocess_seconds / python_rollout_seconds,
        }
        DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
        _write_csv(
            DIAGNOSTIC_DIR / "time_current_voltage.csv",
            ["time_s", "current_A", "estimated_soc", "voltage_python_V", "voltage_matlab_V", "voltage_diff_mV"],
            np.column_stack([
                np.arange(len(current_profile), dtype=float) * p.delta_t,
                current_profile,
                soc_est,
                v_py,
                v_matlab,
                1e3 * (v_py - v_matlab),
            ]),
        )
        _write_csv(DIAGNOSTIC_DIR / "diagnostics_python.csv", diag_names, diag_py)
        _write_csv(DIAGNOSTIC_DIR / "diagnostics_matlab.csv", diag_names, diag_matlab)
        _write_csv(DIAGNOSTIC_DIR / "states_python.csv", [f"x{i}" for i in range(x_py.shape[1])], x_py)
        _write_csv(DIAGNOSTIC_DIR / "states_matlab_reordered.csv", [f"x{i}" for i in range(x_matlab.shape[1])], x_matlab)
        _write_csv(DIAGNOSTIC_DIR / "cex_python.csv", [f"cex{i}" for i in range(cex_py.shape[1])], cex_py)
        _write_csv(DIAGNOSTIC_DIR / "cex_matlab.csv", [f"cex{i}" for i in range(cex_matlab.shape[1])], cex_matlab)
        _write_csv(
            DIAGNOSTIC_DIR / "computational_time.csv",
            ["metric", "value"],
            np.asarray([[key, value] for key, value in timing.items()], dtype=object),
        )
        report_base = DIAGNOSTIC_DIR / "spme_matlab_cross_run_latest"
        png_path, pdf_path = _write_matplotlib_report(
            report_base, current_profile, soc_est, v_py, v_matlab, x_py, x_matlab,
            diag_names, diag_py, diag_matlab, cex_py, cex_matlab, p.delta_t, timing,
        )
        print(f"\nSPMe cross-run diagnostics written to: {png_path}")
        print(f"SPMe cross-run PDF report written to: {pdf_path}")

        np.testing.assert_allclose(v_py, v_matlab, atol=5e-5, rtol=5e-5)
        np.testing.assert_allclose(x_py[-1], x_matlab[-1], atol=5e-4, rtol=5e-6)


if __name__ == "__main__":
    unittest.main()
