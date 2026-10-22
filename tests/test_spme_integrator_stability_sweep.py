"""
Integrator stability/alignment sweep for PINN-SPMe electrolyte dynamics.

Purpose
-------
This test separates three issues:
1) MATLAB-FVM RHS vs Python-FVM RHS: already checked by test_electrolyte_fvm_alignment.py.
2) Python explicit RK4 time stepping vs MATLAB IDAS time stepping.
3) Whether the spiky interface concentrations (cens/cesp) are caused by explicit-time-step instability.

Run from repo root:
    python tests/test_spme_integrator_stability_sweep.py

Optional:
    MATLAB_EXECUTABLE=/Applications/MATLAB_R2024b.app/bin/matlab python tests/test_spme_integrator_stability_sweep.py

Expected interpretation
-----------------------
If larger electrolyte_substeps reduce cens/cesp/c_ex errors strongly, the equations are aligned
but explicit RK4 is under-resolved/unstable for the electrolyte separator/interface dynamics.

This file assumes the repo layout used by your current tests:
    PINN-SPMe/
      init_params.py
      spme_model.py
      electrolyte_fvm.py
      LGM50/SPMe/*.m
      LGM50/param/*.m
"""

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from init_params import InitParamsSPMe
from spme_model import SPMeModel
from electrolyte_fvm import electrolyte_boundary_concentrations


def find_matlab_executable() -> str:
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
    raise RuntimeError(
        "MATLAB executable not found. Set MATLAB_EXECUTABLE or install MATLAB under /Applications."
    )


def base_cfg(h: float, substeps: int) -> dict:
    return {
        "n_r": 15,
        "n_x_n": 10,
        "n_x_s": 5,
        "n_x_p": 10,
        "h": h,
        "integ_type": 0,  # RK4 in current SPMeModel
        "ce_min": 200.0,
        "ce_max": 2000.0,
        "temperature_K": 298.15,
        "use_electrolyte_activity": False,
        "electrolyte_substeps": substeps,
    }


def python_to_matlab_state(x_py: np.ndarray, p: InitParamsSPMe) -> np.ndarray:
    c_n = x_py[: p.n_cs_n]
    c_e = x_py[p.n_cs_n : p.n_cs_n + p.n_ce]
    c_p = x_py[p.n_cs_n + p.n_ce :]
    return np.concatenate([c_n, c_p, c_e])


def matlab_to_python_state(x_mat: np.ndarray, p: InitParamsSPMe) -> np.ndarray:
    c_n = x_mat[: p.n_cs_n]
    c_p = x_mat[p.n_cs_n : p.n_cs_n + p.n_cs_p]
    c_e = x_mat[p.n_cs_n + p.n_cs_p :]
    return np.concatenate([c_n, c_e, c_p])


def cex_from_python_state(model: SPMeModel, x_np: np.ndarray) -> np.ndarray:
    x = torch.tensor(x_np.reshape(1, -1), dtype=torch.float64)
    _, c_e, _ = model.unpack_state(x)
    _, _, _, _, c_ex = electrolyte_boundary_concentrations(c_e, model.p)
    return c_ex.squeeze(0).detach().cpu().numpy()


def boundary_summary(c_ex: np.ndarray, p: InitParamsSPMe) -> np.ndarray:
    # returns [ce0n, cens, cesp, ce0p]
    return np.array([
        c_ex[0],
        c_ex[p.Nxn],
        c_ex[p.Nxn + p.Nxs],
        c_ex[-1],
    ], dtype=float)


def run_python(model: SPMeModel, x0: np.ndarray, current: np.ndarray):
    x = torch.tensor(x0.reshape(1, -1), dtype=torch.float64)
    states = [x0.copy()]
    cex = [cex_from_python_state(model, x0)]
    for k, I in enumerate(current[:-1]):
        I_t = torch.tensor([[float(I)]], dtype=torch.float64)
        x = model.step(x, I_t)
        if not torch.isfinite(x).all():
            raise RuntimeError(f"Python state became non-finite at step {k}")
        states.append(x.squeeze(0).detach().cpu().numpy().copy())
        cex.append(cex_from_python_state(model, states[-1]))
    return np.asarray(states), np.asarray(cex)


def run_matlab_idas(matlab_exe: str, p: InitParamsSPMe, x0_py: np.ndarray, current: np.ndarray, h: float):
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        np.savetxt(td / "current.csv", current, delimiter=",")
        np.savetxt(td / "x0_matlab.csv", python_to_matlab_state(x0_py, p), delimiter=",")
        runner = td / "run_spme_idas.m"
        runner.write_text(
            textwrap.dedent(
                f"""
                clear; clc;
                repo_dir = '{ROOT.as_posix()}';
                lgm50_dir = fullfile(repo_dir, 'LGM50');
                addpath(lgm50_dir);
                addpath(fullfile(lgm50_dir, 'SPMe'));
                addpath(fullfile(lgm50_dir, 'param'));
                startup;

                current = readmatrix(fullfile('{td.as_posix()}', 'current.csv'));
                x0 = readmatrix(fullfile('{td.as_posix()}', 'x0_matlab.csv'));
                current = current(:);
                x0 = x0(:);

                p = load_SPMe_params(1.0, 25, {p.Nr}, {h:.17g}, {p.Nxn}, {p.Nxs}, {p.Nxp}, ...
                    true, 'none', 'fvm', false, false, false);
                p.csn0 = {p.csn0:.17g};
                p.csp0 = {p.csp0:.17g};
                p.x0 = x0;
                model = SPMe(p, 'idas');

                n_steps = numel(current);
                x = x0;
                states = zeros(numel(x0), n_steps);
                cex_hist = zeros(p.Nx + 1, n_steps);
                states(:, 1) = x;

                for k = 1:n_steps
                    ce = model.f_cex('x', x);
                    cex_hist(:, k) = full(ce.c_ex);
                    if k < n_steps
                        [x, ~] = model.step(x, current(k));
                        states(:, k + 1) = x;
                    end
                end

                writematrix(states.', fullfile('{td.as_posix()}', 'states_matlab.csv'));
                writematrix(cex_hist.', fullfile('{td.as_posix()}', 'cex_matlab.csv'));
                """
            ),
            encoding="utf-8",
        )
        res = subprocess.run(
            [matlab_exe, "-nodisplay", "-nosplash", "-nodesktop", "-batch", f"run('{runner.as_posix()}')"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=3600,
        )
        if res.returncode != 0:
            raise RuntimeError(f"MATLAB failed\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
        states_mat_raw = np.atleast_2d(np.loadtxt(td / "states_matlab.csv", delimiter=","))
        cex_mat = np.atleast_2d(np.loadtxt(td / "cex_matlab.csv", delimiter=","))
    states_py_order = np.vstack([matlab_to_python_state(row, p) for row in states_mat_raw])
    return states_py_order, cex_mat


def main():
    torch.set_default_dtype(torch.float64)

    h = float(os.environ.get("SPME_SWEEP_H", "0.1"))
    duration_s = float(os.environ.get("SPME_SWEEP_DURATION", "100.0"))
    current_amp = float(os.environ.get("SPME_SWEEP_CURRENT", "5.0"))
    substeps_list = [int(x) for x in os.environ.get("SPME_SWEEP_SUBSTEPS", "1,2,5,10,20,50,100").split(",")]

    # Use one parameter object for initial state and MATLAB metadata.
    p0 = InitParamsSPMe(base_cfg(h, substeps=1))
    model0 = SPMeModel(p0, base_cfg(h, substeps=1))
    x0 = model0.initial_state(batch_size=1, dtype=torch.float64).squeeze(0).detach().cpu().numpy()

    n_steps = int(round(duration_s / h)) + 1
    current = np.full(n_steps, current_amp, dtype=float)

    matlab = find_matlab_executable()
    print(f"Running MATLAB IDAS reference: h={h}, duration={duration_s}, steps={n_steps}, I={current_amp}")
    x_mat, cex_mat = run_matlab_idas(matlab, p0, x0, current, h)

    out_dir = ROOT / "results" / "spme_integrator_stability_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for substeps in substeps_list:
        cfg = base_cfg(h, substeps)
        p = InitParamsSPMe(cfg)
        model = SPMeModel(p, cfg)

        try:
            x_py, cex_py = run_python(model, x0, current)
            finite = np.isfinite(x_py).all() and np.isfinite(cex_py).all()
            if finite:
                b_py = np.vstack([boundary_summary(row, p) for row in cex_py])
                b_mat = np.vstack([boundary_summary(row, p) for row in cex_mat])
                max_ce_state = np.max(np.abs(x_py[:, p.n_cs_n:p.n_cs_n+p.n_ce] - x_mat[:, p.n_cs_n:p.n_cs_n+p.n_ce]))
                max_cex = np.max(np.abs(cex_py - cex_mat))
                max_boundary = np.max(np.abs(b_py - b_mat))
                final_boundary = b_py[-1] - b_mat[-1]
                status = "OK"
            else:
                max_ce_state = max_cex = max_boundary = np.nan
                final_boundary = np.full(4, np.nan)
                status = "NONFINITE"
        except Exception as exc:
            max_ce_state = max_cex = max_boundary = np.nan
            final_boundary = np.full(4, np.nan)
            status = f"ERROR: {exc}"

        rows.append([
            substeps,
            h / substeps,
            max_ce_state,
            max_cex,
            max_boundary,
            final_boundary[0],
            final_boundary[1],
            final_boundary[2],
            final_boundary[3],
            status,
        ])
        print(
            f"substeps={substeps:4d}, dt_sub={h/substeps:.3e}, "
            f"max|ce_state|={max_ce_state:.3e}, max|c_ex|={max_cex:.3e}, "
            f"max|boundary|={max_boundary:.3e}, status={status}"
        )

    header = [
        "substeps", "dt_sub_s", "max_abs_ce_state_molm3", "max_abs_cex_molm3",
        "max_abs_boundary_molm3", "final_err_ce0n", "final_err_cens",
        "final_err_cesp", "final_err_ce0p", "status",
    ]
    csv_path = out_dir / "integrator_stability_sweep.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")

    print(f"\nWrote: {csv_path}")
    print("\nInterpretation:")
    print("- If errors shrink as substeps increase, RHS/reconstruction are aligned but explicit RK4 is under-resolved.")
    print("- If errors do not shrink, inspect state ordering/current convention or MATLAB/Python parameter mismatch.")
    print("- A practical PINN implementation should use semi-implicit electrolyte stepping, not huge RK4 substeps.")


if __name__ == "__main__":
    main()
