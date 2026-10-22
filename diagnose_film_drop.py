#!/usr/bin/env python3
"""Diagnose SEI film-drop calculation for PINN-SPMe checkpoints.

The terminal-voltage film contribution in the Python SPMe code is intended to
match ``LGM50/SPMe/SPMe.m``:

    R_tot_n = R_f_n_base + delta_sei / kappa_sei
    film_drop = (R_tot_n/(a_s_n*L_n*Area_n)
                 + R_f_p/(a_s_p*L_p*Area_p)) * I

Terminal voltage uses ``-film_drop``.
"""

import argparse
import math
import re
from pathlib import Path

import torch

import init_params as ip
from spme_model import SPMeModel


def _load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _film_drop(p, delta_sei, currents):
    r_tot_n = p.R_f_n_base + float(delta_sei) / p.kappa_sei
    coeff = (
        r_tot_n / (p.a_s_n * p.L_n * p.Area_n)
        + p.R_f_p / (p.a_s_p * p.L_p * p.Area_p)
    )
    return r_tot_n, coeff * currents


def _parse_matlab_params(path):
    """Parse scalar film-drop parameters from LGM50/param/params_LGM50.m."""
    text = Path(path).read_text()
    params = {}
    for name in [
        "L_n",
        "L_p",
        "R_s_n",
        "R_s_p",
        "epsilon_s_n",
        "epsilon_s_p",
        "Area_n",
        "Area_p",
        "R_f_n",
        "R_f_p",
        "kappa_sei",
        "kappa_P",
    ]:
        match = re.search(
            rf"p\.{name}\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][-+]?\d+)?)\s*;",
            text,
        )
        if match:
            params[name] = float(match.group(1).replace("D", "e").replace("d", "e"))
    if {"epsilon_s_n", "R_s_n"} <= set(params):
        params["a_s_n"] = 3.0 * params["epsilon_s_n"] / params["R_s_n"]
    if {"epsilon_s_p", "R_s_p"} <= set(params):
        params["a_s_p"] = 3.0 * params["epsilon_s_p"] / params["R_s_p"]
    return params


def _compare_python_matlab_params(p, matlab_path):
    matlab = _parse_matlab_params(matlab_path)
    python = {
        "L_n": p.L_n,
        "L_p": p.L_p,
        "R_s_n": p.R_s_n,
        "R_s_p": p.R_s_p,
        "epsilon_s_n": p.epsilon_s_n,
        "epsilon_s_p": p.epsilon_s_p,
        "a_s_n": p.a_s_n,
        "a_s_p": p.a_s_p,
        "Area_n": p.Area_n,
        "Area_p": p.Area_p,
        "R_f_n": p.R_f_n_base,
        "R_f_p": p.R_f_p,
        "kappa_sei": p.kappa_sei,
    }
    print("Python vs MATLAB film-drop parameters")
    print(f"MATLAB source: {matlab_path}")
    print("name, python, matlab, abs_diff, rel_diff")
    ok = True
    for name, py_value in python.items():
        mat_value = matlab.get(name)
        if mat_value is None:
            ok = False
            print(f"{name}, {py_value:.12g}, MISSING, nan, nan")
            continue
        diff = float(py_value) - float(mat_value)
        denom = max(abs(float(py_value)), abs(float(mat_value)), 1.0)
        rel = abs(diff) / denom
        if not math.isclose(float(py_value), float(mat_value), rel_tol=1e-12, abs_tol=1e-14):
            ok = False
        print(f"{name}, {float(py_value):.12g}, {float(mat_value):.12g}, {diff:.4e}, {rel:.4e}")

    if "kappa_P" in matlab:
        print()
        print(
            "note: MATLAB also defines kappa_P="
            f"{matlab['kappa_P']:.12g}; main LGM50/SPMe/SPMe.m terminal voltage uses kappa_sei, "
            "while nonlinearSPMeOutputVoltage.m uses kappa_P."
        )
    print(f"parameter_match: {ok}")
    print()
    return matlab, ok


def _matlab_formula_film_drop(matlab, delta_sei, currents, use_kappa="kappa_sei"):
    r_tot_n = matlab["R_f_n"] + float(delta_sei) / matlab[use_kappa]
    coeff = (
        r_tot_n / (matlab["a_s_n"] * matlab["L_n"] * matlab["Area_n"])
        + matlab["R_f_p"] / (matlab["a_s_p"] * matlab["L_p"] * matlab["Area_p"])
    )
    return r_tot_n, coeff * currents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="results/spme_pinn_from_nn_2000/checkpoint_latest.pth",
        help="Checkpoint containing config and model_state.",
    )
    parser.add_argument(
        "--currents",
        type=float,
        nargs="+",
        default=[-20.0, -5.0, 0.0, 5.0, 20.0],
        help="Currents [A] at which to compare film drop.",
    )
    parser.add_argument(
        "--matlab-params",
        default="LGM50/param/params_LGM50.m",
        help="MATLAB parameter file to parse for film-drop parameter comparison.",
    )
    args = parser.parse_args()

    checkpoint = _load_checkpoint(args.checkpoint)
    cfg = checkpoint["config"]
    p = ip.InitParamsSPMe(cfg)
    model = SPMeModel(p, cfg)
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"], strict=False)

    currents = torch.tensor(args.currents, dtype=torch.float64)
    delta_true = float(p.delta_sei_true)
    delta_est = float(model.delta_sei_est.detach().cpu().reshape(()))
    r_true, film_true = _film_drop(p, delta_true, currents)
    r_est, film_est = _film_drop(p, delta_est, currents)
    matlab, params_match = _compare_python_matlab_params(p, args.matlab_params)
    mat_r_true, mat_film_true = _matlab_formula_film_drop(matlab, delta_true, currents, "kappa_sei")
    mat_r_est, mat_film_est = _matlab_formula_film_drop(matlab, delta_est, currents, "kappa_sei")

    print(f"checkpoint: {Path(args.checkpoint)}")
    print("formula: film_drop = (R_tot_n/(a_s_n*L_n*Area_n) + R_f_p/(a_s_p*L_p*Area_p)) * I")
    print("voltage contribution: -film_drop")
    print()
    print(f"a_s_n={p.a_s_n:.12g}")
    print(f"L_n={p.L_n:.12g}")
    print(f"Area_n={p.Area_n:.12g}")
    print(f"a_s_n*L_n*Area_n={p.a_s_n * p.L_n * p.Area_n:.12g}")
    print(f"R_f_n_base={p.R_f_n_base:.12g}")
    print(f"R_f_p={p.R_f_p:.12g}")
    print(f"kappa_sei={p.kappa_sei:.12g}")
    print()
    print(f"delta_sei_true={delta_true:.12g} m")
    print(f"delta_sei_est ={delta_est:.12g} m")
    print(f"R_tot_n_true={r_true:.12g} Ohm m^2")
    print(f"R_tot_n_est ={r_est:.12g} Ohm m^2")
    print(f"MATLAB-formula R_tot_n_true={mat_r_true:.12g} Ohm m^2")
    print(f"MATLAB-formula R_tot_n_est ={mat_r_est:.12g} Ohm m^2")
    print()
    print(
        "I[A], py_film_true[V], matlab_film_true[V], py_minus_matlab_true[V], "
        "py_film_est[V], matlab_film_est[V], py_minus_matlab_est[V], "
        "film_error[V], voltage_error_contribution[V]"
    )
    formula_match = True
    for current, true_value, mat_true, est_value, mat_est in zip(currents, film_true, mat_film_true, film_est, mat_film_est):
        film_error = est_value - true_value
        true_diff = true_value - mat_true
        est_diff = est_value - mat_est
        formula_match = formula_match and math.isclose(float(true_diff), 0.0, rel_tol=1e-12, abs_tol=1e-12)
        formula_match = formula_match and math.isclose(float(est_diff), 0.0, rel_tol=1e-12, abs_tol=1e-12)
        print(
            f"{float(current): .8g}, "
            f"{float(true_value): .12g}, "
            f"{float(mat_true): .12g}, "
            f"{float(true_diff): .4e}, "
            f"{float(est_value): .12g}, "
            f"{float(mat_est): .12g}, "
            f"{float(est_diff): .4e}, "
            f"{float(film_error): .12g}, "
            f"{float(-film_error): .12g}"
        )
    print()
    print(f"formula_match_using_kappa_sei: {formula_match}")
    print(f"overall_match: {params_match and formula_match}")


if __name__ == "__main__":
    main()
