"""
Diagnostic consistency tests for PINN-SPMe against the attached MATLAB SPMe implementation.

Usage:
  1) Copy this file into the root of your PINN-SPMe repository.
  2) Run: python test_spme_consistency.py
  3) Optional strict mode: python test_spme_consistency.py --strict

What it checks:
  - electrolyte FVM zero-current uniform concentration invariance
  - electrolyte lithium conservation of RHS
  - c_ex reconstruction length/indexing used by MATLAB SPMe
  - voltage decomposition: electrolyte concentration term is zero for uniform c_e
  - voltage implementation consistency with MATLAB-style exchange-current calculation

The last test is expected to fail/warn if calc_voltage() uses ce0n/ce0p for exchange current.
The MATLAB code uses region-average electrolyte concentration via ce_bar, so i0n uses cen_bar
and i0p uses cep_bar.
"""

import argparse
import os
import sys
import math
import traceback

import torch
import yaml

# Make imports work when the script is placed in repo root or run with cwd=repo root.
sys.path.insert(0, os.getcwd())

from init_params import InitParamsSPMe
from spme_model import SPMeModel, ref_potential_anode, ref_potential_cathode
from electrolyte_fvm import (
    electrolyte_rhs,
    electrolyte_lithium_inventory,
    electrolyte_boundary_concentrations,
    electrolyte_region_means,
    electrolyte_conductivity,
    electrolyte_activity,
)


def load_cfg():
    cfg_path = "config_spme_pinn.yaml"
    if not os.path.exists(cfg_path):
        raise FileNotFoundError("config_spme_pinn.yaml not found. Run this from the PINN-SPMe repo root.")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def matlab_style_voltage(model, css_n, css_p, c_e, current):
    """Reimplementation of nonlinearSPMeOutputVoltage.m using the repo's torch functions.

    Key MATLAB details mirrored here:
      - ce0n, cens, cesp, ce0p are reconstructed from c_ex.
      - cen_bar/ces_bar/cep_bar are used for electrolyte conductivity.
      - exchange current uses ce_bar: i0n(1) -> cen_bar, i0p(end) -> cep_bar.
      - V = V_noVCE + V_electrolyteCond + V_electrolytePolar.
    """
    p = model.p
    orig_shape = css_n.shape
    css_n = css_n.reshape(-1, 1).clamp(1.0, p.c_s_n_max * 0.999)
    css_p = css_p.reshape(-1, 1).clamp(1.0, p.c_s_p_max * 0.999)
    current = current.reshape(-1, 1)
    if c_e.dim() == 3:
        c_e = c_e.permute(0, 2, 1).reshape(-1, c_e.shape[1])

    ce0n, cens, cesp, ce0p, _ = electrolyte_boundary_concentrations(c_e, p)
    cen_bar, ces_bar, cep_bar = electrolyte_region_means(c_e, p)

    theta_n = css_n / p.c_s_n_max
    theta_p = css_p / p.c_s_p_max
    unref = ref_potential_anode(theta_n)
    upref = ref_potential_cathode(theta_p)

    arrh_kn = torch.exp(torch.as_tensor(p.E_kn / p.R * (1 / p.T_ref - 1 / p.T_amb), dtype=css_n.dtype, device=css_n.device))
    arrh_kp = torch.exp(torch.as_tensor(p.E_kp / p.R * (1 / p.T_ref - 1 / p.T_amb), dtype=css_n.dtype, device=css_n.device))

    # MATLAB nonlinearSPMeOutputVoltage.m uses ce_bar, so use region means here.
    i0n = p.k_n0 * arrh_kn * torch.clamp((p.c_s_n_max - css_n) * css_n * cen_bar.clamp_min(1.0), min=1e-12) ** p.alph
    i0p = p.k_p0 * arrh_kp * torch.clamp((p.c_s_p_max - css_p) * css_p * cep_bar.clamp_min(1.0), min=1e-12) ** p.alph

    rtaf = (p.R * p.T_amb) / (p.alph * p.Faraday)
    v_no_vce = (
        rtaf * torch.arcsinh(-current / (2 * p.a_s_p * p.Area_p * p.L_p * i0p))
        - rtaf * torch.arcsinh(current / (2 * p.a_s_n * p.Area_n * p.L_n * i0n))
        + upref - unref
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
    return (v_no_vce + v_cond + v_polar).reshape(orig_shape), v_no_vce.reshape(orig_shape), v_cond.reshape(orig_shape), v_polar.reshape(orig_shape)


def run_check(name, fn, strict=False):
    try:
        ok, message = fn()
    except Exception as exc:
        ok = False
        message = f"EXCEPTION: {exc}\n{traceback.format_exc()}"
    status = "PASS" if ok else ("FAIL" if strict else "WARN")
    print(f"[{status}] {name}: {message}")
    if strict and not ok:
        raise AssertionError(f"{name} failed: {message}")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="Exit nonzero on failed/warning checks.")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    cfg = load_cfg()
    cfg["integ_type"] = 1
    p = InitParamsSPMe(cfg)
    model = SPMeModel(p, cfg)

    def check_zero_current_uniform_rhs():
        c_e = torch.full((2, p.n_ce), p.c_e0)
        current = torch.zeros((2, 1))
        jn = current / (p.Faraday * p.a_s_n * p.Area_n * p.L_n)
        jp = -current / (p.Faraday * p.a_s_p * p.Area_p * p.L_p)
        rhs = electrolyte_rhs(c_e, jn, jp, p)
        err = rhs.abs().max().item()
        return err < 1e-12, f"max |dc_e/dt| = {err:.3e}"

    def check_electrolyte_inventory_rhs_conservation():
        # Nonuniform c_e and nonzero current. Source terms should balance globally for default Nxn=Nxp.
        base = torch.linspace(850.0, 1150.0, p.n_ce).reshape(1, -1)
        current = torch.tensor([[2.0]])  # A; MATLAB convention positive = discharge
        jn = current / (p.Faraday * p.a_s_n * p.Area_n * p.L_n)
        jp = -current / (p.Faraday * p.a_s_p * p.Area_p * p.L_p)
        rhs = electrolyte_rhs(base, jn, jp, p)
        c2 = base + 1e-6 * rhs
        dndt_fd = ((electrolyte_lithium_inventory(c2, p) - electrolyte_lithium_inventory(base, p)) / 1e-6).item()
        return abs(dndt_fd) < 1e-12, f"dN_e/dt approx = {dndt_fd:.3e} mol/s"

    def check_c_ex_layout():
        c_e = torch.arange(1, p.n_ce + 1, dtype=torch.float64).reshape(1, -1)
        ce0n, cens, cesp, ce0p, c_ex = electrolyte_boundary_concentrations(c_e, p)
        expected_len = p.Nx + 1
        ok_len = c_ex.shape[1] == expected_len
        ok_indices = (c_ex[:, 0:1].shape == ce0n.shape and c_ex[:, p.Nxn:p.Nxn+1].shape == cens.shape
                      and c_ex[:, p.Nxn+p.Nxs:p.Nxn+p.Nxs+1].shape == cesp.shape
                      and c_ex[:, -1:].shape == ce0p.shape)
        return ok_len and ok_indices, f"len(c_ex)={c_ex.shape[1]}, expected={expected_len}; MATLAB indices: 1, Nxn+1, Nxn+Nxs+1, end"

    def check_uniform_ce_concentration_polarization_zero():
        x = model.initial_state(batch_size=1, dtype=torch.float64)
        c_s_n, c_e, c_s_p = model.unpack_state(x)
        cs_n, cs_p = model.calc_css_and_cs0(x, torch.tensor([[2.0]], dtype=torch.float64))
        css_n, css_p = cs_n[:, -1], cs_p[:, -1]
        _, _, _, v_polar = matlab_style_voltage(model, css_n, css_p, c_e, torch.tensor([[2.0]], dtype=torch.float64))
        err = v_polar.abs().max().item()
        return err < 1e-12, f"|V_electrolytePolar|={err:.3e} V for uniform c_e"

    def check_voltage_matches_matlab_exchange_current_convention():
        x = model.initial_state(batch_size=1, dtype=torch.float64)
        c_s_n, c_e, c_s_p = model.unpack_state(x)
        # Create a gradient so ce0n != cen_bar and ce0p != cep_bar. This exposes exchange-current convention mismatch.
        c_e = torch.linspace(750.0, 1300.0, p.n_ce, dtype=torch.float64).reshape(1, -1)
        x = model.pack_state(c_s_n, c_e, c_s_p)
        current = torch.tensor([[3.0]], dtype=torch.float64)
        cs_n, cs_p = model.calc_css_and_cs0(x, current)
        css_n, css_p = cs_n[:, -1], cs_p[:, -1]
        v_repo = model.calc_voltage(css_n, css_p, c_e, current)
        v_matlab, _, _, _ = matlab_style_voltage(model, css_n, css_p, c_e, current)
        diff = (v_repo - v_matlab).abs().max().item()
        ok = diff < 1e-10
        msg = f"max |V_repo - V_MATLAB_style| = {diff:.3e} V. If nonzero, calc_voltage likely uses ce0n/ce0p for i0; MATLAB uses cen_bar/cep_bar."
        return ok, msg

    checks = [
        ("zero-current uniform electrolyte RHS", check_zero_current_uniform_rhs),
        ("electrolyte inventory RHS conservation", check_electrolyte_inventory_rhs_conservation),
        ("MATLAB c_ex layout/indexing", check_c_ex_layout),
        ("uniform electrolyte concentration polarization is zero", check_uniform_ce_concentration_polarization_zero),
        ("voltage matches MATLAB exchange-current convention", check_voltage_matches_matlab_exchange_current_convention),
    ]

    results = [run_check(name, fn, strict=args.strict) for name, fn in checks]
    print("\nSummary:", f"{sum(results)}/{len(results)} checks passed")
    if not all(results):
        print("\nRecommended fix if the voltage convention check warns/fails:")
        print("  In spme_model.py::calc_voltage(), change i_0n/i_0p to use cen_bar/cep_bar")
        print("  instead of ce0n/ce0p, matching nonlinearSPMeOutputVoltage.m and SPMe.m.")
        if args.strict:
            sys.exit(1)


if __name__ == "__main__":
    main()
