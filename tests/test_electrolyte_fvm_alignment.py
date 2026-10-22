"""
Diagnostic tests for Python vs MATLAB SPMe electrolyte FVM alignment.

Place this file at:
    PINN-SPMe/tests/test_electrolyte_fvm_alignment.py

Run:
    python tests/test_electrolyte_fvm_alignment.py

What this checks:
1) The Python electrolyte_rhs() matches MATLAB ode_electrolyte_conservative.m
   at the same c_e and same current-derived jn/jp. This isolates the FVM RHS
   from time-integration/IDAS effects.
2) The Python electrolyte_boundary_concentrations()/c_ex reconstruction matches
   MATLAB's SPMe f_cex/C_ce reconstruction. This tests the hypothesis that the
   weird electrolyte profile comes from reconstruction/output indexing rather than RHS.

The test writes CSV diagnostics to:
    results/electrolyte_fvm_alignment/
"""

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from init_params import InitParamsSPMe
from electrolyte_fvm import electrolyte_rhs, electrolyte_boundary_concentrations

LGM50_DIR = ROOT / "LGM50"
OUT_DIR = ROOT / "results" / "electrolyte_fvm_alignment"


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
    raise AssertionError(
        "MATLAB executable is required. Set MATLAB_EXECUTABLE or install MATLAB under /Applications."
    )


def _make_electrolyte_cases(p):
    """Return [n_cases, n_ce] c_e cases designed to stress reconstruction/interfaces."""
    n_ce = p.n_ce
    n_n = p.Nxn - 1
    n_s = p.Nxs - 1
    n_p = p.Nxp - 1

    cases = []
    names = []

    # 1. Uniform case: RHS should be purely source-driven; c_ex should be uniform.
    cases.append(np.full(n_ce, p.c_e, dtype=float))
    names.append("uniform")

    # 2. Smooth global gradient.
    cases.append(np.linspace(850.0, 1150.0, n_ce))
    names.append("global_linear")

    # 3. Piecewise means similar to discharge electrolyte polarization.
    cases.append(np.concatenate([
        np.linspace(1120.0, 1080.0, n_n),
        np.linspace(1010.0, 990.0, n_s),
        np.linspace(930.0, 880.0, n_p),
    ]))
    names.append("piecewise_gradient")

    # 4. Interface-stressed case to expose C_ce/reconstruction differences.
    c = np.full(n_ce, 1000.0)
    c[:n_n] = np.linspace(1160.0, 1060.0, n_n)
    c[n_n:n_n+n_s] = np.linspace(980.0, 1010.0, n_s)
    c[n_n+n_s:] = np.linspace(940.0, 1130.0, n_p)
    cases.append(c)
    names.append("interface_stress")

    return np.vstack(cases), names


def _run_matlab_fvm(p, cases, currents, tmp_path, matlab_exe):
    np.savetxt(tmp_path / "ce_cases.csv", cases, delimiter=",")
    np.savetxt(tmp_path / "currents.csv", np.asarray(currents).reshape(-1, 1), delimiter=",")

    runner_path = tmp_path / "run_electrolyte_fvm_alignment.m"
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

            ce_cases = readmatrix(fullfile('{tmp_path.as_posix()}', 'ce_cases.csv'));
            currents = readmatrix(fullfile('{tmp_path.as_posix()}', 'currents.csv'));
            currents = currents(:);

            p = load_SPMe_params(1.0, 25, {p.Nr}, {p.delta_t:.17g}, {p.Nxn}, {p.Nxs}, {p.Nxp}, ...
                true, 'none', 'fvm', false, false, false);
            p.csn0 = {p.csn0:.17g};
            p.csp0 = {p.csp0:.17g};

            [~,~,~,~,~, ~,~,~,~, ~,~,~,~,~, C_ce] = c_e_mats(p);
            model = SPMe(p, 'idas');

            n_cases = size(ce_cases, 1);
            n_curr = numel(currents);
            n_ce = size(ce_cases, 2);
            n_cex = p.Nx + 1;

            rhs = zeros(n_cases * n_curr, n_ce);
            cex_from_Cce = zeros(n_cases, n_cex);
            cex_from_model = zeros(n_cases, n_cex);
            bcs = zeros(n_cases, 4);
            meta = zeros(n_cases * n_curr, 3); % case_idx, current_idx, current

            for icase = 1:n_cases
                ce = ce_cases(icase, :).';
                ce_bcs = C_ce * ce;
                c_en = ce(1:(p.Nxn-1));
                c_es = ce((p.Nxn-1)+1:(p.Nxn-1)+(p.Nxs-1));
                c_ep = ce((p.Nxn-1)+p.Nxs:end);
                c_ex = [ce_bcs(1); c_en; ce_bcs(2); c_es; ce_bcs(3); c_ep; ce_bcs(4)];
                cex_from_Cce(icase, :) = full(c_ex).';
                bcs(icase, :) = full(ce_bcs).';

                % Verify reconstruction through SPMe.f_cex too.
                x = zeros(2*(p.Nr-1) + n_ce, 1);
                x((2*(p.Nr-1)+1):end) = ce;
                F_cex = model.f_cex('x', x);
                cex_from_model(icase, :) = full(F_cex.c_ex).';

                for j = 1:n_curr
                    I = currents(j);
                    jn = I / (p.Faraday * p.a_s_n * p.Area_n * p.L_n);
                    jp = -I / (p.Faraday * p.a_s_p * p.Area_p * p.L_p);
                    [ce_dot, aux] = ode_electrolyte_conservative(ce, jn, jp, p);
                    row = (icase - 1) * n_curr + j;
                    rhs(row, :) = full(ce_dot).';
                    meta(row, :) = [icase, j, I];
                end
            end

            writematrix(rhs, fullfile('{tmp_path.as_posix()}', 'rhs_matlab.csv'));
            writematrix(cex_from_Cce, fullfile('{tmp_path.as_posix()}', 'cex_matlab_Cce.csv'));
            writematrix(cex_from_model, fullfile('{tmp_path.as_posix()}', 'cex_matlab_model.csv'));
            writematrix(bcs, fullfile('{tmp_path.as_posix()}', 'bcs_matlab.csv'));
            writematrix(meta, fullfile('{tmp_path.as_posix()}', 'meta.csv'));
            """
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [matlab_exe, "-nodisplay", "-nosplash", "-nodesktop", "-batch", f"run('{runner_path.as_posix()}')"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=1200,
    )
    if result.returncode != 0:
        raise AssertionError(
            "MATLAB electrolyte FVM diagnostic failed.\n"
            f"Return code: {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    return {
        "rhs": np.atleast_2d(np.loadtxt(tmp_path / "rhs_matlab.csv", delimiter=",")),
        "cex_Cce": np.atleast_2d(np.loadtxt(tmp_path / "cex_matlab_Cce.csv", delimiter=",")),
        "cex_model": np.atleast_2d(np.loadtxt(tmp_path / "cex_matlab_model.csv", delimiter=",")),
        "bcs": np.atleast_2d(np.loadtxt(tmp_path / "bcs_matlab.csv", delimiter=",")),
        "meta": np.atleast_2d(np.loadtxt(tmp_path / "meta.csv", delimiter=",")),
    }


def _python_rhs_and_cex(p, cases, currents):
    rhs_rows = []
    cex_rows = []
    bcs_rows = []
    for ce_np in cases:
        ce = torch.tensor(ce_np.reshape(1, -1), dtype=torch.float64)
        ce0n, cens, cesp, ce0p, cex = electrolyte_boundary_concentrations(ce, p)
        cex_rows.append(cex.detach().cpu().numpy().reshape(-1))
        bcs_rows.append(np.array([ce0n.item(), cens.item(), cesp.item(), ce0p.item()]))
        for I in currents:
            jn = torch.tensor([[I / (p.Faraday * p.a_s_n * p.Area_n * p.L_n)]], dtype=torch.float64)
            jp = torch.tensor([[-I / (p.Faraday * p.a_s_p * p.Area_p * p.L_p)]], dtype=torch.float64)
            rhs = electrolyte_rhs(ce, jn, jp, p)
            rhs_rows.append(rhs.detach().cpu().numpy().reshape(-1))
    return np.vstack(rhs_rows), np.vstack(cex_rows), np.vstack(bcs_rows)


class TestElectrolyteFVMAlignment(unittest.TestCase):
    def setUp(self):
        if not LGM50_DIR.exists():
            raise AssertionError(f"LGM50 MATLAB folder not found: {LGM50_DIR}")
        self.matlab = _find_matlab_executable()

    def test_electrolyte_fvm_rhs_and_reconstruction_against_matlab(self):
        torch.set_default_dtype(torch.float64)
        cfg = _base_cfg()
        p = InitParamsSPMe(cfg)
        cases, case_names = _make_electrolyte_cases(p)
        currents = np.array([0.0, p.OneC, -p.OneC, 2.0 * p.OneC], dtype=float)

        rhs_py, cex_py, bcs_py = _python_rhs_and_cex(p, cases, currents)

        with tempfile.TemporaryDirectory(prefix="electrolyte_fvm_alignment_") as tmp:
            tmp_path = Path(tmp)
            matlab = _run_matlab_fvm(p, cases, currents, tmp_path, self.matlab)

        rhs_mat = matlab["rhs"]
        cex_mat_Cce = matlab["cex_Cce"]
        cex_mat_model = matlab["cex_model"]
        bcs_mat = matlab["bcs"]

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        np.savetxt(OUT_DIR / "ce_cases.csv", cases, delimiter=",", header=",".join(case_names), comments="")
        np.savetxt(OUT_DIR / "rhs_python.csv", rhs_py, delimiter=",")
        np.savetxt(OUT_DIR / "rhs_matlab.csv", rhs_mat, delimiter=",")
        np.savetxt(OUT_DIR / "rhs_diff_python_minus_matlab.csv", rhs_py - rhs_mat, delimiter=",")
        np.savetxt(OUT_DIR / "cex_python.csv", cex_py, delimiter=",")
        np.savetxt(OUT_DIR / "cex_matlab_Cce.csv", cex_mat_Cce, delimiter=",")
        np.savetxt(OUT_DIR / "cex_matlab_model.csv", cex_mat_model, delimiter=",")
        np.savetxt(OUT_DIR / "cex_diff_python_minus_matlab_Cce.csv", cex_py - cex_mat_Cce, delimiter=",")
        np.savetxt(OUT_DIR / "bcs_python.csv", bcs_py, delimiter=",", header="ce0n,cens,cesp,ce0p", comments="")
        np.savetxt(OUT_DIR / "bcs_matlab.csv", bcs_mat, delimiter=",", header="ce0n,cens,cesp,ce0p", comments="")

        print("\nElectrolyte FVM alignment diagnostics")
        print(f"  Output dir: {OUT_DIR}")
        print(f"  max |RHS Python - MATLAB|: {np.max(np.abs(rhs_py - rhs_mat)):.6e}")
        print(f"  max |c_ex MATLAB C_ce - MATLAB model.f_cex|: {np.max(np.abs(cex_mat_Cce - cex_mat_model)):.6e}")
        print(f"  max |c_ex Python - MATLAB C_ce|: {np.max(np.abs(cex_py - cex_mat_Cce)):.6e}")
        print(f"  max |BC Python - MATLAB C_ce|: {np.max(np.abs(bcs_py - bcs_mat)):.6e}")

        # 1) This should pass if Python electrolyte_rhs is the same as MATLAB ode_electrolyte_conservative.
        np.testing.assert_allclose(
            rhs_py,
            rhs_mat,
            atol=1e-8,
            rtol=1e-8,
            err_msg=(
                "Python electrolyte_rhs() does not match MATLAB ode_electrolyte_conservative.m. "
                "If this fails, debug FVM flux/source/sign/scaling before testing voltage."
            ),
        )

        # MATLAB's manual C_ce reconstruction and SPMe.f_cex should be identical.
        np.testing.assert_allclose(
            cex_mat_Cce,
            cex_mat_model,
            atol=1e-12,
            rtol=1e-12,
            err_msg="MATLAB C_ce reconstruction does not match model.f_cex; check MATLAB test script.",
        )

        # 2) This tests the reconstruction hypothesis. If this fails while RHS passes,
        #    the electrolyte dynamics are aligned but Python c_ex/boundary reconstruction is not.
        np.testing.assert_allclose(
            cex_py,
            cex_mat_Cce,
            atol=1e-10,
            rtol=1e-10,
            err_msg=(
                "Python electrolyte c_ex reconstruction does not match MATLAB C_ce reconstruction. "
                "This supports the hypothesis that the weird electrolyte curve comes from reconstruction/output, "
                "not from the FVM RHS. Port MATLAB C_ce reconstruction into Python."
            ),
        )


if __name__ == "__main__":
    unittest.main()
