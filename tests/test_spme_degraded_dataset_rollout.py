import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from init_params import InitParamsSPMe
from spme_model import SPMeModel
from train_pinn_spme import _spme_state_columns


def _true_degraded_cfg():
    with open(ROOT / "config_spme_pinn.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.update({
        "estimate_delta_sei": True,
        "delta_sei_init": cfg["delta_sei_true"],
        "epsilon_e_n_fresh": 0.25,
        "h": 1.0,
        "integ_type": 0,
        "electrolyte_integrator": "implicit",
        "electrolyte_substeps": 1,
    })
    return cfg


def _soh90_files(cfg):
    data_dir = ROOT / cfg.get("data_dir", "data")
    names = [
        *cfg.get("train_data_files", []),
        *cfg.get("val_data_files", []),
        *cfg.get("test_data_files", []),
    ]
    paths = [data_dir / name for name in names if name.endswith("_soh90.csv")]
    if not paths:
        paths = sorted(data_dir.glob("*_soh90.csv"))
    return paths


def _load_profile(path, p):
    df = pd.read_csv(path)
    c_s_n_cols, c_e_cols, c_s_p_cols = _spme_state_columns(p)
    state_cols = c_s_n_cols + c_e_cols + c_s_p_cols
    missing = [col for col in ["current_A", "voltage_V", "soc", *state_cols] if col not in df.columns]
    if missing:
        raise AssertionError(f"{path} is missing required columns: {missing}")
    x0 = torch.tensor(df.loc[0, state_cols].to_numpy(dtype=np.float64), dtype=torch.float64).view(1, -1)
    states = torch.tensor(df[state_cols].to_numpy(dtype=np.float64), dtype=torch.float64)
    current = torch.tensor(df["current_A"].to_numpy(dtype=np.float64), dtype=torch.float64).view(-1, 1)
    voltage = df["voltage_V"].to_numpy(dtype=np.float64)
    return df, x0, states, current, voltage


def _voltage_from_states(model, states, current, batch_size=4096):
    voltage = []
    with torch.no_grad():
        for start in range(0, states.shape[0], batch_size):
            x = states[start:start + batch_size]
            i = current[start:start + batch_size]
            c_n, c_p = model.calc_css_and_cs0(x, i)
            _, c_e, _ = model.unpack_state(x)
            voltage.append(model.calc_voltage(c_n[:, -1], c_p[:, -1], c_e, i).reshape(-1).cpu().numpy())
    return np.concatenate(voltage)


def _roll_from_initial_state(model, x0, current):
    x = x0.clone()
    voltage = []
    states = []
    with torch.no_grad():
        for k in range(current.shape[0]):
            states.append(x.detach().clone())
            i = current[k:k + 1]
            c_n, c_p = model.calc_css_and_cs0(x, i)
            _, c_e, _ = model.unpack_state(x)
            voltage.append(float(model.calc_voltage(c_n[:, -1], c_p[:, -1], c_e, i).reshape(())))
            if k < current.shape[0] - 1:
                x = model.step(x, i)
    return np.asarray(voltage, dtype=np.float64), torch.cat(states, dim=0)


def _error_summary(pred, true):
    err = np.asarray(pred) - np.asarray(true)
    return {
        "max_abs": float(np.max(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mean": float(np.mean(err)),
        "idx_max": int(np.argmax(np.abs(err))),
    }


def _shifted_voltage_summary(pred, true):
    return {
        "same_time": _error_summary(pred, true),
        "pred_k_true_k_plus_1": _error_summary(pred[:-1], true[1:]) if len(pred) > 1 else None,
    }


def _state_error_summary(model, pred_states, true_states):
    pred_n, pred_e, pred_p = model.unpack_state(pred_states)
    true_n, true_e, true_p = model.unpack_state(true_states)

    def rmse(a, b):
        return float(torch.sqrt(torch.mean((a - b) ** 2)).detach().cpu())

    return {
        "c_s_n_rmse": rmse(pred_n, true_n),
        "c_e_rmse": rmse(pred_e, true_e),
        "c_s_p_rmse": rmse(pred_p, true_p),
    }


@unittest.skipUnless(
    os.environ.get("RUN_SPME_DEGRADED_DATASET_ROLLOUT") == "1",
    "Set RUN_SPME_DEGRADED_DATASET_ROLLOUT=1 to run this expensive degraded-dataset diagnostic.",
)
class TestSPMeDegradedDatasetRollout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_default_dtype(torch.float64)
        cls.cfg = _true_degraded_cfg()
        cls.p = InitParamsSPMe(cls.cfg)
        cls.model = SPMeModel(cls.p, cls.cfg).double()
        cls.paths = _soh90_files(cls.cfg)
        if not cls.paths:
            raise AssertionError("No *_soh90.csv datasets found.")

    def test_true_degraded_parameter_sanity(self):
        delta_tol = 1e-15
        epsilon_tol = 5e-7
        delta_est = float(self.model.delta_sei_est.detach().cpu())
        epsilon_est = float(self.model.epsilon_e_n_est.detach().cpu())
        nli_sei_est = float(self.model.nLi_sei_est.detach().cpu())
        nli_s_est = float(self.model.nLi_s_est.detach().cpu())
        r_f_n_est = float(self.model.R_f_n_est.detach().cpu())

        print(
            "true degraded parameter sanity: "
            f"delta_sei_est={delta_est:.12g}, "
            f"epsilon_e_n_est={epsilon_est:.12g}, "
            f"nLi_sei_est={nli_sei_est:.12g}, "
            f"nLi_s_est={nli_s_est:.12g}, "
            f"R_f_n_est={r_f_n_est:.12g}"
        )
        self.assertAlmostEqual(delta_est, float(self.cfg["delta_sei_true"]), delta=delta_tol)
        self.assertAlmostEqual(epsilon_est, float(self.cfg["epsilon_e_n_true"]), delta=epsilon_tol)
        self.assertGreater(nli_sei_est, 0.0)
        self.assertLess(nli_s_est, float(self.cfg["nLi_s_fresh"]))
        self.assertGreaterEqual(r_f_n_est, 0.0)

    def test_recorded_states_and_true_degraded_parameters_reproduce_voltage(self):
        """Voltage model sanity check using recorded dataset states at each time."""
        max_allowed = float(self.cfg.get("degraded_direct_voltage_max_abs_tol", 5e-3))
        rmse_allowed = float(self.cfg.get("degraded_direct_voltage_rmse_tol", 1e-3))
        failures = []
        for path in self.paths:
            df, _, states, current, voltage_true = _load_profile(path, self.p)
            voltage_pred = _voltage_from_states(self.model, states, current)
            summary = _error_summary(voltage_pred, voltage_true)
            if summary["max_abs"] > max_allowed or summary["rmse"] > rmse_allowed:
                idx = summary["idx_max"]
                failures.append(
                    f"{path.name}: max_abs={summary['max_abs']:.6g} V, "
                    f"rmse={summary['rmse']:.6g} V, mean={summary['mean']:.6g} V, "
                    f"idx_max={idx}, t={df['time_s'].iloc[idx]}, I={df['current_A'].iloc[idx]}, "
                    f"soc={df['soc'].iloc[idx]}, V_true={voltage_true[idx]:.9g}, "
                    f"V_pred={voltage_pred[idx]:.9g}"
                )
        self.assertFalse(
            failures,
            "Recorded states do not reproduce recorded voltage with true degraded parameters.\n"
            + "\n".join(failures),
        )

    def test_rollout_from_actual_initial_state_and_true_degraded_parameters_reproduces_voltage(self):
        """PINN-rollout sanity check from the actual initial state/SOC in each SOH90 dataset."""
        max_profiles = int(os.environ.get("SPME_DEGRADED_ROLLOUT_MAX_PROFILES", "0"))
        paths = self.paths[:max_profiles] if max_profiles > 0 else self.paths
        max_allowed = float(self.cfg.get("degraded_rollout_voltage_max_abs_tol", 2e-2))
        rmse_allowed = float(self.cfg.get("degraded_rollout_voltage_rmse_tol", 5e-3))
        failures = []
        for path in paths:
            df, x0, states_true, current, voltage_true = _load_profile(path, self.p)
            voltage_pred, states_pred = _roll_from_initial_state(self.model, x0, current)
            shifted_summary = _shifted_voltage_summary(voltage_pred, voltage_true)
            summary = shifted_summary["same_time"]
            state_summary = _state_error_summary(self.model, states_pred, states_true)
            shifted = shifted_summary["pred_k_true_k_plus_1"]
            shifted_text = (
                f", shifted_rmse_pred_k_true_k_plus_1={shifted['rmse']:.6g} V, "
                f"shifted_max_abs={shifted['max_abs']:.6g} V"
                if shifted is not None else ""
            )
            if summary["max_abs"] > max_allowed or summary["rmse"] > rmse_allowed:
                idx = summary["idx_max"]
                failures.append(
                    f"{path.name}: max_abs={summary['max_abs']:.6g} V, "
                    f"rmse={summary['rmse']:.6g} V, mean={summary['mean']:.6g} V"
                    f"{shifted_text}, "
                    f"c_s_n_rmse={state_summary['c_s_n_rmse']:.6g}, "
                    f"c_e_rmse={state_summary['c_e_rmse']:.6g}, "
                    f"c_s_p_rmse={state_summary['c_s_p_rmse']:.6g}, "
                    f"idx_max={idx}, t={df['time_s'].iloc[idx]}, I={df['current_A'].iloc[idx]}, "
                    f"soc0={df['soc'].iloc[0]}, soc={df['soc'].iloc[idx]}, "
                    f"V_true={voltage_true[idx]:.9g}, V_pred={voltage_pred[idx]:.9g}"
                )
        self.assertFalse(
            failures,
            "SPMe rollout from actual initial SOH/SOC does not reproduce recorded voltage.\n"
            + "\n".join(failures),
        )


if __name__ == "__main__":
    unittest.main()
