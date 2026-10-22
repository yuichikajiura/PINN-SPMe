import copy
import sys
import unittest
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from init_params import InitParamsSPMe
from pinn_spme_network import create_spme_nn_models, predict_spme_state_raw
from spme_model import SPMeModel
from train_pinn_spme import _load_pretrained_nn_models, _load_training_dataset, _prepare_dataset, _prepare_state_dataset


class TestSPMeWarmStartConsistency(unittest.TestCase):
    def test_pretrained_nn_matches_pinn_bare_warm_start_path(self):
        cfg_path = ROOT / "config_spme_pinn.yaml"
        if not cfg_path.exists():
            self.skipTest("config_spme_pinn.yaml is not available.")

        with cfg_path.open() as f:
            cfg = yaml.safe_load(f)
        cfg = copy.deepcopy(cfg)
        checkpoint_path = cfg.get("pretrained_nn_checkpoint")
        if not checkpoint_path:
            self.skipTest("pretrained_nn_checkpoint is not configured.")
        checkpoint_path = ROOT / checkpoint_path
        if not checkpoint_path.exists():
            self.skipTest(f"pretrained NN checkpoint is not available: {checkpoint_path}")

        cfg.update({
            "use_wandb": False,
            "max_val_windows": None,
            "val_window_stride": max(1, int(cfg["k"])),
            "pretrained_nn_checkpoint": str(checkpoint_path),
        })

        device = torch.device("cpu")
        p = InitParamsSPMe(cfg)
        model = SPMeModel(p, cfg).to(device)

        raw_dataset = _load_training_dataset(cfg, device, p)
        raw_dataset["val"]["split"] = "val"
        nn_val = _prepare_state_dataset(raw_dataset["val"], cfg, p, device)
        pinn_val = _prepare_dataset(raw_dataset["val"], cfg, p, device)

        n_samples = min(nn_val["u"].shape[0], pinn_val["u"].shape[0], 8)
        self.assertGreater(n_samples, 0)
        self.assertTrue(torch.allclose(nn_val["u"][:n_samples], pinn_val["u"][:n_samples], atol=0.0, rtol=0.0))
        self.assertTrue(
            torch.equal(
                torch.as_tensor(nn_val["metadata"]["window_start_idx"][:n_samples]),
                torch.as_tensor(pinn_val["metadata"]["window_start_idx"][:n_samples]),
            )
        )
        pinn_bare_target_idx = torch.as_tensor(pinn_val["metadata"]["window_start_idx"][:n_samples]) + int(cfg["k"]) - 1
        self.assertTrue(torch.equal(torch.as_tensor(nn_val["metadata"]["target_idx"][:n_samples]), pinn_bare_target_idx))

        nn_models_for_nn = create_spme_nn_models(cfg, p).to(device)
        nn_models_for_pinn = create_spme_nn_models(cfg, p).to(device)
        _load_pretrained_nn_models(cfg, nn_models_for_nn, device)
        _load_pretrained_nn_models(cfg, nn_models_for_pinn, device)

        with torch.no_grad():
            x_raw_nn = predict_spme_state_raw(nn_models_for_nn, nn_val["u"][:n_samples])
            x_raw_pinn = predict_spme_state_raw(nn_models_for_pinn, pinn_val["u"][:n_samples])
            self.assertLess(float((x_raw_nn - x_raw_pinn).abs().max()), 1e-7)

            current_nn = nn_val["i_target"][:n_samples]
            current_pinn = pinn_val["i_future_sampled"][:n_samples, 0:1]
            self.assertTrue(torch.allclose(current_nn, current_pinn, atol=0.0, rtol=0.0))

            c_n_nn, c_p_nn = model.calc_css_and_cs0(x_raw_nn, current_nn)
            c_n_pinn, c_p_pinn = model.calc_css_and_cs0(x_raw_pinn, current_pinn)
            _, c_e_nn, _ = model.unpack_state(x_raw_nn)
            _, c_e_pinn, _ = model.unpack_state(x_raw_pinn)

            theta_n_nn = c_n_nn[:, -1] / p.c_s_n_max
            theta_p_nn = c_p_nn[:, -1] / p.c_s_p_max
            theta_n_pinn = c_n_pinn[:, -1] / p.c_s_n_max
            theta_p_pinn = c_p_pinn[:, -1] / p.c_s_p_max
            self.assertLess(float((theta_n_nn - theta_n_pinn).abs().max()), 1e-12)
            self.assertLess(float((theta_p_nn - theta_p_pinn).abs().max()), 1e-12)

            components_nn = model.calc_voltage_components(c_n_nn[:, -1], c_p_nn[:, -1], c_e_nn, current_nn)
            components_pinn = model.calc_voltage_components(c_n_pinn[:, -1], c_p_pinn[:, -1], c_e_pinn, current_pinn)

            for key in [
                "voltage",
                "U_p",
                "U_n",
                "eta_p",
                "eta_n",
                "electrolyte_ohmic",
                "electrolyte_concentration",
                "film_drop",
                "theta_p",
                "theta_n",
            ]:
                self.assertLess(
                    float((components_nn[key] - components_pinn[key]).abs().max()),
                    1e-10,
                    msg=f"component mismatch: {key}",
                )


if __name__ == "__main__":
    unittest.main()
