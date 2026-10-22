import unittest

import torch

from electrolyte_fvm import (
    build_electrolyte_linear_operator,
    build_electrolyte_linear_operator_debug,
    electrolyte_rhs,
)
from init_params import InitParamsSPMe
from spme_model import SPMeModel
from train_pinn_spme import (
    _bare_nn_lithium_inventory_losses,
    _integration_consistency_loss_from_predictions,
    _true_degradation_for_plot,
)


class TestSPMe(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            'n_r': 15,
            'n_x_n': 10,
            'n_x_s': 5,
            'n_x_p': 10,
            'h': 1,
            'integ_type': 1,
            'ce_min': 200.0,
            'ce_max': 2000.0,
            'temperature_K': 298.15,
            'use_electrolyte_activity': False,
            'electrolyte_integrator': 'implicit',
            'electrolyte_substeps': 1,
            'estimate_delta_sei': False,
            'nLi_s_fresh': 0.2840,
            'nLi_s_init': 0.2840,
            'nLi_e_init': 0.0053677182,
            'delta_sei_init': 0.0,
            'delta_sei_lower': 0.0,
            'delta_sei_upper': 2.0e-6,
            'delta_sei_true': 0.0,
            'epsilon_e_n_true': 0.25,
            'OneC': 5.0,
        }
        self.p = InitParamsSPMe(self.cfg)
        self.model = SPMeModel(self.p, self.cfg)
        self.x0 = self.model.initial_state(batch_size=2, dtype=torch.float64)

    def test_pack_unpack_identity(self):
        c_s_n, c_e, c_s_p = self.model.unpack_state(self.x0)
        x = self.model.pack_state(c_s_n, c_e, c_s_p)
        self.assertTrue(torch.allclose(self.x0, x))

    def test_zero_current_keeps_electrolyte_uniform(self):
        x = self.x0.clone()
        current = torch.zeros(2, 1)
        for _ in range(20):
            x = self.model.step(x, current)
        _, c_e, _ = self.model.unpack_state(x)
        self.assertTrue(torch.allclose(c_e, torch.full_like(c_e, self.p.c_e0), atol=1e-6, rtol=1e-6))

    def test_electrolyte_lithium_conserved(self):
        x = self.x0[:1].clone()
        current_seq = torch.tensor([[3.0, -2.0, 1.5, -1.0, 0.5, -0.25, 0.0, 2.0]])
        x_roll = torch.cat([x.unsqueeze(-1), self.model(x, current_seq)], dim=-1)
        _, c_e_seq, _ = self.model.unpack_state(x_roll)
        inventories = []
        from electrolyte_fvm import electrolyte_lithium_inventory
        for t in range(c_e_seq.shape[-1]):
            inventories.append(electrolyte_lithium_inventory(c_e_seq[:, :, t], self.p))
        inventories = torch.stack(inventories, dim=1)
        self.assertTrue(torch.allclose(inventories, inventories[:, :1], atol=1e-8, rtol=1e-6))

    def test_solid_lithium_conserved(self):
        x = self.x0[:1].clone()
        current_seq = torch.tensor([[3.0, -2.0, 1.5, -1.0, 0.5, -0.25, 0.0, 2.0]])
        x_roll = torch.cat([x.unsqueeze(-1), self.model(x, current_seq)], dim=-1)
        c_n_seq, c_p_seq = self.model.calc_css_and_cs0_seq(x_roll, torch.cat([torch.zeros(1, 1), current_seq], dim=1))
        inventories = self.model.calc_solid_lithium(c_n_seq, c_p_seq)
        self.assertTrue(torch.allclose(inventories, inventories[:, :1], atol=3e-4, rtol=1e-3))

    def test_calc_cs_bar_vectorized_matches_shell_loop(self):
        generator = torch.Generator().manual_seed(23)
        cs_2d = torch.rand(4, self.model.n_r + 1, generator=generator, dtype=torch.float64)
        cs_3d = torch.rand(4, self.model.n_r + 1, 6, generator=generator, dtype=torch.float64)

        def old_loop(cs):
            result = torch.zeros(cs.shape[0], *cs.shape[2:], dtype=cs.dtype, device=cs.device)
            for idx in range(self.model.n_r):
                r_u = (idx + 1) / self.model.n_r
                r_l = idx / self.model.n_r
                shell_weight = 0.5 * (r_u ** 3 - r_l ** 3)
                result = result + (cs[:, idx, ...] + cs[:, idx + 1, ...]) * shell_weight
            return result

        self.assertTrue(torch.allclose(self.model.calc_cs_bar(cs_2d), old_loop(cs_2d), rtol=1e-6, atol=1e-8))
        self.assertTrue(torch.allclose(self.model.calc_cs_bar(cs_3d), old_loop(cs_3d), rtol=1e-6, atol=1e-8))

    def test_electrolyte_remains_positive_for_reasonable_current(self):
        x = self.x0[:1].clone()
        current = torch.full((1, 30), 5.0)
        x_roll = self.model(x, current)
        _, c_e_seq, _ = self.model.unpack_state(x_roll)
        self.assertGreater(float(c_e_seq.min()), 0.0)

    def test_electrolyte_linear_operator_matches_rhs(self):
        p = self.p
        generator = torch.Generator().manual_seed(7)
        c_e = 900.0 + 200.0 * torch.rand(3, p.n_ce, generator=generator, dtype=torch.float64)
        current = torch.tensor([[2.0], [-1.5], [0.75]], dtype=torch.float64)
        jn = current / (p.Faraday * p.a_s_n * p.Area_n * p.L_n)
        jp = -current / (p.Faraday * p.a_s_p * p.Area_p * p.L_p)
        A, b = build_electrolyte_linear_operator(c_e, jn, jp, p)
        rhs_linear = (A @ c_e.unsqueeze(-1)).squeeze(-1) + b
        rhs_direct = electrolyte_rhs(c_e, jn, jp, p)
        self.assertLess(float((rhs_linear - rhs_direct).abs().max()), 5e-11)

    def test_fast_electrolyte_operator_matches_debug_operator(self):
        p = self.p
        generator = torch.Generator().manual_seed(11)
        c_e = 800.0 + 400.0 * torch.rand(2, p.n_ce, generator=generator, dtype=torch.float64)
        current = torch.tensor([[4.0], [-3.0]], dtype=torch.float64)
        jn = current / (p.Faraday * p.a_s_n * p.Area_n * p.L_n)
        jp = -current / (p.Faraday * p.a_s_p * p.Area_p * p.L_p)
        A_fast, b_fast = build_electrolyte_linear_operator(c_e, jn, jp, p)
        A_debug, b_debug = build_electrolyte_linear_operator_debug(c_e, jn, jp, p)
        self.assertLess(float((A_fast - A_debug).abs().max()), 1e-12)
        self.assertLess(float((b_fast - b_debug).abs().max()), 1e-12)

    def test_implicit_electrolyte_matches_rk4_for_tiny_step(self):
        cfg_ref = dict(self.cfg)
        cfg_ref.update({'h': 0.001, 'integ_type': 0, 'electrolyte_integrator': 'rk4', 'electrolyte_substeps': 1})
        cfg_imp = dict(self.cfg)
        cfg_imp.update({'h': 0.001, 'integ_type': 0, 'electrolyte_integrator': 'implicit', 'electrolyte_substeps': 1})
        p_ref = InitParamsSPMe(cfg_ref)
        p_imp = InitParamsSPMe(cfg_imp)
        model_ref = SPMeModel(p_ref, cfg_ref)
        model_imp = SPMeModel(p_imp, cfg_imp)
        x0 = model_ref.initial_state(batch_size=1, dtype=torch.float64)
        current = torch.tensor([[2.0]], dtype=torch.float64)

        x_ref = model_ref.step(x0, current)
        x_imp = model_imp.step(x0, current)
        _, ce_ref, _ = model_ref.unpack_state(x_ref)
        _, ce_imp, _ = model_imp.unpack_state(x_imp)
        self.assertLess(float((ce_ref - ce_imp).abs().max()), 1e-4)

        cn_ref, cp_ref = model_ref.calc_css_and_cs0(x_ref, current)
        cn_imp, cp_imp = model_imp.calc_css_and_cs0(x_imp, current)
        v_ref = model_ref.calc_voltage(cn_ref[:, -1], cp_ref[:, -1], ce_ref, current)
        v_imp = model_imp.calc_voltage(cn_imp[:, -1], cp_imp[:, -1], ce_imp, current)
        self.assertLess(float((v_ref - v_imp).abs().max()), 1e-9)

    def test_implicit_electrolyte_matches_rk4_substep_reference(self):
        cfg_ref = dict(self.cfg)
        cfg_ref.update({'h': 0.1, 'integ_type': 0, 'electrolyte_integrator': 'rk4', 'electrolyte_substeps': 10})
        cfg_imp = dict(self.cfg)
        cfg_imp.update({'h': 0.1, 'integ_type': 0, 'electrolyte_integrator': 'implicit', 'electrolyte_substeps': 1})
        p_ref = InitParamsSPMe(cfg_ref)
        p_imp = InitParamsSPMe(cfg_imp)
        model_ref = SPMeModel(p_ref, cfg_ref)
        model_imp = SPMeModel(p_imp, cfg_imp)
        x0 = model_ref.initial_state(batch_size=1, dtype=torch.float64)
        current = torch.tensor([[2.0]], dtype=torch.float64)

        x_ref = model_ref.step(x0, current)
        x_imp = model_imp.step(x0, current)
        _, ce_ref, _ = model_ref.unpack_state(x_ref)
        _, ce_imp, _ = model_imp.unpack_state(x_imp)
        self.assertLess(float((ce_ref - ce_imp).abs().max()), 1e-1)

        cn_ref, cp_ref = model_ref.calc_css_and_cs0(x_ref, current)
        cn_imp, cp_imp = model_imp.calc_css_and_cs0(x_imp, current)
        v_ref = model_ref.calc_voltage(cn_ref[:, -1], cp_ref[:, -1], ce_ref, current)
        v_imp = model_imp.calc_voltage(cn_imp[:, -1], cp_imp[:, -1], ce_imp, current)
        self.assertLess(float((v_ref - v_imp).abs().max()), 1e-4)

    def test_implicit_electrolyte_stable_with_one_second_step(self):
        cfg = dict(self.cfg)
        cfg.update({'h': 1.0, 'electrolyte_integrator': 'implicit', 'electrolyte_substeps': 1})
        p = InitParamsSPMe(cfg)
        model = SPMeModel(p, cfg)
        x = model.initial_state(batch_size=1, dtype=torch.float64)
        current = torch.full((1, 60), 5.0, dtype=torch.float64)
        x_roll = model(x, current)
        _, c_e_seq, _ = model.unpack_state(x_roll)
        self.assertTrue(torch.isfinite(c_e_seq).all())
        self.assertGreater(float(c_e_seq.min()), 0.0)

    def test_voltage_reduces_to_spm_without_electrolyte_terms(self):
        x = self.x0[:1]
        current = torch.tensor([[2.0]])
        c_n, c_p = self.model.calc_css_and_cs0(x, current)
        _, c_e, _ = self.model.unpack_state(x)
        v_spme = self.model.calc_voltage(c_n[:, -1], c_p[:, -1], c_e, current,
                                         disable_electrolyte_ohmic=True,
                                         disable_electrolyte_concentration=True)
        v_spm = self.model.calc_spm_base_voltage(c_n[:, -1], c_p[:, -1], current)
        self.assertTrue(torch.allclose(v_spme, v_spm, atol=1e-8, rtol=1e-6))

    def test_voltage_components_sum_to_voltage(self):
        x = self.x0[:1]
        current = torch.tensor([[2.0]])
        c_n, c_p = self.model.calc_css_and_cs0(x, current)
        _, c_e, _ = self.model.unpack_state(x)
        components = self.model.calc_voltage_components(c_n[:, -1], c_p[:, -1], c_e, current)
        voltage_from_components = (
            components['ocp']
            + components['kinetic']
            - components['film_drop']
            + components['electrolyte_ohmic']
            + components['electrolyte_concentration']
        )
        voltage = self.model.calc_voltage(c_n[:, -1], c_p[:, -1], c_e, current)
        self.assertTrue(torch.allclose(components['voltage'], voltage, atol=1e-12, rtol=1e-12))
        self.assertTrue(torch.allclose(voltage_from_components, voltage, atol=1e-12, rtol=1e-12))

    def test_integration_consistency_uses_window_stride_offsets(self):
        n_samples = 5
        rollout_steps = 4
        stride = 2
        x_nn = torch.arange(n_samples * self.p.state_dim, dtype=torch.float64).reshape(n_samples, self.p.state_dim)
        x_roll = torch.zeros(n_samples, self.p.state_dim, rollout_steps + 1, dtype=torch.float64)
        x_roll[:, :, 2] = x_nn[1:].mean()
        x_roll[:, :, 4] = x_nn[2:].mean()

        loss_bad, stats_bad = _integration_consistency_loss_from_predictions(self.model, x_roll, x_nn, self.p, stride, rollout_steps)

        for offset in (1, 2):
            step = offset * stride
            x_roll[:-offset, :, step] = x_nn[offset:]
        loss_good, stats_good = _integration_consistency_loss_from_predictions(self.model, x_roll, x_nn, self.p, stride, rollout_steps)

        self.assertGreater(float(loss_bad), 0.0)
        loss_bad_expected = (stats_bad['loss_integ_n'] + stats_bad['loss_integ_e'] + stats_bad['loss_integ_p']) / 3.0
        self.assertLess(float((loss_bad - loss_bad_expected).abs()), 1e-12)
        self.assertLess(float(loss_good), 1e-12)
        self.assertLess(float(stats_good['loss_integ_n']), 1e-12)
        self.assertLess(float(stats_good['loss_integ_e']), 1e-12)
        self.assertLess(float(stats_good['loss_integ_p']), 1e-12)

    def test_bare_nn_lithium_inventory_losses_use_sei_derived_targets(self):
        x_identical = self.x0[:1].repeat(4, 1)
        self.model.delta_sei_scaled.requires_grad_(True)
        with torch.no_grad():
            c_n, c_p = self.model.calc_css_and_cs0(
                x_identical,
                torch.zeros(x_identical.shape[0], 1, dtype=x_identical.dtype),
            )
            _, c_e, _ = self.model.unpack_state(x_identical)
            from electrolyte_fvm import electrolyte_lithium_inventory
            n_li_s = self.model.calc_solid_lithium(c_n, c_p).mean()
            n_li_e = electrolyte_lithium_inventory(c_e, self.p).mean()
            self.model.nLi_total.copy_(n_li_s.to(dtype=self.model.nLi_total.dtype))
            self.model.nLi_s_init.copy_(n_li_s.to(dtype=self.model.nLi_s_init.dtype))
            self.model.nLi_e_init.copy_(n_li_e.to(dtype=self.model.nLi_e_init.dtype))
            self.model.delta_sei_scaled.zero_()

        loss_s_identical, loss_e_identical, stats_identical = _bare_nn_lithium_inventory_losses(self.model, x_identical, self.p)
        self.assertLess(float(loss_s_identical.detach()), 1e-7)
        self.assertLess(float(loss_e_identical.detach()), 1e-7)
        self.assertLess(float(stats_identical['nLi_s_rel_std'].detach()), 1e-12)
        self.assertLess(float(stats_identical['nLi_e_rel_std'].detach()), 1e-12)

        x_perturbed = x_identical.clone()
        x_perturbed[1, :self.p.n_cs_n] *= 1.01
        x_perturbed[2, self.p.n_cs_n:self.p.n_cs_n + self.p.n_ce] *= 1.01
        loss_s_perturbed, loss_e_perturbed, stats_perturbed = _bare_nn_lithium_inventory_losses(self.model, x_perturbed, self.p)
        self.assertGreater(float(loss_s_perturbed.detach()), float(loss_s_identical.detach()))
        self.assertGreater(float(loss_e_perturbed.detach()), float(loss_e_identical.detach()))
        self.assertGreater(float(stats_perturbed['nLi_s_rel_std'].detach()), 0.0)
        self.assertGreater(float(stats_perturbed['nLi_e_rel_std'].detach()), 0.0)

        (loss_s_perturbed + loss_e_perturbed).backward()
        self.assertIsNotNone(self.model.delta_sei_scaled.grad)

    def test_true_profile_degradation_uses_true_sei_target_for_plot_actuals(self):
        cfg = dict(self.cfg)
        cfg.update({
            'estimate_delta_sei': True,
            'delta_sei_init': 0.0,
            'delta_sei_true': 5.0e-7,
            'epsilon_e_n_fresh': self.cfg['epsilon_e_n_true'],
        })
        p = InitParamsSPMe(cfg)
        model = SPMeModel(p, cfg)
        ref = torch.zeros(3, p.n_ce, dtype=torch.float64)

        actual = _true_degradation_for_plot(model, 3, ref)
        expected_nli_sei = model.C_li_sei.to(dtype=ref.dtype) * cfg['delta_sei_true']
        expected_nli_s = model.nLi_total.to(dtype=ref.dtype) - expected_nli_sei
        expected_epsilon = model.epsilon_e_n_init.to(dtype=ref.dtype) - p.a_s_n * cfg['delta_sei_true']
        expected_nli_e = model.nLi_e_init.to(dtype=ref.dtype) - model.nLi_e_delta_sei_coeff.to(dtype=ref.dtype) * cfg['delta_sei_true']

        self.assertTrue(torch.allclose(actual['nLi_sei'], torch.full_like(actual['nLi_sei'], expected_nli_sei)))
        self.assertTrue(torch.allclose(actual['nLi_s'], torch.full_like(actual['nLi_s'], expected_nli_s)))
        self.assertTrue(torch.allclose(actual['nLi_e'], torch.full_like(actual['nLi_e'], expected_nli_e)))
        self.assertTrue(torch.allclose(actual['dynamic']['epsilon_e_n'].reshape(()), expected_epsilon.reshape(())))


if __name__ == '__main__':
    unittest.main()
