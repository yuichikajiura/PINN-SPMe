import time

import torch
import torch.nn as nn

from electrolyte_fvm import (
    electrolyte_activity,
    electrolyte_boundary_concentrations,
    electrolyte_conductivity,
    electrolyte_region_means,
    electrolyte_rhs,
    electrolyte_step_implicit,
)


def ref_potential_anode(theta):
    return (1.9793 * torch.exp(-39.3631 * theta)
            + 0.2482
            - 0.09069 * torch.tanh(29.8538 * (theta - 0.1234))
            - 0.04478 * torch.tanh(14.9159 * (theta - 0.2769))
            - 0.0205 * torch.tanh(30.4444 * (theta - 0.6103)))


def ref_potential_cathode(theta):
    return (-0.8090 * theta
            + 4.4875
            - 0.0428 * torch.tanh(18.5138 * (theta - 0.5542))
            - 17.7326 * torch.tanh(15.7890 * (theta - 0.3117))
            + 17.5842 * torch.tanh(15.9308 * (theta - 0.3120)))


class SolidDiffusionFDM(nn.Module):
    def __init__(self, n_r):
        super().__init__()
        self.n_r = n_r
        self.register_buffer('matA', self.calc_matA())
        self.register_buffer('matB', self.calc_matB())
        self.register_buffer('matC', self.calc_matC())
        self.register_buffer('matD', self.calc_matD())

    def calc_matA(self):
        return self.matM1() - self.matM2() @ torch.inverse(self.matN2()) @ self.matN1()

    def calc_matB(self):
        return self.matM2() @ torch.inverse(self.matN2()) @ self.matN3()

    def calc_matC(self):
        return -torch.inverse(self.matN2()) @ self.matN1()

    def calc_matD(self):
        return torch.inverse(self.matN2()) @ self.matN3()

    def matM1(self):
        mat = torch.zeros(self.n_r - 1, self.n_r - 1)
        for idx in range(self.n_r - 1):
            mat[idx, idx] = -2.0
            if idx >= 1:
                mat[idx, idx - 1] = idx / (idx + 1)
            if idx <= self.n_r - 3:
                mat[idx, idx + 1] = (idx + 2) / (idx + 1)
        return mat

    def matM2(self):
        mat = torch.zeros(self.n_r - 1, 2)
        mat[self.n_r - 2, 1] = self.n_r / (self.n_r - 1)
        return mat

    def matN1(self):
        mat = torch.zeros(2, self.n_r - 1)
        mat[0, 0] = 4.0
        mat[0, 1] = -1.0
        mat[1, self.n_r - 3] = 1.0
        mat[1, self.n_r - 2] = -4.0
        return mat

    @staticmethod
    def matN2():
        return torch.tensor([[-3.0, 0.0], [0.0, 3.0]])

    @staticmethod
    def matN3():
        return torch.tensor([[0.0], [1.0]])


class SPMeModel(nn.Module):
    def __init__(self, p, cfg):
        super().__init__()
        self.p = p
        self.cfg = cfg
        self.h = cfg.get('h', 1)
        self.integ_type = cfg.get('integ_type', 1)
        self.n_r = p.Nr
        self.delta_r_bar = 1 / self.n_r
        self.solid_model = SolidDiffusionFDM(self.n_r)
        self.register_buffer('radial_average_weights', self._calc_radial_average_weights(), persistent=False)
        self._timing_enabled = False
        self._timing = {}
        self._update_solid_dynamics()
        param_dtype = torch.get_default_dtype()
        self.register_buffer(
            'nLi_total',
            torch.tensor(float(getattr(p, 'nLi_s_fresh', getattr(p, 'nLi_s_init', p.nLi_s))), dtype=param_dtype),
        )
        self.register_buffer('nLi_s_init', self.nLi_total.detach().clone())
        self.register_buffer(
            'nLi_e_init',
            torch.tensor(float(getattr(p, 'nLi_e_init', p.nLi_e)), dtype=param_dtype),
        )
        self.register_buffer(
            'epsilon_e_n_init',
            torch.tensor(float(getattr(p, 'epsilon_e_n_fresh', p.epsilon_e_n)), dtype=param_dtype),
        )
        self.register_buffer(
            'delta_sei_scale',
            torch.tensor(float(getattr(p, 'delta_sei_scale', 2e-6)), dtype=param_dtype),
        )
        self.register_buffer(
            'C_li_sei_value',
            torch.tensor((2 * p.rho_sei / p.M_sei) * p.a_s_n * p.Area_n * p.L_n, dtype=param_dtype),
        )
        self.register_buffer(
            'nLi_e_delta_sei_coeff',
            torch.tensor(p.A_e * p.c_e0 * p.L_n * p.a_s_n, dtype=param_dtype),
        )
        delta_init = float(getattr(p, 'delta_sei_init', 0.0))
        delta_scale = max(float(self.delta_sei_scale.detach().cpu()), 1e-30)
        self.delta_sei_scaled = nn.Parameter(
            torch.tensor([delta_init / delta_scale], dtype=param_dtype),
            requires_grad=bool(getattr(p, 'estimate_delta_sei', False)),
        )

    @property
    def delta_sei_est(self):
        return self.delta_sei_scaled.to(dtype=self.delta_sei_scale.dtype, device=self.delta_sei_scale.device) * self.delta_sei_scale

    @property
    def C_li_sei(self):
        return self.C_li_sei_value.to(dtype=self.delta_sei_est.dtype, device=self.delta_sei_est.device)

    def nLi_sei_from_delta(self, delta_sei):
        delta_sei = torch.as_tensor(delta_sei, dtype=self.C_li_sei.dtype, device=self.C_li_sei.device)
        return self.C_li_sei * delta_sei

    def nLi_s_from_sei_inventory(self, nLi_sei):
        nLi_sei = torch.as_tensor(nLi_sei, dtype=self.C_li_sei.dtype, device=self.C_li_sei.device)
        return self.nLi_total.to(dtype=nLi_sei.dtype, device=nLi_sei.device) - nLi_sei

    def epsilon_e_n_from_delta(self, delta_sei):
        delta_sei = torch.as_tensor(delta_sei, dtype=self.C_li_sei.dtype, device=self.C_li_sei.device)
        return self.epsilon_e_n_init.to(dtype=delta_sei.dtype, device=delta_sei.device) - self.p.a_s_n * delta_sei

    def nLi_e_from_delta(self, delta_sei):
        delta_sei = torch.as_tensor(delta_sei, dtype=self.C_li_sei.dtype, device=self.C_li_sei.device)
        return (
            self.nLi_e_init.to(dtype=delta_sei.dtype, device=delta_sei.device)
            - self.nLi_e_delta_sei_coeff.to(dtype=delta_sei.dtype, device=delta_sei.device) * delta_sei
        )

    def electrolyte_dynamic_from_delta(self, delta_sei, ref):
        epsilon_e_n = self.epsilon_e_n_from_delta(delta_sei).to(dtype=ref.dtype, device=ref.device)
        return {
            'epsilon_e_n': epsilon_e_n,
            'src_coeff_n': (1 - self.p.t_plus) * self.p.a_s_n / epsilon_e_n,
            'eps_dx_n': epsilon_e_n * self.p.dx_n,
        }

    @property
    def nLi_sei_est(self):
        return self.nLi_sei_from_delta(self.delta_sei_est)

    @property
    def nLi_s_est(self):
        return self.nLi_s_from_sei_inventory(self.nLi_sei_est)

    @property
    def nLi_e_est(self):
        return self.nLi_e_from_delta(self.delta_sei_est)

    @property
    def epsilon_e_n_est(self):
        return self.epsilon_e_n_from_delta(self.delta_sei_est)

    @property
    def R_f_n_est(self):
        base = torch.as_tensor(self.p.R_f_n_base, dtype=self.delta_sei_est.dtype, device=self.delta_sei_est.device)
        return base + self.delta_sei_est / self.p.kappa_sei

    def electrolyte_dynamic(self, ref):
        return self.electrolyte_dynamic_from_delta(self.delta_sei_est, ref)

    def clamp_degradation_parameters(self):
        lower = getattr(self.p, 'delta_sei_lower', None)
        upper = getattr(self.p, 'delta_sei_upper', None)
        if lower is None and upper is None:
            return
        scale = max(float(self.delta_sei_scale.detach().cpu()), 1e-30)
        lower_scaled = -float('inf') if lower is None else float(lower) / scale
        upper_scaled = float('inf') if upper is None else float(upper) / scale
        with torch.no_grad():
            self.delta_sei_scaled.clamp_(lower_scaled, upper_scaled)

    def clamp_inventory_parameters(self):
        self.clamp_degradation_parameters()

    def reset_timing(self, enabled=False):
        self._timing_enabled = bool(enabled)
        self._timing = {
            'solid_update_sec': 0.0,
            'electrolyte_update_sec': 0.0,
        }

    def timing_summary(self):
        return dict(self._timing)

    def _timed_add(self, key, start):
        if self._timing_enabled:
            self._timing[key] = self._timing.get(key, 0.0) + (time.perf_counter() - start)

    @staticmethod
    def _sync_if_cuda(tensor):
        if tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)

    def _calc_radial_average_weights(self):
        weights = torch.zeros(self.n_r + 1)
        for idx in range(self.n_r):
            r_u = (idx + 1) / self.n_r
            r_l = idx / self.n_r
            shell_weight = 0.5 * (r_u ** 3 - r_l ** 3)
            weights[idx] += shell_weight
            weights[idx + 1] += shell_weight
        return weights

    def _beta_base(self, R_s, D_s, a_s, area, length):
        return 2 * self.delta_r_bar * R_s / (D_s * self.p.Faraday * a_s * area * length)

    def _alpha(self, D_s, R_s):
        return D_s / ((R_s * self.delta_r_bar) ** 2)

    def _update_solid_dynamics(self):
        alpha_n = self._alpha(self.p.D_s_n, self.p.R_s_n)
        alpha_p = self._alpha(self.p.D_s_p, self.p.R_s_p)
        beta_n = -self._beta_base(self.p.R_s_n, self.p.D_s_n, self.p.a_s_n, self.p.Area_n, self.p.L_n)
        beta_p = self._beta_base(self.p.R_s_p, self.p.D_s_p, self.p.a_s_p, self.p.Area_p, self.p.L_p)
        device = self.solid_model.matA.device
        dtype = self.solid_model.matA.dtype
        self.register_buffer('matAn', alpha_n * self.solid_model.matA.to(device=device, dtype=dtype), persistent=False)
        self.register_buffer('matBn', alpha_n * beta_n * self.solid_model.matB.to(device=device, dtype=dtype), persistent=False)
        self.register_buffer('matAp', alpha_p * self.solid_model.matA.to(device=device, dtype=dtype), persistent=False)
        self.register_buffer('matBp', alpha_p * beta_p * self.solid_model.matB.to(device=device, dtype=dtype), persistent=False)
        self.register_buffer('matDn', beta_n * self.solid_model.matD.to(device=device, dtype=dtype), persistent=False)
        self.register_buffer('matDp', beta_p * self.solid_model.matD.to(device=device, dtype=dtype), persistent=False)

    def initial_state(self, batch_size=1, device=None, dtype=torch.float32):
        device = device if device is not None else self.solid_model.matA.device
        c_s_n = torch.full((batch_size, self.p.n_cs_n), self.p.csn0, device=device, dtype=dtype)
        c_e = torch.full((batch_size, self.p.n_ce), self.p.c_e0, device=device, dtype=dtype)
        c_s_p = torch.full((batch_size, self.p.n_cs_p), self.p.csp0, device=device, dtype=dtype)
        return self.pack_state(c_s_n, c_e, c_s_p)

    def pack_state(self, c_s_n, c_e, c_s_p):
        return torch.cat([c_s_n, c_e, c_s_p], dim=1)

    def unpack_state(self, x):
        n_end = self.p.n_cs_n
        e_end = n_end + self.p.n_ce
        if x.dim() == 2:
            return x[:, :n_end], x[:, n_end:e_end], x[:, e_end:]
        if x.dim() == 3:
            return x[:, :n_end, :], x[:, n_end:e_end, :], x[:, e_end:, :]
        raise ValueError('State tensor must be 2D or 3D.')

    @staticmethod
    def _linear_rhs(x, current, matA, matB):
        matA = matA.to(dtype=x.dtype, device=x.device)
        matB = matB.to(dtype=x.dtype, device=x.device)
        current = current.to(dtype=x.dtype, device=x.device)
        return x @ matA.T + current @ matB.T

    def _linear_step(self, x, current, matA, matB):
        if self.integ_type == 0:
            k1 = self._linear_rhs(x, current, matA, matB)
            k2 = self._linear_rhs(x + 0.5 * self.h * k1, current, matA, matB)
            k3 = self._linear_rhs(x + 0.5 * self.h * k2, current, matA, matB)
            k4 = self._linear_rhs(x + self.h * k3, current, matA, matB)
            return x + self.h * (k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6)
        return x + self.h * self._linear_rhs(x, current, matA, matB)

    def _electrolyte_step(self, c_e, jn, jp):
        dynamic = self.electrolyte_dynamic(c_e)
        method = self.cfg.get("electrolyte_integrator", "rk4")
        if method == "implicit":
            return electrolyte_step_implicit(c_e, jn, jp, self.p, self.h, dynamic)
        if method != "rk4":
            raise ValueError("electrolyte_integrator must be 'rk4' or 'implicit'.")

        n_sub = self.cfg.get("electrolyte_substeps", 1)
        dt = self.h / n_sub
        ce = c_e

        for _ in range(n_sub):
            k1 = electrolyte_rhs(ce, jn, jp, self.p, dynamic)
            k2 = electrolyte_rhs(ce + 0.5 * dt * k1, jn, jp, self.p, dynamic)
            k3 = electrolyte_rhs(ce + 0.5 * dt * k2, jn, jp, self.p, dynamic)
            k4 = electrolyte_rhs(ce + dt * k3, jn, jp, self.p, dynamic)
            ce = ce + dt * (k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6)

        return ce

    def step(self, x, current):
        current = current.to(dtype=x.dtype, device=x.device)
        c_s_n, c_e, c_s_p = self.unpack_state(x)
        jn = current / (self.p.Faraday * self.p.a_s_n * self.p.Area_n * self.p.L_n)
        jp = -current / (self.p.Faraday * self.p.a_s_p * self.p.Area_p * self.p.L_p)
        if self._timing_enabled:
            self._sync_if_cuda(x)
            t0 = time.perf_counter()
            c_s_n_next = self._linear_step(c_s_n, current, self.matAn, self.matBn)
            self._sync_if_cuda(x)
            t1 = time.perf_counter()
            c_e_next = self._electrolyte_step(c_e, jn, jp)
            self._sync_if_cuda(x)
            t2 = time.perf_counter()
            c_s_p_next = self._linear_step(c_s_p, current, self.matAp, self.matBp)
            self._sync_if_cuda(x)
            t3 = time.perf_counter()
            self._timing['solid_update_sec'] = self._timing.get('solid_update_sec', 0.0) + (t1 - t0) + (t3 - t2)
            self._timing['electrolyte_update_sec'] = self._timing.get('electrolyte_update_sec', 0.0) + (t2 - t1)
        else:
            c_s_n_next = self._linear_step(c_s_n, current, self.matAn, self.matBn)
            c_e_next = self._electrolyte_step(c_e, jn, jp)
            c_s_p_next = self._linear_step(c_s_p, current, self.matAp, self.matBp)
        return self.pack_state(c_s_n_next, c_e_next, c_s_p_next)

    def forward(self, x, current_seq, k=None):
        if k is None:
            k = current_seq.shape[1]
        x_all = torch.zeros(x.shape[0], x.shape[1], k, device=x.device, dtype=x.dtype)
        xk = x
        for step_idx in range(k):
            current = current_seq[:, step_idx:step_idx + 1]
            xk = self.step(xk, current)
            x_all[:, :, step_idx] = xk
        return x_all

    def calc_css_and_cs0(self, x, current):
        current = current.to(dtype=x.dtype, device=x.device)
        c_s_n, _, c_s_p = self.unpack_state(x)
        matC = self.solid_model.matC.to(device=x.device, dtype=x.dtype)
        cs_n = torch.zeros(x.shape[0], self.n_r + 1, device=x.device, dtype=x.dtype)
        cs_p = torch.zeros(x.shape[0], self.n_r + 1, device=x.device, dtype=x.dtype)
        cs0_n = matC[0, 0] * c_s_n[:, 0] + matC[0, 1] * c_s_n[:, 1]
        cs0_p = matC[0, 0] * c_s_p[:, 0] + matC[0, 1] * c_s_p[:, 1]
        css_n = matC[1, self.n_r - 3] * c_s_n[:, self.n_r - 3] + matC[1, self.n_r - 2] * c_s_n[:, self.n_r - 2] + self.matDn[1] * current.squeeze(-1)
        css_p = matC[1, self.n_r - 3] * c_s_p[:, self.n_r - 3] + matC[1, self.n_r - 2] * c_s_p[:, self.n_r - 2] + self.matDp[1] * current.squeeze(-1)
        cs_n[:, 0] = cs0_n
        cs_n[:, 1:-1] = c_s_n
        cs_n[:, -1] = css_n
        cs_p[:, 0] = cs0_p
        cs_p[:, 1:-1] = c_s_p
        cs_p[:, -1] = css_p
        return cs_n, cs_p

    def calc_css_and_cs0_seq(self, x_seq, current_seq):
        current_seq = current_seq.to(dtype=x_seq.dtype, device=x_seq.device)
        c_s_n, _, c_s_p = self.unpack_state(x_seq)
        matC = self.solid_model.matC.to(device=x_seq.device, dtype=x_seq.dtype)
        cs_n = torch.zeros(x_seq.shape[0], self.n_r + 1, x_seq.shape[2], device=x_seq.device, dtype=x_seq.dtype)
        cs_p = torch.zeros(x_seq.shape[0], self.n_r + 1, x_seq.shape[2], device=x_seq.device, dtype=x_seq.dtype)
        cs0_n = matC[0, 0] * c_s_n[:, 0, :] + matC[0, 1] * c_s_n[:, 1, :]
        cs0_p = matC[0, 0] * c_s_p[:, 0, :] + matC[0, 1] * c_s_p[:, 1, :]
        css_n = matC[1, self.n_r - 3] * c_s_n[:, self.n_r - 3, :] + matC[1, self.n_r - 2] * c_s_n[:, self.n_r - 2, :] + self.matDn[1] * current_seq
        css_p = matC[1, self.n_r - 3] * c_s_p[:, self.n_r - 3, :] + matC[1, self.n_r - 2] * c_s_p[:, self.n_r - 2, :] + self.matDp[1] * current_seq
        cs_n[:, 0, :] = cs0_n
        cs_n[:, 1:-1, :] = c_s_n
        cs_n[:, -1, :] = css_n
        cs_p[:, 0, :] = cs0_p
        cs_p[:, 1:-1, :] = c_s_p
        cs_p[:, -1, :] = css_p
        return cs_n, cs_p

    def calc_cs_bar(self, cs):
        weights = self.radial_average_weights.to(device=cs.device, dtype=cs.dtype)
        if cs.dim() == 2:
            return torch.sum(cs * weights.unsqueeze(0), dim=1)
        if cs.dim() == 3:
            return torch.sum(cs * weights.view(1, -1, 1), dim=1)
        raise ValueError('cs must be 2D or 3D.')

    def calc_solid_lithium(self, c_n, c_p):
        cs_bar_n = self.calc_cs_bar(c_n)
        cs_bar_p = self.calc_cs_bar(c_p)
        return cs_bar_n * self.p.epsilon_s_n * self.p.L_n * self.p.Area_n + cs_bar_p * self.p.epsilon_s_p * self.p.L_p * self.p.Area_p

    def _exchange_current_density(self, css, ce, k0, activation_energy):
        arrh = torch.exp(torch.as_tensor(activation_energy / self.p.R * (1 / self.p.T_ref - 1 / self.p.T_amb), dtype=css.dtype, device=css.device))
        return k0 * arrh * torch.clamp(((css.max() * 0) + 1), min=1e-12)

    def calc_spm_base_voltage(self, css_n, css_p, current):
        current = current.to(dtype=css_n.dtype, device=css_n.device)
        css_n = css_n.clamp(1.0, self.p.c_s_n_max * 0.999)
        css_p = css_p.clamp(1.0, self.p.c_s_p_max * 0.999)
        theta_n = css_n / self.p.c_s_n_max
        theta_p = css_p / self.p.c_s_p_max
        ce_ref = torch.full_like(css_n, self.p.c_e)
        i_0n = self.p.k_n0 * torch.exp(torch.as_tensor(self.p.E_kn / self.p.R * (1 / self.p.T_ref - 1 / self.p.T_amb), dtype=css_n.dtype, device=css_n.device)) \
            * torch.clamp((self.p.c_s_n_max - css_n) * css_n * ce_ref, min=1e-12) ** self.p.alph
        i_0p = self.p.k_p0 * torch.exp(torch.as_tensor(self.p.E_kp / self.p.R * (1 / self.p.T_ref - 1 / self.p.T_amb), dtype=css_n.dtype, device=css_n.device)) \
            * torch.clamp((self.p.c_s_p_max - css_p) * css_p * ce_ref, min=1e-12) ** self.p.alph
        rtaf = (self.p.R * self.p.T_amb) / (self.p.alph * self.p.Faraday)
        R_f_n = self.R_f_n_est.to(dtype=css_n.dtype, device=css_n.device)
        return (rtaf * torch.arcsinh(-current / (2 * self.p.a_s_p * self.p.Area_p * self.p.L_p * i_0p))
                - rtaf * torch.arcsinh(current / (2 * self.p.a_s_n * self.p.Area_n * self.p.L_n * i_0n))
                + ref_potential_cathode(theta_p)
                - ref_potential_anode(theta_n)
                - (R_f_n / (self.p.a_s_n * self.p.L_n * self.p.Area_n)
                   + self.p.R_f_p / (self.p.a_s_p * self.p.L_p * self.p.Area_p)) * current)

    def calc_voltage_components(self, css_n, css_p, c_e, current,
                                disable_electrolyte_ohmic=False,
                                disable_electrolyte_concentration=False):
        original_shape = css_n.shape
        css_n_flat = css_n.reshape(-1, 1).clamp(1.0, self.p.c_s_n_max * 0.999)
        css_p_flat = css_p.reshape(-1, 1).clamp(1.0, self.p.c_s_p_max * 0.999)
        current_flat = current.to(dtype=css_n_flat.dtype, device=css_n_flat.device).reshape(-1, 1)

        if c_e.dim() == 2:
            c_e_flat = c_e
        elif c_e.dim() == 3:
            c_e_flat = c_e.permute(0, 2, 1).reshape(-1, c_e.shape[1])
        else:
            raise ValueError('c_e must be 2D or 3D.')

        dynamic = self.electrolyte_dynamic(c_e_flat)
        ce0n, cens, cesp, ce0p, c_ex = electrolyte_boundary_concentrations(c_e_flat, self.p, dynamic)

        cen_bar = c_ex[:, :self.p.Nxn + 1].mean(dim=1, keepdim=True)
        ces_bar = c_ex[:, self.p.Nxn:self.p.Nxn + self.p.Nxs + 1].mean(dim=1, keepdim=True)
        cep_bar = c_ex[:, self.p.Nxn + self.p.Nxs:self.p.Nxn + self.p.Nxs + self.p.Nxp + 1].mean(dim=1, keepdim=True)

        # Voltage/OCP uses stoichiometry from raw concentration and theoretical
        # maximum concentration. NN practical min/max scaling is separate.
        theta_n = css_n_flat / self.p.c_s_n_max
        theta_p = css_p_flat / self.p.c_s_p_max
        unref = ref_potential_anode(theta_n)
        upref = ref_potential_cathode(theta_p)

        arrh_kn = torch.exp(torch.as_tensor(self.p.E_kn / self.p.R * (1 / self.p.T_ref - 1 / self.p.T_amb), dtype=css_n_flat.dtype, device=css_n_flat.device))
        arrh_kp = torch.exp(torch.as_tensor(self.p.E_kp / self.p.R * (1 / self.p.T_ref - 1 / self.p.T_amb), dtype=css_n_flat.dtype, device=css_n_flat.device))
        i_0n = self.p.k_n0 * arrh_kn * ((self.p.c_s_n_max - css_n_flat) * css_n_flat * cen_bar) ** self.p.alph
        i_0p = self.p.k_p0 * arrh_kp * ((self.p.c_s_p_max - css_p_flat) * css_p_flat * cep_bar) ** self.p.alph

        rtaf = (self.p.R * self.p.T_amb) / (self.p.alph * self.p.Faraday)
        eta_p = rtaf * torch.arcsinh(-current_flat / (2 * self.p.a_s_p * self.p.Area_p * self.p.L_p * i_0p))
        eta_n = rtaf * torch.arcsinh(current_flat / (2 * self.p.a_s_n * self.p.Area_n * self.p.L_n * i_0n))
        R_f_n = self.R_f_n_est.to(dtype=css_n_flat.dtype, device=css_n_flat.device)
        epsilon_e_n = self.epsilon_e_n_est.to(dtype=css_n_flat.dtype, device=css_n_flat.device)
        film_drop = (R_f_n / (self.p.a_s_n * self.p.L_n * self.p.Area_n)
                     + self.p.R_f_p / (self.p.a_s_p * self.p.L_p * self.p.Area_p)) * current_flat

        kap_n_eff = electrolyte_conductivity(cen_bar, self.p) * (epsilon_e_n ** self.p.brug)
        kap_s_eff = electrolyte_conductivity(ces_bar, self.p) * (self.p.epsilon_e_s ** self.p.brug)
        kap_p_eff = electrolyte_conductivity(cep_bar, self.p) * (self.p.epsilon_e_p ** self.p.brug)
        v_ohmic = (self.p.L_n / (2 * kap_n_eff) + self.p.L_s / kap_s_eff + self.p.L_p / (2 * kap_p_eff)) * current_flat

        if getattr(self.p, 'use_electrolyte_activity', False):
            dfca_n = electrolyte_activity(cen_bar, self.p)
            dfca_s = electrolyte_activity(ces_bar, self.p)
            dfca_p = electrolyte_activity(cep_bar, self.p)
        else:
            dfca_n = torch.zeros_like(cen_bar)
            dfca_s = torch.zeros_like(ces_bar)
            dfca_p = torch.zeros_like(cep_bar)

        v_conc = (2 * self.p.R * self.p.T_amb / self.p.Faraday) * (
            (1 + dfca_n) * (torch.log(cens.clamp_min(1.0)) - torch.log(ce0n.clamp_min(1.0)))
            + (1 + dfca_s) * (torch.log(cesp.clamp_min(1.0)) - torch.log(cens.clamp_min(1.0)))
            + (1 + dfca_p) * (torch.log(ce0p.clamp_min(1.0)) - torch.log(cesp.clamp_min(1.0)))
        )

        if disable_electrolyte_ohmic:
            v_ohmic = torch.zeros_like(v_ohmic)
        if disable_electrolyte_concentration:
            v_conc = torch.zeros_like(v_conc)

        ocp = upref - unref
        kinetic = eta_p - eta_n
        voltage = ocp + kinetic - film_drop + v_ohmic + v_conc

        def shaped(value):
            return value.reshape(original_shape)

        return {
            'voltage': shaped(voltage),
            'U_p': shaped(upref),
            'U_n': shaped(unref),
            'ocp': shaped(ocp),
            'eta_p': shaped(eta_p),
            'eta_n': shaped(eta_n),
            'kinetic': shaped(kinetic),
            'electrolyte_ohmic': shaped(v_ohmic),
            'electrolyte_concentration': shaped(v_conc),
            'film_drop': shaped(film_drop),
            'theta_p': shaped(theta_p),
            'theta_n': shaped(theta_n),
            'c_e_bar_n': shaped(cen_bar),
            'c_e_bar_s': shaped(ces_bar),
            'c_e_bar_p': shaped(cep_bar),
        }

    def calc_voltage(self, css_n, css_p, c_e, current, disable_electrolyte_ohmic=False, disable_electrolyte_concentration=False):
        return self.calc_voltage_components(
            css_n,
            css_p,
            c_e,
            current,
            disable_electrolyte_ohmic=disable_electrolyte_ohmic,
            disable_electrolyte_concentration=disable_electrolyte_concentration,
        )['voltage']
