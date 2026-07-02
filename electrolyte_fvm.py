import torch


def _harmonic_mean(a, b):
    return 2.0 * a * b / (a + b)


def electrolyte_diffusivity(c_e, p):
    arrh = torch.exp(torch.as_tensor(p.E_De / p.R * (1 / p.T_ref - 1 / p.T_amb), dtype=c_e.dtype, device=c_e.device))
    c_scaled = c_e / 1000.0
    return (8.794e-11 * c_scaled ** 2 - 3.972e-10 * c_scaled + 4.862e-10) * arrh


def electrolyte_conductivity(c_e, p):
    arrh = torch.exp(torch.as_tensor(p.E_kappa_e / p.R * (1 / p.T_ref - 1 / p.T_amb), dtype=c_e.dtype, device=c_e.device))
    c_scaled = c_e / 1000.0
    return (0.1297 * c_scaled ** 3 - 2.51 * c_scaled ** 1.5 + 3.329 * c_scaled) * arrh


def electrolyte_activity(c_e, p):
    v00 = 0.601
    v10 = -0.24
    v30 = 0.982
    v31 = -0.0052
    c_scaled = c_e / 1000.0
    return ((v00 + v10 * c_scaled.sqrt() + v30 * (1 + v31 * (p.T_amb - p.T_ref)) * c_scaled ** 1.5)
            / (1 - p.t_plus)) - 1


def split_electrolyte_regions(c_e, p):
    n_end = p.Nxn - 1
    s_end = n_end + p.Nxs - 1
    c_en = c_e[..., :n_end]
    c_es = c_e[..., n_end:s_end]
    c_ep = c_e[..., s_end:]
    return c_en, c_es, c_ep


def _dynamic_value(dynamic, name, default, ref):
    if dynamic is None or name not in dynamic or dynamic[name] is None:
        return torch.as_tensor(default, dtype=ref.dtype, device=ref.device)
    return torch.as_tensor(dynamic[name], dtype=ref.dtype, device=ref.device)


def _dynamic_electrolyte_terms(p, ref, dynamic=None):
    epsilon_e_n = _dynamic_value(dynamic, 'epsilon_e_n', p.epsilon_e_n, ref).reshape(())
    src_coeff_n = _dynamic_value(dynamic, 'src_coeff_n', (1 - p.t_plus) * p.a_s_n / p.epsilon_e_n, ref).reshape(())
    eps_dx_n = _dynamic_value(dynamic, 'eps_dx_n', p.eps_dx_n, ref).reshape(())
    return epsilon_e_n, src_coeff_n, eps_dx_n


def build_C_ce_dynamic(p, epsilon_e_n, dtype=None, device=None):
    dtype = dtype or epsilon_e_n.dtype
    device = device or epsilon_e_n.device
    epsilon_e_n = torch.as_tensor(epsilon_e_n, dtype=dtype, device=device).reshape(())

    Del_xn = torch.as_tensor(p.L_n * p.delta_x_n, dtype=dtype, device=device)
    Del_xs = torch.as_tensor(p.L_s * p.delta_x_s, dtype=dtype, device=device)
    Del_xp = torch.as_tensor(p.L_p * p.delta_x_p, dtype=dtype, device=device)
    eps_s = torch.as_tensor(p.epsilon_e_s, dtype=dtype, device=device)
    eps_p = torch.as_tensor(p.epsilon_e_p, dtype=dtype, device=device)

    N1 = torch.zeros((4, p.Nx - 3), dtype=dtype, device=device)
    N2 = torch.zeros((4, 4), dtype=dtype, device=device)

    N1[0, 0] = 4.0
    N1[0, 1] = -1.0
    N2[0, 0] = -3.0

    eps_n_brug = epsilon_e_n ** p.brug
    eps_s_brug = eps_s ** p.brug
    eps_p_brug = eps_p ** p.brug
    N1[1, p.Nxn - 3] = eps_n_brug / (2 * Del_xn)
    N1[1, p.Nxn - 2] = (-4 * eps_n_brug) / (2 * Del_xn)
    N2[1, 1] = (3 * eps_n_brug) / (2 * Del_xn) + (3 * eps_s_brug) / (2 * Del_xs)
    N1[1, p.Nxn - 1] = (-4 * eps_s_brug) / (2 * Del_xs)
    N1[1, p.Nxn] = eps_s_brug / (2 * Del_xs)

    N1[2, p.Nxn + p.Nxs - 4] = eps_s_brug / (2 * Del_xs)
    N1[2, p.Nxn + p.Nxs - 3] = (-4 * eps_s_brug) / (2 * Del_xs)
    N2[2, 2] = (3 * eps_s_brug) / (2 * Del_xs) + (3 * eps_p_brug) / (2 * Del_xp)
    N1[2, p.Nxn + p.Nxs - 2] = (-4 * eps_p_brug) / (2 * Del_xp)
    N1[2, p.Nxn + p.Nxs - 1] = eps_p_brug / (2 * Del_xp)

    N1[3, -2] = 1.0
    N1[3, -1] = -4.0
    N2[3, 3] = 3.0
    return -torch.linalg.solve(N2, N1)


def _effective_electrolyte_diffusivities(c_e, p, dynamic=None):
    c_en, c_es, c_ep = split_electrolyte_regions(c_e, p)
    epsilon_e_n, _, _ = _dynamic_electrolyte_terms(p, c_e, dynamic)
    D_en_eff = electrolyte_diffusivity(c_en, p) * (epsilon_e_n ** (p.brug - 1))
    D_es_eff = electrolyte_diffusivity(c_es, p) * (p.epsilon_e_s ** (p.brug - 1))
    D_ep_eff = electrolyte_diffusivity(c_ep, p) * (p.epsilon_e_p ** (p.brug - 1))
    return D_en_eff, D_es_eff, D_ep_eff


def _electrolyte_rhs_frozen_diffusivity(c_e, jn, jp, p, D_en_eff, D_es_eff, D_ep_eff, dynamic=None):
    if c_e.dim() != 2:
        raise ValueError('electrolyte RHS expects c_e with shape [batch, n_ce].')

    batch_size = c_e.shape[0]
    c_en, c_es, c_ep = split_electrolyte_regions(c_e, p)

    dx_n = p.dx_n
    dx_s = p.dx_s
    dx_p = p.dx_p
    _, src_coeff_n, eps_dx_n = _dynamic_electrolyte_terms(p, c_e, dynamic)

    src_n = src_coeff_n * jn.expand(batch_size, c_en.shape[1])
    src_s = torch.zeros_like(c_es)
    src_p = p.src_coeff_p * jp.expand(batch_size, c_ep.shape[1])

    F_n = torch.zeros(batch_size, max(c_en.shape[1] - 1, 0), dtype=c_e.dtype, device=c_e.device)
    F_s = torch.zeros(batch_size, max(c_es.shape[1] - 1, 0), dtype=c_e.dtype, device=c_e.device)
    F_p = torch.zeros(batch_size, max(c_ep.shape[1] - 1, 0), dtype=c_e.dtype, device=c_e.device)

    if c_en.shape[1] > 1:
        F_n = -_harmonic_mean(D_en_eff[:, :-1], D_en_eff[:, 1:]) * (c_en[:, 1:] - c_en[:, :-1]) / dx_n
    if c_es.shape[1] > 1:
        F_s = -_harmonic_mean(D_es_eff[:, :-1], D_es_eff[:, 1:]) * (c_es[:, 1:] - c_es[:, :-1]) / dx_s
    if c_ep.shape[1] > 1:
        F_p = -_harmonic_mean(D_ep_eff[:, :-1], D_ep_eff[:, 1:]) * (c_ep[:, 1:] - c_ep[:, :-1]) / dx_p

    F_ns = -_harmonic_mean(D_en_eff[:, -1:], D_es_eff[:, :1]) * (c_es[:, :1] - c_en[:, -1:]) / (0.5 * dx_n + 0.5 * dx_s)
    F_sp = -_harmonic_mean(D_es_eff[:, -1:], D_ep_eff[:, :1]) * (c_ep[:, :1] - c_es[:, -1:]) / (0.5 * dx_s + 0.5 * dx_p)

    c_en_dot = torch.zeros_like(c_en)
    c_es_dot = torch.zeros_like(c_es)
    c_ep_dot = torch.zeros_like(c_ep)

    for idx in range(c_en.shape[1]):
        f_left = torch.zeros(batch_size, 1, dtype=c_e.dtype, device=c_e.device) if idx == 0 else F_n[:, idx - 1:idx]
        f_right = F_ns if idx == c_en.shape[1] - 1 else F_n[:, idx:idx + 1]
        c_en_dot[:, idx:idx + 1] = (f_left - f_right) / eps_dx_n + src_n[:, idx:idx + 1]

    for idx in range(c_es.shape[1]):
        f_left = F_ns if idx == 0 else F_s[:, idx - 1:idx]
        f_right = F_sp if idx == c_es.shape[1] - 1 else F_s[:, idx:idx + 1]
        c_es_dot[:, idx:idx + 1] = (f_left - f_right) / p.eps_dx_s + src_s[:, idx:idx + 1]

    for idx in range(c_ep.shape[1]):
        f_left = F_sp if idx == 0 else F_p[:, idx - 1:idx]
        f_right = torch.zeros(batch_size, 1, dtype=c_e.dtype, device=c_e.device) if idx == c_ep.shape[1] - 1 else F_p[:, idx:idx + 1]
        c_ep_dot[:, idx:idx + 1] = (f_left - f_right) / p.eps_dx_p + src_p[:, idx:idx + 1]

    return torch.cat([c_en_dot, c_es_dot, c_ep_dot], dim=1)


def electrolyte_rhs(c_e, jn, jp, p, dynamic=None):
    if c_e.dim() != 2:
        raise ValueError('electrolyte_rhs expects c_e with shape [batch, n_ce].')

    D_en_eff, D_es_eff, D_ep_eff = _effective_electrolyte_diffusivities(c_e, p, dynamic)
    return _electrolyte_rhs_frozen_diffusivity(c_e, jn, jp, p, D_en_eff, D_es_eff, D_ep_eff, dynamic)


def build_electrolyte_linear_operator(c_e, jn, jp, p, dynamic=None):
    """Build frozen-diffusivity FVM operator for ``dc_e/dt = A @ c_e + b``."""
    if c_e.dim() != 2:
        raise ValueError('build_electrolyte_linear_operator expects c_e with shape [batch, n_ce].')

    batch_size, n_ce = c_e.shape
    dtype = c_e.dtype
    device = c_e.device
    D_en_eff, D_es_eff, D_ep_eff = _effective_electrolyte_diffusivities(c_e, p, dynamic)
    _, src_coeff_n, eps_dx_n = _dynamic_electrolyte_terms(p, c_e, dynamic)

    A = torch.zeros(batch_size, n_ce, n_ce, dtype=dtype, device=device)
    n_len = p.Nxn - 1
    s_len = p.Nxs - 1
    p_len = p.Nxp - 1
    s0 = n_len
    p0 = n_len + s_len

    def add_interface(left_idx, right_idx, conductance, left_volume, right_volume):
        A[:, left_idx, left_idx] -= conductance / left_volume
        A[:, left_idx, right_idx] += conductance / left_volume
        A[:, right_idx, left_idx] += conductance / right_volume
        A[:, right_idx, right_idx] -= conductance / right_volume

    if n_len > 1:
        g_n = _harmonic_mean(D_en_eff[:, :-1], D_en_eff[:, 1:]) * p.inv_dx_n
        for idx in range(n_len - 1):
            add_interface(idx, idx + 1, g_n[:, idx], eps_dx_n, eps_dx_n)

    if s_len > 1:
        g_s = _harmonic_mean(D_es_eff[:, :-1], D_es_eff[:, 1:]) * p.inv_dx_s
        for idx in range(s_len - 1):
            add_interface(s0 + idx, s0 + idx + 1, g_s[:, idx], p.eps_dx_s, p.eps_dx_s)

    if p_len > 1:
        g_p = _harmonic_mean(D_ep_eff[:, :-1], D_ep_eff[:, 1:]) * p.inv_dx_p
        for idx in range(p_len - 1):
            add_interface(p0 + idx, p0 + idx + 1, g_p[:, idx], p.eps_dx_p, p.eps_dx_p)

    g_ns = _harmonic_mean(D_en_eff[:, -1], D_es_eff[:, 0]) / (0.5 * p.dx_n + 0.5 * p.dx_s)
    add_interface(n_len - 1, s0, g_ns, eps_dx_n, p.eps_dx_s)

    g_sp = _harmonic_mean(D_es_eff[:, -1], D_ep_eff[:, 0]) / (0.5 * p.dx_s + 0.5 * p.dx_p)
    add_interface(s0 + s_len - 1, p0, g_sp, p.eps_dx_s, p.eps_dx_p)

    b = torch.zeros_like(c_e)
    b[:, :n_len] = src_coeff_n * jn.expand(batch_size, n_len)
    b[:, p0:] = p.src_coeff_p * jp.expand(batch_size, p_len)
    return A, b


def build_electrolyte_linear_operator_debug(c_e, jn, jp, p, dynamic=None):
    """Reference frozen-diffusivity operator assembled by basis probing."""
    if c_e.dim() != 2:
        raise ValueError('build_electrolyte_linear_operator_debug expects c_e with shape [batch, n_ce].')

    batch_size, n_ce = c_e.shape
    dtype = c_e.dtype
    device = c_e.device
    D_en_eff, D_es_eff, D_ep_eff = _effective_electrolyte_diffusivities(c_e, p, dynamic)

    zeros = torch.zeros_like(c_e)
    zero_jn = torch.zeros_like(jn)
    zero_jp = torch.zeros_like(jp)
    b = _electrolyte_rhs_frozen_diffusivity(zeros, jn, jp, p, D_en_eff, D_es_eff, D_ep_eff, dynamic)

    columns = []
    for col_idx in range(n_ce):
        basis = torch.zeros(batch_size, n_ce, dtype=dtype, device=device)
        basis[:, col_idx] = 1.0
        columns.append(
            _electrolyte_rhs_frozen_diffusivity(
                basis, zero_jn, zero_jp, p, D_en_eff, D_es_eff, D_ep_eff, dynamic
            )
        )
    A = torch.stack(columns, dim=-1)
    return A, b


def electrolyte_step_implicit(c_e, jn, jp, p, dt, dynamic=None):
    """
    Semi-implicit electrolyte update:
        c_next = c + dt * rhs(c_next, source)
    Nonlinear diffusivity is frozen at the current electrolyte concentration.
    """
    dtype = c_e.dtype
    device = c_e.device
    A, b = build_electrolyte_linear_operator(c_e, jn, jp, p, dynamic)
    I = torch.eye(A.shape[-1], dtype=dtype, device=device)
    lhs = I.unsqueeze(0) - dt * A
    rhs = c_e.unsqueeze(-1) + dt * b.unsqueeze(-1)
    c_next = torch.linalg.solve(lhs, rhs).squeeze(-1)
    return c_next


def reconstruct_full_concentration(c_e, p, dynamic=None):
    c_en, c_es, c_ep = split_electrolyte_regions(c_e, p)

    if dynamic is not None and 'epsilon_e_n' in dynamic and dynamic['epsilon_e_n'] is not None:
        C_ce = build_C_ce_dynamic(p, dynamic['epsilon_e_n'], dtype=c_e.dtype, device=c_e.device)
    else:
        C_ce = p.C_ce.to(dtype=c_e.dtype, device=c_e.device)
    c_e_bcs = c_e @ C_ce.T

    return torch.cat([
        c_e_bcs[:, 0:1],
        c_en,
        c_e_bcs[:, 1:2],
        c_es,
        c_e_bcs[:, 2:3],
        c_ep,
        c_e_bcs[:, 3:4],
    ], dim=1)

def electrolyte_boundary_concentrations(c_e, p, dynamic=None):
    c_ex = reconstruct_full_concentration(c_e, p, dynamic)
    ce0n = c_ex[:, :1]
    cens = c_ex[:, p.Nxn:p.Nxn + 1]
    cesp = c_ex[:, p.Nxn + p.Nxs:p.Nxn + p.Nxs + 1]
    ce0p = c_ex[:, -1:]
    return ce0n, cens, cesp, ce0p, c_ex


def electrolyte_region_means(c_e, p, dynamic=None):
    c_ex = reconstruct_full_concentration(c_e, p, dynamic)
    cen_bar = c_ex[:, :p.Nxn + 1].mean(dim=1, keepdim=True)
    ces_bar = c_ex[:, p.Nxn:p.Nxn + p.Nxs + 1].mean(dim=1, keepdim=True)
    cep_bar = c_ex[:, p.Nxn + p.Nxs:p.Nxn + p.Nxs + p.Nxp + 1].mean(dim=1, keepdim=True)
    return cen_bar, ces_bar, cep_bar


def electrolyte_lithium_inventory(c_e, p, dynamic=None):
    c_en, c_es, c_ep = split_electrolyte_regions(c_e, p)
    _, _, eps_dx_n = _dynamic_electrolyte_terms(p, c_e, dynamic)
    n_li_n = p.A_e * eps_dx_n * c_en.sum(dim=-1)
    n_li_s = p.A_e * p.eps_dx_s * c_es.sum(dim=-1)
    n_li_p = p.A_e * p.eps_dx_p * c_ep.sum(dim=-1)
    return n_li_n + n_li_s + n_li_p
