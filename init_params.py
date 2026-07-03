import torch

class InitParams:
    def __init__(self, cfg):
        'Battery parameters'
        # Geometric params
        self.L_n = 100e-6  # Thickness of negative electrode [m]
        self.L_p = 100e-6  # Thickness of positive electrode [m]
        self.A_n = 1  # Area of negative electrode [m2]
        self.A_p = 1  # Area of negative electrode [m2]
        self.R_s_n = 10e-6  # Radius of solid particles in negative electrode [m]
        self.R_s_p = 10e-6  # Radius of solid particles in positive electrode [m]
        self.epsilon_s_n = 0.6  # Volume fraction in solid for neg.electrode
        self.epsilon_s_p = 0.5  # Volume fraction in solid for pos.electrode
        self.density_n = 1800  # Mass density of anode material [kg/m3]
        self.density_p = 5010  # Mass density of anode material [kg/m3]
        self.a_s_n = 3 * self.epsilon_s_n / self.R_s_n  # Specific surface area for neg.electrode [m^2/m^3]
        self.a_s_p = 3 * self.epsilon_s_p / self.R_s_p  # Specific surface area for pos.electrode [m^2/m^3]

        # Transport Params
        self.D_s_n = 3.9e-14  # Diffusion coeff for solid in neg. electrode, [m^2/s]
        self.D_s_p = 1e-13  # Diffusion coeff for solid in pos. electrode, [m^2/s]
        self.Faraday = 96487  # Faraday's constant, [Coulumbs/mol]

        # Kinetic params
        self.R = 8.314472  # Gas constant [J/mol-K]
        self.alph = 0.5  # Charge transfer coefficients
        self.R_f_n = 1e-3  # Resistivity of SEI layer, [Ohms * m ^ 2]
        self.R_f_p = 0  # Resistivity of SEI layer, [Ohms * m ^ 2]
        # Reaction rate constant[m2.5mol - 0.5s - 1]*[C / mol] = [(A / m ^ 2) * (m ^ 3 / mol) ^ (1.5)] from
        # https: // github.com / davidhowey / Spectral_li - ion_SPM / blob / master / model_parameters / get_modelData.m
        self.k_n = 1.764e-11 * self.Faraday
        self.k_p = 6.667e-11 * self.Faraday

        # Thermodynamics params
        self.T_amb = 298  # Ambient temperature in [K]

        # Concentrations
        self.specific_capa_n = 372  # Anode material's specific capacity [mAh/g]
        self.specific_capa_p = 247  # Cathode material's specific capacity [mAh/g]
        self.c_s_n_max = 3.6e3 * self.specific_capa_n * self.density_n \
                         / self.Faraday  # Max concentration in anode, [mol / m ^ 3]
        self.c_s_p_max = 3.6e3 * self.specific_capa_p * self.density_p \
                         / self.Faraday  # Max concentration in cathode, [mol / m ^ 3]
        self.nLi_s = 2.50  # Total moles of lithium in solid phase[mol]
        self.c_e = 1e3  # Fixed electrolyte concentration for SPM, [mol / m ^ 3]

        # Set true value of the parameter to be estimated
        if cfg['cell_target'] == 1:  # fresh cell
            self.nLi_s_true = self.nLi_s
            self.R_f_n_true = self.R_f_n
            self.k_n_true = self.k_n
            self.k_p_true = self.k_p
            self.D_s_n_true = self.D_s_n
            self.D_s_p_true = self.D_s_p
        else:  # degraded cell
            self.nLi_s_true = self.nLi_s * 0.8
            self.R_f_n_true = self.R_f_n * 40
            self.k_n_true = self.k_n
            self.k_p_true = self.k_p
            self.D_s_n_true = self.D_s_n
            self.D_s_p_true = self.D_s_p

        # Correct parameter if pre-trained NN is trained on degraded cell
        if cfg['cell_known'] == 2:  # degraded cell
            self.nLi_s = self.nLi_s * 0.8
            self.R_f_n = self.R_f_n * 40
            self.k_n = self.k_n
            self.k_p = self.k_p
            self.D_s_n = self.D_s_n
            self.D_s_p = self.D_s_p

        # Set true value if the parameter is not to be estimated
        if not cfg['p_targets'][0]:
            self.nLi_s = self.nLi_s_true
        if not cfg['p_targets'][1]:
            self.R_f_n = self.R_f_n_true
        if not cfg['p_targets'][2]:
            self.k_n = self.k_n_true
        if not cfg['p_targets'][3]:
            self.k_p = self.k_p_true
        if not cfg['p_targets'][4]:
            self.D_s_n = self.D_s_n_true
        if not cfg['p_targets'][5]:
            self.D_s_p = self.D_s_p_true


class InitParamsSPMe:
    def __init__(self, cfg):
        """SPMe parameter set based on the LGM50 MATLAB implementation."""

        # Model identifier
        self.model_type = 'spme'

        # Geometric parameters
        self.L_n = 8.52e-5
        self.L_s = 1.2e-5
        self.L_p = 7.56e-5
        self.L_ccn = 1.2e-5
        self.L_ccp = 1.6e-6
        self.R_s_n = 5.86e-6
        self.R_s_p = 5.22e-6

        # Volume fractions
        self.epsilon_s_n = 0.75
        self.epsilon_s_p = 0.665
        self.epsilon_e_n = 0.25
        self.epsilon_e_s = 0.47
        self.epsilon_e_p = 0.335
        self.epsilon_f_n = 1 - self.epsilon_s_n - self.epsilon_e_n
        self.epsilon_f_p = 1 - self.epsilon_s_p - self.epsilon_e_p

        # Areas / aliases
        self.Area_n = 0.1027
        self.Area_s = 0.1027
        self.Area_p = 0.1027
        self.Area_ref = (self.Area_n + self.Area_p) / 2
        self.A_n = self.Area_n
        self.A_p = self.Area_p
        self.A_e = self.Area_ref

        # Derived interfacial areas
        self.a_s_n = 3 * self.epsilon_s_n / self.R_s_n
        self.a_s_p = 3 * self.epsilon_s_p / self.R_s_p
        self.estimate_delta_sei = bool(cfg['estimate_delta_sei'])
        self.delta_sei_init = float(cfg['delta_sei_init'])
        self.delta_sei_lower = cfg['delta_sei_lower']
        self.delta_sei_upper = cfg['delta_sei_upper']
        self.delta_sei_scale = float(cfg.get('delta_sei_scale', cfg['delta_sei_upper']))
        self.delta_sei_true = float(cfg['delta_sei_true'])
        self.epsilon_e_n_true = float(cfg['epsilon_e_n_true'])
        self.nLi_s_fresh = float(cfg['nLi_s_fresh'])
        self.nLi_s_init = float(cfg['nLi_s_init'])
        self.nLi_e_init_cfg = cfg['nLi_e_init']
        self.epsilon_e_n_fresh = float(cfg.get('epsilon_e_n_fresh', self.epsilon_e_n))
        self.R_f_n_base = 1e-3

        if self.estimate_delta_sei:
            self.epsilon_e_n = self.epsilon_e_n - self.a_s_n * self.delta_sei_init
            self.epsilon_f_n = 1 - self.epsilon_s_n - self.epsilon_e_n

        # Electrochemical parameters
        self.D_s_n = 3.3e-14
        self.D_s_p = 4e-15
        self.D_s_n_true = self.D_s_n
        self.D_s_p_true = self.D_s_p
        self.brug = 1.5
        self.t_plus = 0.2594
        self.Faraday = 96485.33212
        self.R = 8.314472
        self.alph = 0.5
        self.R_f_p = 0.0
        self.R_c = 0.0
        self.rho_sei = 1690.0
        self.M_sei = 0.162
        self.kappa_sei = 5.0e-6
        self.R_f_n = self.R_f_n_base
        if self.estimate_delta_sei:
            self.R_f_n = self.R_f_n_base + self.delta_sei_init / self.kappa_sei
        self.k_n0 = 6.48e-7
        self.k_p0 = 3.42e-6
        self.k_n = self.k_n0
        self.k_p = self.k_p0

        # Thermal / Arrhenius parameters
        self.T_amb = cfg['temperature_K']
        self.T_ref = 298.15
        self.use_electrolyte_activity = bool(cfg['use_electrolyte_activity'])
        self.E_kn = 35e3
        self.E_kp = 17800
        self.E_Dsn = 3.03e4
        self.E_Dsp = 25000
        self.E_De = 17000
        self.E_kappa_e = 17000

        # Concentrations
        self.c_s_n_max = 33133.0
        self.c_s_p_max = 63104.0
        self.csn0 = 29866.0
        self.csp0 = 17038.0
        # LGM50 SoC endpoint concentrations from init_cs(p, p.volt_min/max).
        self.c_n_soc0 = 872.669821771
        self.c_p_soc0 = 53889.5295856
        self.c_n_soc100 = 30174.8017348
        self.c_p_soc100 = 16645.5022705
        self.c_n_min = self.c_n_soc0
        self.c_n_max = self.c_n_soc100
        self.c_p_min = self.c_p_soc100
        self.c_p_max = self.c_p_soc0
        self.c_e = 1e3
        self.c_e0 = self.c_e
        self.ce_min = cfg['ce_min']
        self.ce_max = cfg['ce_max']
        self.n_Li_s = 0.2840
        self.nLi_s = self.n_Li_s

        # Voltage limits and current scale
        self.volt_max = 4.2
        self.volt_min = 2.5
        self.OneC = cfg['OneC']

        # Discretization
        self.Nr = cfg['n_r']
        self.Nxn = cfg['n_x_n']
        self.Nxs = cfg['n_x_s']
        self.Nxp = cfg['n_x_p']
        self.Nx = self.Nxn + self.Nxs + self.Nxp
        self.delta_x_n = 1 / self.Nxn
        self.delta_x_s = 1 / self.Nxs
        self.delta_x_p = 1 / self.Nxp
        self.dx_n = self.L_n * self.delta_x_n
        self.dx_s = self.L_s * self.delta_x_s
        self.dx_p = self.L_p * self.delta_x_p
        self.inv_dx_n = 1.0 / self.dx_n
        self.inv_dx_s = 1.0 / self.dx_s
        self.inv_dx_p = 1.0 / self.dx_p
        self.src_coeff_n = (1 - self.t_plus) * self.a_s_n / self.epsilon_e_n
        self.src_coeff_p = (1 - self.t_plus) * self.a_s_p / self.epsilon_e_p
        self.eps_dx_n = self.epsilon_e_n * self.dx_n
        self.eps_dx_s = self.epsilon_e_s * self.dx_s
        self.eps_dx_p = self.epsilon_e_p * self.dx_p
        self.delta_t = cfg['h']

        # State dimensions
        self.n_cs_n = self.Nr - 1
        self.n_ce = self.Nx - 3
        self.n_cs_p = self.Nr - 1
        self.state_dim = self.n_cs_n + self.n_ce + self.n_cs_p

        # Network IO
        self.input_size = 2
        self.output_size = self.state_dim

        # Initial lithium inventory in electrolyte (uniform concentration)
        self.nLi_e = self.A_e * self.c_e * (
            self.epsilon_e_n * self.L_n + self.epsilon_e_s * self.L_s + self.epsilon_e_p * self.L_p
        )
        self.nLi_e_init = self.nLi_e if self.nLi_e_init_cfg is None else float(self.nLi_e_init_cfg)

        self.C_ce = build_C_ce(self)

def build_C_ce(p, dtype=torch.float64, device=None):
    device = device or torch.device("cpu")

    Del_xn = p.L_n * p.delta_x_n
    Del_xs = p.L_s * p.delta_x_s
    Del_xp = p.L_p * p.delta_x_p

    N1 = torch.zeros((4, p.Nx - 3), dtype=dtype, device=device)
    N2 = torch.zeros((4, 4), dtype=dtype, device=device)

    # BC1
    N1[0, 0] = 4.0
    N1[0, 1] = -1.0
    N2[0, 0] = -3.0

    # BC2
    N1[1, p.Nxn - 3] = (p.epsilon_e_n ** p.brug) / (2 * Del_xn)
    N1[1, p.Nxn - 2] = (-4 * p.epsilon_e_n ** p.brug) / (2 * Del_xn)
    N2[1, 1] = (
        (3 * p.epsilon_e_n ** p.brug) / (2 * Del_xn)
        + (3 * p.epsilon_e_s ** p.brug) / (2 * Del_xs)
    )
    N1[1, p.Nxn - 1] = (-4 * p.epsilon_e_s ** p.brug) / (2 * Del_xs)
    N1[1, p.Nxn] = (p.epsilon_e_s ** p.brug) / (2 * Del_xs)

    # BC3
    N1[2, p.Nxn + p.Nxs - 4] = (p.epsilon_e_s ** p.brug) / (2 * Del_xs)
    N1[2, p.Nxn + p.Nxs - 3] = (-4 * p.epsilon_e_s ** p.brug) / (2 * Del_xs)
    N2[2, 2] = (
        (3 * p.epsilon_e_s ** p.brug) / (2 * Del_xs)
        + (3 * p.epsilon_e_p ** p.brug) / (2 * Del_xp)
    )
    N1[2, p.Nxn + p.Nxs - 2] = (-4 * p.epsilon_e_p ** p.brug) / (2 * Del_xp)
    N1[2, p.Nxn + p.Nxs - 1] = (p.epsilon_e_p ** p.brug) / (2 * Del_xp)

    # BC4
    N1[3, -2] = 1.0
    N1[3, -1] = -4.0
    N2[3, 3] = 3.0

    return -torch.linalg.solve(N2, N1)
