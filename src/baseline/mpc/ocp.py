import numpy as np
import casadi as ca

from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver


# -----------------------------------------------------------------------
# CasADi symbolic ODE
# -----------------------------------------------------------------------

def _ode_expr(x: ca.SX, zq: ca.SX, p: dict) -> ca.SX:
    # quarter-car ODE — state: [ζ−z_W, ż_W, z_W−z_B, ż_B, v, z_B]
    susp = x[2]   # z_W - z_B
    v_S  = x[1] - x[3]   # ż_W - ż_B

    # nonlinear bumpstop — clamp exponent to ≤30 to prevent exp overflow in SQP jacobians
    dz_fc      = susp - p['dz_cmp']
    exp_cmp    = ca.exp(ca.fmin(dz_fc * p['f2_cmp'] / p['dz_S_stat'], 30.0))
    F_cmp      = ca.fmin(
        p['k_S'] * (p['dz_S_stat'] * p['f1_cmp'] * (exp_cmp - 1.0) - dz_fc),
        p['F_ks_nlin_max'],
    )
    dz_fr      = -susp - p['dz_rbd']
    exp_rbd    = ca.exp(ca.fmin(dz_fr * p['f2_rbd'] / p['dz_S_stat'], 30.0))
    F_rbd      = ca.fmax(
        -p['k_S'] * (p['dz_S_stat'] * p['f1_rbd'] * (exp_rbd - 1.0) - dz_fr),
        -p['F_ks_nlin_max'],
    )
    F_nlin   = ca.if_else(susp >  p['dz_cmp'], F_cmp,
               ca.if_else(susp < -p['dz_rbd'], F_rbd, 0.0))
    F_spring = p['k_S'] * susp + F_nlin

    # sigmoid-blended piecewise damper — clamp exponent to [-200,200] to prevent overflow
    k    = 50.0
    w_c  = 1.0 / (1.0 + ca.exp(ca.fmin(ca.fmax(-k * v_S,              -200.0), 200.0)))
    w_hc = 1.0 / (1.0 + ca.exp(ca.fmin(ca.fmax(-k * (v_S - p['v_d']), -200.0), 200.0)))
    w_hr = 1.0 / (1.0 + ca.exp(ca.fmin(ca.fmax(-k * (-v_S - p['v_z']),-200.0), 200.0)))
    F_damp = (w_c * p['d1'] + (1.0 - w_c) * p['z1']) * v_S \
           + w_hc * (p['d2'] - p['d1']) * (v_S - p['v_d']) \
           - w_hr * (p['z2'] - p['z1']) * (-v_S - p['v_z'])

    F_tk = p['k_T'] * x[0]
    F_tc = p['c_T'] * (zq - x[1])

    return ca.vertcat(
        zq - x[1],
        (-F_spring - F_damp + F_tk + F_tc) / p['m_W'],
        x[1] - x[3],
        (F_spring + F_damp) / p['m_B'],
        0.0,    # v slot driven by the control input, not ODE
        x[3],
    )


def _zq_expr(s: ca.SX, v: ca.SX, bumps: list) -> ca.SX:
    # ζ̇(s, v) = dζ/dx · v — symbolic road velocity for given bump layout
    zd = ca.SX(0.0)
    for x0, A, L in bumps:
        dx   = s - x0
        dzdx = (A / 2.0) * (2.0 * ca.pi / L) * ca.sin(2.0 * ca.pi * dx / L)
        zd  += ca.if_else((dx > 0.0) * (dx < L), dzdx * v, 0.0)
    return zd


# -----------------------------------------------------------------------
# acados model builder
# -----------------------------------------------------------------------

def build_acados_model(physics: dict, bumps: list, dt: float) -> AcadosModel:
    model      = AcadosModel()
    model.name = 'quarter_car'

    nx = 7   # [ζ−z_W, ż_W, z_W−z_B, ż_B, v, z_B, s_pos]
    nu = 1   # normalised acceleration command u ∈ [-1, 1]
    # online parameters: v_ref (1 scalar per shooting node, passed as p)
    # we bake bumps into the symbolic expression at build time

    x  = ca.SX.sym('x',  nx)
    u  = ca.SX.sym('u',  nu)
    p  = ca.SX.sym('p',  1)    # p[0] = v_ref at this node

    # augmented state: x[0:6] = ODE state, x[6] = s_pos
    x_ode = x[:6]
    s_pos = x[6]
    v     = x[4]               # longitudinal speed slot

    # speed integration:  v_new = clip(v + u * a_max * dt, 0, v_max)
    # acados doesn't clip inside dynamics, so we rely on constraints
    v_max = physics.get('v_max', 20.0)
    a_max = physics.get('a_max',  5.0)
    v_new = v + u[0] * a_max * dt

    # road disturbance at current position
    zq = _zq_expr(s_pos, v_new, bumps)

    # ODE derivatives (use v_new for road sampling — one midpoint approx)
    zq_mid = _zq_expr(s_pos + 0.5 * dt * v_new, v_new, bumps)
    dx_ode = _ode_expr(x_ode, zq_mid, physics)

    # augmented continuous dynamics: ẋ_aug = [dx_ode; v_new] (s_pos integrates v)
    # acados uses explicit continuous-time model f(x,u,p); it discretises internally
    f_expl = ca.vertcat(dx_ode, v_new)

    # — but x[4] (speed) is a pure integrator driven by u, not the ODE.
    # Override: dx[4] = (v_new - v) / dt is the discrete update;
    # for continuous-time we write it as the commanded acceleration.
    # f_expl[4] = u[0] * a_max  (continuous speed derivative)
    f_expl_fixed = ca.vertcat(
        dx_ode[:4],
        u[0] * a_max,    # ẋ[4] = v̇ = u · a_max
        dx_ode[5],       # ẋ[5] = z_B derivative from ODE
        v_new,           # ẋ[6] = ds/dt = v_new  (arc-length rate)
    )

    model.x    = x
    model.u    = u
    model.p    = p
    model.f_expl_expr  = f_expl_fixed
    model.f_impl_expr  = ca.SX.sym('xdot', nx) - f_expl_fixed   # unused but required
    model.xdot = ca.SX.sym('xdot', nx)

    return model


# -----------------------------------------------------------------------
# full OCP solver factory
# -----------------------------------------------------------------------

def build_solver(
    physics:  dict,
    bumps:    list,
    cfg,                   # RewardConfig
    N:        int   = 50,
    dt:       float = 0.02,
    gen_dir:  str   = '/tmp/acados_qc',
) -> AcadosOcpSolver:

    model = build_acados_model(physics, bumps, dt)

    ocp              = AcadosOcp()
    ocp.model        = model
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf        = N * dt

    # --- dimensions ---
    nx, nu, np_ = 7, 1, 1
    ocp.dims.nx  = nx
    ocp.dims.nu  = nu
    ocp.dims.np  = np_

    # --- cost: external (lets us express the full reward-aligned cost) ---
    # cost = w_heave*(zBddot/aBc)^2 + w_wheel*(zWddot/aWc)^2
    #      + w_track*((vref-v)/vmax)^2 + w_accel*(a/ac)^2
    #      + w_smooth*(u - u_prev)^2 - w_prog*v/vmax
    # We use NONLINEAR_LS cost so acados can use Gauss-Newton Hessian approx.

    x_sym = model.x
    u_sym = model.u
    p_sym = model.p

    v     = x_sym[4]
    s_pos = x_sym[6]
    a_max_v = physics.get('a_max', 5.0)
    v_max_v = float(cfg.v_max)
    v_min_v = float(cfg.v_min)

    # body/wheel accel: evaluate ODE derivative at current state for z_B_ddot, z_W_ddot
    zq_cost = _zq_expr(s_pos, v, bumps)
    dx_cost = _ode_expr(x_sym[:6], zq_cost, physics)
    z_B_ddot = dx_cost[3]
    z_W_ddot = dx_cost[1]
    a_long   = u_sym[0] * a_max_v

    v_ref   = p_sym[0]
    # two-sided speed tracking: negative = too slow, positive = too fast
    # v_ref is set per-node by _v_ref_at (v_max on flat, lower near bumps)
    # so this residual provides both the "go fast" and "brake before bumps" signal
    speed_err = (v - v_ref) / v_max_v

    # residuals for NONLINEAR_LS  (cost = 0.5 * (r - yref)' W (r - yref))
    r = ca.vertcat(
        z_B_ddot / cfg.a_B_comfort,   # heave       — target 0
        z_W_ddot / cfg.a_W_comfort,   # wheel       — target 0
        speed_err,                    # speed err   — target 0 (two-sided)
        a_long    / cfg.a_comfort,    # long. accel — target 0
        u_sym[0],                     # action mag  — target 0
    )
    nr = r.shape[0]

    # terminal residual — must not depend on u; drop a_long and u_sym[0]
    r_e = ca.vertcat(
        z_B_ddot / cfg.a_B_comfort,
        z_W_ddot / cfg.a_W_comfort,
        speed_err,
    )
    nr_e = r_e.shape[0]

    yref   = np.zeros(nr)
    yref_e = np.zeros(nr_e)

    ocp.cost.cost_type    = 'NONLINEAR_LS'
    ocp.cost.cost_type_e  = 'NONLINEAR_LS'
    ocp.model.cost_y_expr   = r
    ocp.model.cost_y_expr_e = r_e

    W = np.diag([
        cfg.w_heave,
        cfg.w_wheel,
        cfg.w_tracking,
        cfg.w_accel,
        cfg.w_action_smooth,
    ])
    W_e = np.diag([
        cfg.w_heave,
        cfg.w_wheel,
        cfg.w_tracking,
    ]) * 2.0
    ocp.cost.W      = W
    ocp.cost.W_e    = W_e
    ocp.cost.yref   = yref
    ocp.cost.yref_e = yref_e

    # --- constraints ---
    # u ∈ [-1, 1]
    ocp.constraints.lbu = np.array([-1.0])
    ocp.constraints.ubu = np.array([ 1.0])
    ocp.constraints.idxbu = np.array([0])

    # v ∈ [v_min, v_max]  (state constraint on x[4])
    ocp.constraints.lbx   = np.array([v_min_v])
    ocp.constraints.ubx   = np.array([v_max_v])
    ocp.constraints.idxbx = np.array([4])

    # initial state constraint (set at runtime)
    ocp.constraints.x0 = np.zeros(nx)

    # --- initial parameter values ---
    ocp.parameter_values = np.array([v_max_v])  # v_ref placeholder

    # --- solver options ---
    ocp.solver_options.integrator_type      = 'ERK'
    ocp.solver_options.num_stages           = 4        # RK4
    ocp.solver_options.num_steps            = 4        # 4 sub-steps: k_T=262kN/m → ω≈72rad/s, need smaller h
    ocp.solver_options.nlp_solver_type      = 'SQP'
    ocp.solver_options.nlp_solver_max_iter  = 10       # 10 SQP iters — better quality, ~5ms acceptable offline
    ocp.solver_options.qp_solver            = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.qp_solver_cond_N     = min(N, 10)
    ocp.solver_options.hessian_approx       = 'GAUSS_NEWTON'
    ocp.solver_options.print_level          = 0

    ocp.code_export_directory = gen_dir

    return AcadosOcpSolver(ocp, json_file=f'{gen_dir}/acados_ocp.json')
