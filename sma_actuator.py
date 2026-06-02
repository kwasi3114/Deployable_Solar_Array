"""
step3_sma_actuator.py
---------------------
Step 3: Antagonistic dual-wire SMA actuator model for the SMPC boom.

Adapted from the author's unimorph model (sma_dynamic_model_v2_2.ipynb).

WHAT CHANGED FROM THE ORIGINAL AND WHY
---------------------------------------
Original model was designed for a single SMA wire bending a cantilever
in air. Four things are fundamentally different here:

1. TWO ANTAGONISTIC WIRES
   The boom has one SMA wire bonded to the top flange and one to the
   bottom. Top wire activated → boom bends up. Bottom wire activated →
   boom bends down. The RL agent commands both independently via
   I_top and I_bot. When one is active the other cools and goes slack.

2. THE BOOM RESISTS WITH EI, NOT THE WIRE
   In the original, the wire and beam were the same structure. Here the
   SMPC boom has its own bending stiffness EI(T_boom) from Step 1. The
   wire generates a MOMENT M = F × d_arm about the boom neutral axis.
   The boom tip angle is θ = M × L / EI (Euler-Bernoulli cantilever).
   When the boom is Joule-heated and soft (EI small), the same SMA force
   produces a much larger angle. This is the stiffness-control coupling.

3. RADIATIVE COOLING, NOT CONVECTION
   Original: Q_cool = hc × π × d × L × (T - Te)   [convection in air]
   This:     Q_cool = σ_sb × ε × π × d × L × (T⁴ - T_space⁴)  [radiation]
   In vacuum there is no convective medium. The wire can only lose heat
   by radiating to deep space (4 K background). This makes cooling much
   slower than in air, which matters for how the RL agent plans commands.

4. INITIAL CONDITIONS AND CURRENT RANGE
   The original used v=0.1 m/s and large initial strain/stress.
   Here we properly initialize: wire is fully martensitic (xi=1),
   prestrained to e_t with a small pretension (20 MPa), T_dot=0, v=0.
   Current range: 1–2 A gives equilibrium 77–230°C in vacuum,
   which spans the T_as = 70°C transformation temperature comfortably.

PHYSICS CHAIN (per wire, per timestep)
---------------------------------------
  I(t)  →  T_wire(t)        via radiative thermal model
        →  ξ(T_wire)         phase fraction: 1=martensite, 0=austenite
        →  σ, ε              constitutive model (Brinson-type)
        →  F = σ × A_wire    recovery force [N]
        →  M = F × d_arm     moment about boom neutral axis [N·m]
        →  θ = M × L / EI    tip angle [rad] from Euler-Bernoulli
  Net:   θ_net = θ_top − θ_bot  (positive = upward bend)

INTERFACES FOR RL GYMNASIUM ENVIRONMENT
-----------------------------------------
  Actions:     [I_top [A], I_bot [A]]   (RL outputs these)
  Observations:[θ_net [°], T_top [°C], T_bot [°C],
                ξ_top, ξ_bot, EI_boom [N·m²]]
  Reward:      cos(θ_sun − θ_net)  −  λ_E × power_consumed

References:
  Author's sma_dynamic_model_v2_2.ipynb
  Brinson (1993) — One-dimensional constitutive behavior of SMAs
  Lagoudas (2008) — Shape Memory Alloys: Modeling and Engineering
  Liu et al. (2023) — SMPC boom geometry
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

# ── Physical constants ─────────────────────────────────────────────────────
SIGMA_SB  = 5.670374419e-8   # W/(m²·K⁴)   Stefan-Boltzmann
T_SPACE_K = 4.0              # K            deep-space background

# ── SMA material parameters (Nitinol) ─────────────────────────────────────
# Elastic moduli match original model exactly (GPa → Pa conversion applied)
SMA_PARAMS = {
    "Ea":         80.9e9,    # Pa    austenite modulus   (original: 80.9 GPa)
    "Em":         44.6e9,    # Pa    martensite modulus  (original: 44.6 GPa)
    "e_t":        0.044,     # —     max transformation strain (original: 0.044)
    "rho":        6450.0,    # kg/m³ density
    "cp":          836.0,    # J/(kg·K) specific heat
    "deltaH":    20800.0,    # J/kg  latent heat of transformation
    # Electrical resistivity (phase + temperature dependent)
    # Units: Ω·m. Nitinol: ~70-80 μΩ·cm = 70-80e-8 Ω·m
    "rho_eA":    70.7e-8,    # Ω·m   austenite reference resistivity
    "rho_eM":    76.9e-8,    # Ω·m   martensite reference resistivity
    "mu_eA":     0.034e-8,   # Ω·m/°C  austenite temp coefficient
    "mu_eM":     0.134e-8,   # Ω·m/°C  martensite temp coefficient
    "emissivity": 0.35,      # —     surface emissivity (bare Nitinol)
    # Phase transformation temperatures (original sigmoid approach)
    "T_as":       70.0,      # °C   austenite start (heating path)
    "k_ma":        0.25,     # 1/°C sigmoid steepness  (original: 0.25)
    "T_ms":       45.0,      # °C   martensite start (cooling path)
    "k_am":        0.15,     # 1/°C sigmoid steepness  (original: 0.15)
}

# ── Boom / wire geometry ───────────────────────────────────────────────────
BOOM_PARAMS = {
    "L_boom":    1.5,       # m   deployed boom length
    "d_arm":     0.00866,   # m   lever arm = lenticular sag from Step 1a (8.66 mm)
    "wire_diam": 0.001,     # m   SMA wire diameter (1 mm)
}


# ============================================================================
# 1. Phase fraction  (unchanged from original model)
# ============================================================================

def phase_fraction(T, p=SMA_PARAMS):
    """
    Returns martensite fraction ξ on both transformation paths.

    xi_ma : heating path (martensite → austenite)
            Goes from 1.0 → 0.0 as T increases past T_as.
            ξ = 0 means fully austenitic: wire is short and stiff.

    xi_am : cooling path (austenite → martensite)
            Goes from 0.0 → 1.0 as T drops below T_ms.
            ξ = 1 means fully martensitic: wire is long and compliant.

    Sigmoid form identical to original model.
    """
    xi_ma = ((-1.0) / (1.0 + np.exp(-p["k_ma"] * (T - p["T_as"])))) + 1.0
    xi_am = ((-1.0) / (1.0 + np.exp(-p["k_am"] * (T - p["T_ms"])))) + 1.0
    return float(xi_ma), float(xi_am)


def phase_fraction_dot(T_dot, p=SMA_PARAMS):
    """dξ/dt for both paths. Same 1/80 scaling as original model."""
    return (1.0 / 80.0) * T_dot, (1.0 / 80.0) * T_dot


# ============================================================================
# 2. Constitutive model  (directly from original, units corrected to Pa)
# ============================================================================

def constitutive_model(epsilon, xi, xi_dot, v, p=SMA_PARAMS):
    """
    Stress rate σ̇ and strain rate ε̇ for one SMA wire.

    Identical to original model:
      σ̇ = E·ε̇ + Ė·ε − E·e_t·ξ̇ − Ė·e_t·ξ

    Changes from original:
      - E values in Pa (not GPa) for SI consistency
      - L uses boom length (wire runs along the boom axis)
      - v is wire endpoint velocity from boom tip rotation

    Parameters
    ----------
    epsilon  : float  axial strain in wire
    xi       : float  martensite fraction (0–1)
    xi_dot   : float  dξ/dt
    v        : float  wire endpoint velocity [m/s]
    """
    Ea  = p["Ea"];  Em  = p["Em"];  e_t = p["e_t"]
    L   = BOOM_PARAMS["L_boom"]
    E   = Ea * (1.0 - xi) + Em * xi

    # Strain rate: zero if no phase change, kinematic otherwise
    epsilon_dot = 0.0 if xi_dot == 0 else v / L

    # Rate of change of elastic modulus (same sign logic as original)
    if xi_dot > 0:
        E_dot = (Ea - Em) * xi_dot
    else:
        E_dot = -1.0 * (Em - Ea) * xi_dot

    sigma_dot = (E * epsilon_dot
                 + E_dot * epsilon
                 - E * e_t * xi_dot
                 - E_dot * e_t * xi)

    return sigma_dot, epsilon_dot


# ============================================================================
# 3. Thermal model  (RADIATIVE — replaces convection from original)
# ============================================================================

def thermal_model_wire(sigma, sigma_dot, epsilon_dot, xi, xi_dot, T, I,
                        wire_diam, p=SMA_PARAMS):
    """
    dT/dt for one SMA wire in vacuum (radiation cooling).

    Energy balance:
      ρ·cp·V·Ṫ = Q_joule − Q_rad + Q_latent − W_mechanical

    KEY CHANGE from original:
      Original used Q_cool = hc × π × d × L × (T − Te)  [convection]
      This uses    Q_rad  = σ_sb × ε × π × d × L × (T_K⁴ − T_space_K⁴)  [radiation]

    All other terms (Joule heating, latent heat, mechanical work)
    are structurally identical to the original thermal_model().

    Parameters
    ----------
    T         : float  wire temperature [°C]
    I         : float  current [A]
    wire_diam : float  wire diameter [m]
    """
    d      = wire_diam
    L      = BOOM_PARAMS["L_boom"]
    p_rho  = p["rho"];    cp     = p["cp"]
    deltaH = p["deltaH"]; eps_r  = p["emissivity"]

    A_cross = 0.25 * np.pi * d**2
    A_surf  = np.pi * d * L
    V       = A_cross * L
    T_K     = T + 273.15

    # Phase-dependent, temperature-dependent resistivity (original formula)
    rho_e = (xi * (p["rho_eM"] + p["mu_eM"] * T)
             + (1.0 - xi) * (p["rho_eA"] + p["mu_eA"] * T))
    R_wire = rho_e * L / A_cross

    # Energy terms
    Q_joule  = I**2 * R_wire                                          # Joule heating
    Q_rad    = SIGMA_SB * eps_r * A_surf * (T_K**4 - T_SPACE_K**4)  # Radiation loss
    Q_latent = p_rho * V * deltaH * xi_dot                           # Latent heat
    W_mech   = A_cross * L * sigma_dot + sigma * A_cross * L * epsilon_dot  # Work out

    T_dot = (Q_joule - Q_rad + Q_latent - W_mech) / (p_rho * cp * V)
    return float(T_dot)


# ============================================================================
# 4. Boom tip angle from SMA recovery moment  (NEW — not in original model)
# ============================================================================

def boom_tip_angle(sigma_top, sigma_bot, EI_boom,
                   wire_diam=None, p_boom=BOOM_PARAMS):
    """
    Net boom tip angle from antagonistic wire recovery stresses.

    F_recovery = σ × A_wire        [N]   only tensile — wires can't push
    M          = F × d_arm         [N·m] moment about boom neutral axis
    θ          = M × L / EI        [rad] Euler-Bernoulli cantilever

    θ_net = θ_top − θ_bot
      positive = boom bent upward  (top wire pulled it)
      negative = boom bent downward (bottom wire pulled it)
    """
    if wire_diam is None:
        wire_diam = p_boom["wire_diam"]
    L     = p_boom["L_boom"]
    d_arm = p_boom["d_arm"]
    A_w   = 0.25 * np.pi * wire_diam**2
    EI    = max(EI_boom, 1e-10)   # guard against full softening

    # Wires are tension-only: negative stress means slack wire → no force
    F_top = max(sigma_top, 0.0) * A_w
    F_bot = max(sigma_bot, 0.0) * A_w

    theta_top = np.degrees((F_top * d_arm * L) / EI)   # [°]
    theta_bot = np.degrees((F_bot * d_arm * L) / EI)   # [°]

    return theta_top - theta_bot, theta_top, theta_bot


# ============================================================================
# 5. Single wire state — one timestep
# ============================================================================

def step_wire(state, I_current, dt, EI_boom, wire_diam=None, p=SMA_PARAMS):
    """
    Advance one SMA wire by dt seconds.

    State dict keys: T, sigma, epsilon, xi, T_dot, sigma_dot, epsilon_dot, v
    Returns updated state dict.
    """
    if wire_diam is None:
        wire_diam = BOOM_PARAMS["wire_diam"]

    T       = state["T"]
    sigma   = state["sigma"]
    epsilon = state["epsilon"]
    T_dot   = state["T_dot"]
    v       = state["v"]

    # Phase fraction at current T (same logic as original main loop)
    xi_ma, xi_am = phase_fraction(T, p)
    xi_ma_dot, xi_am_dot = phase_fraction_dot(T_dot, p)

    if T_dot > 1e-6:        # heating → martensite → austenite path
        xi     = xi_ma
        xi_dot = xi_ma_dot
    elif T_dot < -1e-6:     # cooling → austenite → martensite path
        xi     = xi_am
        xi_dot = xi_am_dot
    else:                   # isothermal → no transformation
        xi     = xi_ma
        xi_dot = 0.0

    # Update thermal model
    sigma_dot, epsilon_dot = constitutive_model(epsilon, xi, xi_dot, v, p)
    T_dot_new = thermal_model_wire(sigma, sigma_dot, epsilon_dot,
                                   xi, xi_dot, T, I_current, wire_diam, p)

    # Euler-forward integration (same as original get_temperature/stress/strain)
    T_new       = T       + dt * T_dot_new
    sigma_new   = sigma   + dt * sigma_dot
    epsilon_new = epsilon + dt * epsilon_dot

    # Physical bounds
    sigma_new   = max(sigma_new, 0.0)   # wire can't push
    epsilon_new = max(epsilon_new, 0.0) # wire can't go into compression

    return {
        "T":           T_new,
        "sigma":       sigma_new,
        "epsilon":     epsilon_new,
        "xi":          xi,
        "T_dot":       T_dot_new,
        "sigma_dot":   sigma_dot,
        "epsilon_dot": epsilon_dot,
        "v":           v,
    }


# ============================================================================
# 6. Full dual-wire simulation
# ============================================================================

def _to_callable(schedule, dt):
    """Convert array or callable to a callable of time t."""
    if callable(schedule):
        return schedule
    arr = np.asarray(schedule, dtype=float)
    return lambda t: float(arr[min(int(round(t / dt)), len(arr) - 1)])


def initial_wire_state(T_init=21.0, pretension_MPa=20.0, p=SMA_PARAMS):
    """
    Correct initial conditions for a wire sitting at rest at room temperature.

    At room temp the wire is fully martensitic (ξ=1).
    The initial strain is the transformation strain e_t plus a small
    prestrain from the pretension load. This ensures sigma > 0 from the start
    (wires must be pretensioned on the boom to remain taut when cold).
    """
    xi_init      = 1.0
    epsilon_init = p["e_t"] * xi_init + (pretension_MPa * 1e6) / p["Em"]
    sigma_init   = pretension_MPa * 1e6   # Pa
    return {
        "T":           T_init,
        "sigma":       sigma_init,
        "epsilon":     epsilon_init,
        "xi":          xi_init,
        "T_dot":       0.0,
        "sigma_dot":   0.0,
        "epsilon_dot": 0.0,
        "v":           0.0,
    }


def simulate(I_top_schedule, I_bot_schedule,
             EI_boom_fn=None,
             dt=0.1, t_end=60.0,
             wire_diam=None,
             T_init=21.0,
             pretension_MPa=20.0,
             p=SMA_PARAMS,
             p_boom=BOOM_PARAMS):
    """
    Simulate both SMA wires and boom tip angle over time.

    Parameters
    ----------
    I_top_schedule : callable(t)->float  OR  array
        Current in top wire [A]. Positive only (RL action).
    I_bot_schedule : callable(t)->float  OR  array
        Current in bottom wire [A]. Positive only.
    EI_boom_fn : callable(t)->float, optional
        Boom bending stiffness [N·m²] as a function of time.
        If None, uses EI at room temperature from Step 1.
    dt     : float  timestep [s]
    t_end  : float  end time [s]
    T_init : float  initial wire temperature [°C]
    pretension_MPa : float  wire pretension [MPa] — keeps wires taut when cold

    Returns
    -------
    hist : dict of numpy arrays (all time-series, length = n_steps)
        Keys: t, T_top, T_bot, sigma_top [MPa], sigma_bot [MPa],
              xi_top, xi_bot, theta_net [°], theta_top [°], theta_bot [°],
              EI_boom, I_top, I_bot, F_top [N], F_bot [N], power [W]
    """
    if wire_diam is None:
        wire_diam = p_boom.get("wire_diam", 0.001)

    # Default EI: stiff boom at room temperature (Step 1 default)
    if EI_boom_fn is None:
        try:
            from smpc_constitutive import SMPCModel
            EI_default = SMPCModel().EI(25.0)
        except ImportError:
            EI_default = 1.59  # N·m²
        EI_boom_fn = lambda t: EI_default

    n_steps  = int(t_end / dt)
    t_arr    = np.arange(n_steps) * dt
    I_top_fn = _to_callable(I_top_schedule, dt)
    I_bot_fn = _to_callable(I_bot_schedule, dt)

    state_top = initial_wire_state(T_init, pretension_MPa, p)
    state_bot = initial_wire_state(T_init, pretension_MPa, p)

    A_w = 0.25 * np.pi * wire_diam**2
    hist = {k: np.zeros(n_steps) for k in
            ["t","T_top","T_bot","sigma_top","sigma_bot","xi_top","xi_bot",
             "theta_net","theta_top","theta_bot","EI_boom",
             "I_top","I_bot","F_top","F_bot","power"]}

    theta_prev = 0.0
    for i, t in enumerate(t_arr):
        I_top = max(I_top_fn(t), 0.0)
        I_bot = max(I_bot_fn(t), 0.0)
        EI    = EI_boom_fn(t)

        # Wire endpoint velocity from previous boom angle rate
        # (feedback from boom motion back into constitutive model — same role as
        #  the v variable in the original model)
        dtheta_dt = 0.0 if i == 0 else (hist["theta_net"][i-1] - theta_prev) / dt
        v_tip = np.radians(dtheta_dt) * p_boom["L_boom"]
        state_top["v"] =  v_tip
        state_bot["v"] = -v_tip   # antagonistic: one extends as the other shortens

        state_top = step_wire(state_top, I_top, dt, EI, wire_diam, p)
        state_bot = step_wire(state_bot, I_bot, dt, EI, wire_diam, p)

        theta_net, theta_top, theta_bot = boom_tip_angle(
            state_top["sigma"], state_bot["sigma"], EI, wire_diam, p_boom
        )

        # Estimate electrical power consumed (for RL energy penalty)
        rho_e_top = (state_top["xi"] * (p["rho_eM"] + p["mu_eM"] * state_top["T"])
                     + (1 - state_top["xi"]) * (p["rho_eA"] + p["mu_eA"] * state_top["T"]))
        rho_e_bot = (state_bot["xi"] * (p["rho_eM"] + p["mu_eM"] * state_bot["T"])
                     + (1 - state_bot["xi"]) * (p["rho_eA"] + p["mu_eA"] * state_bot["T"]))
        R_top = rho_e_top * p_boom["L_boom"] / A_w
        R_bot = rho_e_bot * p_boom["L_boom"] / A_w

        hist["t"][i]         = t
        hist["T_top"][i]     = state_top["T"]
        hist["T_bot"][i]     = state_bot["T"]
        hist["sigma_top"][i] = state_top["sigma"] / 1e6    # MPa
        hist["sigma_bot"][i] = state_bot["sigma"] / 1e6
        hist["xi_top"][i]    = state_top["xi"]
        hist["xi_bot"][i]    = state_bot["xi"]
        hist["theta_net"][i] = theta_net
        hist["theta_top"][i] = theta_top
        hist["theta_bot"][i] = theta_bot
        hist["EI_boom"][i]   = EI
        hist["I_top"][i]     = I_top
        hist["I_bot"][i]     = I_bot
        hist["F_top"][i]     = max(state_top["sigma"], 0.0) * A_w
        hist["F_bot"][i]     = max(state_bot["sigma"], 0.0) * A_w
        hist["power"][i]     = I_top**2 * R_top + I_bot**2 * R_bot

        theta_prev = theta_net

    return hist


# ============================================================================
# 7. Plotting
# ============================================================================

def plot_simulation(hist, title="SMA Dual-Wire Boom Simulation", save_path=None):
    """
    Six-panel plot covering all key state variables.
    Layout matches the variable structure of the original model (T, σ, ε)
    extended for the dual-wire boom application.
    """
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(title, fontsize=13, fontweight='bold')
    gs  = gridspec.GridSpec(3, 3, hspace=0.50, wspace=0.38)
    t   = hist["t"]

    def shade(ax, I_arr, color, alpha=0.10):
        """Shade time windows when a wire is active."""
        in_b = False
        for i, a in enumerate(I_arr > 0):
            if a and not in_b:  x0 = t[i]; in_b = True
            elif not a and in_b: ax.axvspan(x0, t[i], color=color, alpha=alpha); in_b = False
        if in_b: ax.axvspan(x0, t[-1], color=color, alpha=alpha)

    # ── [0,0]  Wire temperatures ──────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t, hist["T_top"], color='tomato',    lw=2, label="T top wire")
    ax.plot(t, hist["T_bot"], color='steelblue', lw=2, label="T bot wire")
    ax.axhline(SMA_PARAMS["T_as"], ls='--', color='tomato',    lw=1,
               alpha=0.7, label=f"T_as = {SMA_PARAMS['T_as']}°C (austenite start)")
    ax.axhline(SMA_PARAMS["T_ms"], ls='--', color='steelblue', lw=1,
               alpha=0.7, label=f"T_ms = {SMA_PARAMS['T_ms']}°C (martensite start)")
    shade(ax, hist["I_top"], 'tomato');  shade(ax, hist["I_bot"], 'steelblue')
    ax.set_xlabel("Time [s]"); ax.set_ylabel("T [°C]")
    ax.set_title("Wire temperatures"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # ── [0,1]  Martensite fraction ξ ──────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t, hist["xi_top"], color='tomato',    lw=2, label="ξ top")
    ax.plot(t, hist["xi_bot"], color='steelblue', lw=2, label="ξ bot")
    ax.axhline(0.0, ls=':', color='gray', lw=1, label="ξ=0 fully austenite")
    ax.axhline(1.0, ls=':', color='gray', lw=1, label="ξ=1 fully martensite")
    ax.set_ylim(-0.05, 1.10); ax.set_xlabel("Time [s]"); ax.set_ylabel("ξ  [—]")
    ax.set_title("Martensite fraction\n(1=cold/slack, 0=hot/contracted)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # ── [0,2]  Current commands (RL actions) ──────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    ax.step(t, hist["I_top"], color='tomato',    lw=2, where='post', label="I_top [A]")
    ax.step(t, hist["I_bot"], color='steelblue', lw=2, where='post', label="I_bot [A]")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Current [A]")
    ax.set_title("Current commands\n(RL agent actions)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── [1,0]  Wire stress ────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t, hist["sigma_top"], color='tomato',    lw=2, label="σ top [MPa]")
    ax.plot(t, hist["sigma_bot"], color='steelblue', lw=2, label="σ bot [MPa]")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("σ [MPa]")
    ax.set_title("Wire stress\n(drives recovery force)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── [1,1]  Recovery force ─────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t, hist["F_top"] * 1e3, color='tomato',    lw=2, label="F top [mN]")
    ax.plot(t, hist["F_bot"] * 1e3, color='steelblue', lw=2, label="F bot [mN]")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Force [mN]")
    ax.set_title("Wire recovery forces"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── [1,2]  Boom stiffness EI (from Step 1/2) ─────────────────────
    ax = fig.add_subplot(gs[1, 2])
    ax.semilogy(t, hist["EI_boom"], color='mediumseagreen', lw=2)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("EI [N·m²]")
    ax.set_title("Boom bending stiffness EI(t)\n(from SMPC Joule heating — Step 2)")
    ax.grid(alpha=0.3, which='both')

    # ── [2, 0:2]  Boom tip angle — THE KEY OUTPUT ─────────────────────
    ax = fig.add_subplot(gs[2, :2])
    ax.plot(t, hist["theta_net"], color='darkviolet', lw=2.5, label="θ net — boom tip [°]")
    ax.plot(t, hist["theta_top"], color='tomato',     lw=1.5, ls='--', alpha=0.7, label="θ from top wire")
    ax.plot(t, hist["theta_bot"], color='steelblue',  lw=1.5, ls='--', alpha=0.7, label="θ from bot wire")
    ax.axhline(0, color='gray', lw=1, ls=':')
    shade(ax, hist["I_top"], 'tomato'); shade(ax, hist["I_bot"], 'steelblue')
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Angle [°]")
    ax.set_title("Boom tip angle  θ(t)  — solar panel pointing direction",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # ── [2,2]  Phase portrait σ vs ξ (diagnostic, from original model) ─
    ax = fig.add_subplot(gs[2, 2])
    ax.plot(hist["xi_top"], hist["sigma_top"], color='tomato',    lw=1.5, label="Top wire")
    ax.plot(hist["xi_bot"], hist["sigma_bot"], color='steelblue', lw=1.5, label="Bot wire")
    ax.scatter([hist["xi_top"][0]], [hist["sigma_top"][0]], color='tomato',    s=50, zorder=5)
    ax.scatter([hist["xi_bot"][0]], [hist["sigma_bot"][0]], color='steelblue', s=50, zorder=5)
    ax.set_xlabel("Martensite fraction ξ"); ax.set_ylabel("Stress σ [MPa]")
    ax.set_title("Phase portrait: σ vs ξ\n(shape memory hysteresis)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved → {save_path}")
    return fig


# ============================================================================
# 8. Demonstration scenarios
# ============================================================================

if __name__ == "__main__":
    OUTPUT = "/mnt/user-data/outputs"
    os.makedirs(OUTPUT, exist_ok=True)

    try:
        from smpc_constitutive import SMPCModel
        smpc = SMPCModel()
        EI_stiff = smpc.EI(25.0)
        EI_soft  = smpc.EI(80.0)
        print(f"SMPC model: EI(25°C)={EI_stiff:.3f} N·m²  EI(80°C)={EI_soft:.4f} N·m²")
    except ImportError:
        EI_stiff, EI_soft, smpc = 1.59, 0.12, None

    # ── Scenario A: Bend up — top wire only, fixed stiff boom ──────────
    print("\n── Scenario A: Top wire on (1.5 A), fixed stiff boom ──")
    I_top_A = lambda t: 1.5 if t < 40.0 else 0.0
    hist_A = simulate(I_top_A, lambda t: 0.0,
                      EI_boom_fn=lambda t: EI_stiff,
                      dt=0.1, t_end=80.0)
    fig_A = plot_simulation(hist_A,
        title="Scenario A: Top wire activated (1.5 A) — boom bends UP\nBoom at room temperature (stiff)",
        save_path=os.path.join(OUTPUT, "step3A_bend_up.png"))
    plt.close(fig_A)
    print(f"  Peak θ upward:    {hist_A['theta_net'].max():.3f}°")
    print(f"  Peak wire T:      {hist_A['T_top'].max():.1f}°C")
    print(f"  Peak σ top:       {hist_A['sigma_top'].max():.1f} MPa")

    # ── Scenario B: Antagonistic sequence ──────────────────────────────
    print("\n── Scenario B: Top then bottom (antagonistic sequence) ──")
    I_top_B = lambda t: 1.5 if t < 35.0 else 0.0
    I_bot_B = lambda t: 0.0 if t < 40.0 else 1.5
    hist_B = simulate(I_top_B, I_bot_B,
                      EI_boom_fn=lambda t: EI_stiff,
                      dt=0.1, t_end=100.0)
    fig_B = plot_simulation(hist_B,
        title="Scenario B: Antagonistic sequence — up then down",
        save_path=os.path.join(OUTPUT, "step3B_antagonistic.png"))
    plt.close(fig_B)
    print(f"  Peak angle up:    {hist_B['theta_net'].max():.3f}°")
    print(f"  Peak angle down:  {hist_B['theta_net'].min():.3f}°")

    # ── Scenario C: Stiffness coupling — same SMA command, soft vs stiff boom ─
    print("\n── Scenario C: Effect of SMPC Joule heating on deflection ──")
    I_top_C = lambda t: 1.5 if t < 40.0 else 0.0

    hist_stiff = simulate(I_top_C, lambda t: 0.0,
                          EI_boom_fn=lambda t: EI_stiff, dt=0.1, t_end=80.0)
    hist_soft  = simulate(I_top_C, lambda t: 0.0,
                          EI_boom_fn=lambda t: EI_soft,  dt=0.1, t_end=80.0)

    fig_C, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig_C.suptitle(
        "Scenario C: SMPC stiffness coupling\n"
        "Same SMA current (1.5 A) — stiff boom (25°C) vs soft boom (80°C)",
        fontsize=12, fontweight='bold')
    t_c = hist_stiff["t"]
    axes[0].plot(t_c, hist_stiff["theta_net"], 'steelblue', lw=2,
                 label=f"Stiff: EI={EI_stiff:.3f} N·m²  (T_boom=25°C)")
    axes[0].plot(t_c, hist_soft["theta_net"],  'tomato',    lw=2,
                 label=f"Soft:  EI={EI_soft:.4f} N·m²  (T_boom=80°C)")
    axes[0].set_xlabel("Time [s]"); axes[0].set_ylabel("θ tip [°]")
    axes[0].set_title("Tip angle: stiff vs soft boom")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    axes[1].plot(t_c, hist_stiff["T_top"], 'steelblue', lw=2, label="T_wire (stiff)")
    axes[1].plot(t_c, hist_soft["T_top"],  'tomato',    lw=2, label="T_wire (soft)")
    axes[1].axhline(SMA_PARAMS["T_as"], ls='--', color='gray', lw=1,
                    label=f"T_as = {SMA_PARAMS['T_as']}°C")
    axes[1].set_xlabel("Time [s]"); axes[1].set_ylabel("Wire T [°C]")
    axes[1].set_title("Wire temperature (nearly identical)")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
    fig_C.tight_layout()
    fig_C.savefig(os.path.join(OUTPUT, "step3C_EI_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close(fig_C)

    ratio = hist_soft["theta_net"].max() / max(hist_stiff["theta_net"].max(), 1e-9)
    print(f"  EI ratio (stiff/soft): {EI_stiff/EI_soft:.0f}×")
    print(f"  Angle amplification:   {ratio:.0f}× more deflection when boom is soft")

    print("\n  All Step 3 outputs saved:")
    import glob
    for f in sorted(glob.glob(os.path.join(OUTPUT, "step3*.png"))):
        print(f"    {os.path.basename(f)}")

    print("\n  ── RL environment interface ──")
    print("  Actions (continuous):  [I_top [A], I_bot [A]]")
    print("  Observations:          [θ_net [°], T_top [°C], T_bot [°C],")
    print("                          ξ_top, ξ_bot, EI_boom [N·m²]]")
    print("  Reward:                cos(θ_sun − θ_net) − λ × power [W]")
