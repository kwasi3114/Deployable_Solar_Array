"""
step4_boom_env.py
-----------------
Step 4: Gymnasium environment for the SMPC boom solar tracking RL problem.

WHAT THIS FILE IS
-----------------
A Gymnasium-compatible environment that wraps the physics from Steps 1–3
into the standard (observation, action, reward, done) loop that any RL
algorithm can train on. Think of it as a flight simulator: the physics
models inside are the "aircraft dynamics", and this file is the cockpit
interface that translates pilot inputs (currents, power) into instrument
readings (tip angle, temperatures) and a score (solar flux captured).

THE THREE-ACTUATOR CONTROL PROBLEM
-----------------------------------
The RL agent controls three things simultaneously:

  1. I_top  [0, 2 A]  — current through top SMA wire  (bends boom UP)
  2. I_bot  [0, 2 A]  — current through bottom SMA wire (bends boom DOWN)
  3. P_joule [0, 50 W] — Joule heating power to the SMPC boom itself
                          (softens EI, amplifies SMA authority)

This is a continuous action space — the agent picks a real number in each
range every control step, not a discrete choice.

WHY THREE ACTUATORS TOGETHER IS NON-TRIVIAL
--------------------------------------------
The three actuators are coupled in a non-obvious way:

  - If you activate I_top without P_joule, the boom is stiff and barely moves.
  - If you apply P_joule first, EI drops 300×, then a small I_top moves the boom a lot.
  - But P_joule takes time (thermal lag) to soften the boom — the agent must plan ahead.
  - After bending, you want P_joule OFF so the boom re-stiffens and holds position
    without continuous SMA current (which wastes power and fatigues the wire).
  - Meanwhile the sun is moving: orbital period ~90 min, so the target angle
    changes at roughly 4°/min. The agent must track this continuously.

This temporal coupling — heat first, bend, re-stiffen, hold — is exactly what
RL is good at learning. A simple PID controller would struggle because the
optimal strategy changes depending on how soft the boom currently is.

OBSERVATION SPACE (what the agent can see)
-------------------------------------------
  [0]  theta_boom     [°]   current boom tip angle         (-90 to +90)
  [1]  theta_sun      [°]   current sun angle target        (-90 to +90)
  [2]  theta_error    [°]   sun - boom (what to minimize)   (-180 to +180)
  [3]  T_wire_top     [°C]  top SMA wire temperature        (0 to 300)
  [4]  T_wire_bot     [°C]  bottom SMA wire temperature     (0 to 300)
  [5]  xi_top         [—]   top wire martensite fraction    (0 to 1)
  [6]  xi_bot         [—]   bottom wire martensite fraction (0 to 1)
  [7]  T_boom_mean    [°C]  mean boom temperature           (0 to 150)
  [8]  EI_norm        [—]   boom stiffness, normalised      (0 to 1)
                              0 = fully soft, 1 = room-temp stiff
  [9]  t_orbital_norm [—]   fraction of orbital period elapsed (0 to 1)

ACTION SPACE (what the agent controls)
----------------------------------------
  [0]  I_top   [A]  top wire current     (0 to 2 A)
  [1]  I_bot   [A]  bottom wire current  (0 to 2 A)
  [2]  P_joule [W]  boom Joule heating   (0 to 50 W)
All continuous, clipped to their bounds before being applied.

REWARD FUNCTION
----------------
  r = r_track − λ_E × r_energy − λ_rate × r_rate − λ_temp × r_temp

  r_track  = cos(theta_error × π/180)
             1.0 when perfectly pointed, 0 when 90° off, -1 when 180° off
             This is the projected solar flux fraction collected.

  r_energy = (I_top² + I_bot²) × R_wire  +  P_joule  [normalised by max power]
             Penalises wasted electrical energy — spacecraft power budget.

  r_rate   = |Δaction| between consecutive steps
             Penalises rapid switching — SMA fatigue, thermal shock.

  r_temp   = max(0, T_wire_top - T_limit) + max(0, T_wire_bot - T_limit)
             Penalises overheating the SMA wires beyond safe operating temp.

EPISODE STRUCTURE
------------------
  - Each episode covers one full orbital period (5400 s ≈ 90 min LEO)
  - Control timestep: 5 s (the agent makes a new decision every 5 seconds)
  - The physics runs at its own sub-timestep internally (CFL-limited for
    the thermal FD solver, 0.1 s for the SMA model)
  - Sun angle profile: sinusoidal with eclipse (sun hidden for ~35% of orbit)
  - Episode terminates early if wire temperature exceeds hard limit (300°C)

HOW THE PHYSICS PIECES CONNECT
--------------------------------
                 ┌──────────────────────────────────────┐
    RL agent     │  P_joule → ThermalSolver._step()     │
    action ───►  │           → T_boom(x,t)              │
                 │           → EI(T_boom_mean)  ─────┐  │
                 │  I_top  → SMA wire top state  ─►  │  │
                 │  I_bot  → SMA wire bot state  ─►  │  │
                 │                               │   │  │
                 │     boom_tip_angle(σ_top,     │   │  │
                 │       σ_bot, EI) ◄────────────┘───┘  │
                 │           → θ_boom                    │
                 └──────────────────────────────────────┘
                           │
                    observation, reward
                           │
                         agent

References:
  Gymnasium docs — https://gymnasium.farama.org
  Stable-Baselines3 — https://stable-baselines3.readthedocs.io
  Steps 1-3 of this project (smpc_constitutive, thermal_solver, step3_sma_actuator)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys, warnings
sys.path.insert(0, os.path.dirname(__file__))

from smpc_constitutive import SMPCModel
from thermal_solver import ThermalSolver
from step3_sma_actuator import (
    SMA_PARAMS, BOOM_PARAMS,
    phase_fraction, phase_fraction_dot,
    constitutive_model, thermal_model_wire,
    boom_tip_angle, initial_wire_state,
)


# ============================================================================
# Environment configuration — all tunable knobs in one place
# ============================================================================

ENV_CONFIG = {
    # ── Timing ───────────────────────────────────────────────────────────
    "control_dt":       5.0,      # s   how often the agent makes a decision
    "orbital_period":   5400.0,   # s   LEO ~90 min
    "eclipse_fraction": 0.35,     # —   fraction of orbit in Earth's shadow

    # ── Action bounds ────────────────────────────────────────────────────
    "I_max":            2.0,      # A   max SMA wire current
    "P_joule_max":      50.0,     # W   max boom Joule heating power

    # ── Sun angle model ──────────────────────────────────────────────────
    # Sun sweeps ±theta_sun_amp degrees over one orbit
    "theta_sun_amp":    45.0,     # °   amplitude of sun angle variation
    "theta_sun_offset": 0.0,      # °   mean sun elevation (0 = equatorial orbit)

    # ── Reward weights ───────────────────────────────────────────────────
    "lambda_energy":    0.05,     # weight on energy penalty
    "lambda_rate":      0.02,     # weight on action-rate penalty
    "lambda_temp":      0.10,     # weight on overtemperature penalty
    "T_wire_limit":     200.0,    # °C  SMA safe operating temperature

    # ── Episode termination ──────────────────────────────────────────────
    "T_wire_hard_limit": 300.0,   # °C  hard cutoff — terminates episode

    # ── Physics sub-stepping ─────────────────────────────────────────────
    "sma_dt":           0.5,      # s   SMA integration sub-timestep
    "boom_thermal_dt":  1.0,      # s   boom thermal sub-timestep (≤ max_stable_dt)
    "n_boom_nodes":     30,       # —   spatial nodes in boom thermal FD solver
                                  #     (30 is fast; increase to 60 for accuracy)

    # ── Normalisation references (for EI_norm observation) ───────────────
    "EI_ref_stiff":     None,     # N·m²  set automatically from SMPCModel
    "EI_ref_soft":      None,     # N·m²  set automatically from SMPCModel
}


# ============================================================================
# Sun angle model
# ============================================================================

def sun_angle(t, config=ENV_CONFIG):
    """
    Sun angle as seen by the boom [degrees] as a function of time [s].

    Models a simple sinusoidal variation over one orbital period, with
    eclipse (sun angle set to None / flag) during the shadow portion.

    Returns
    -------
    angle : float or None
        Angle in degrees, or None if in eclipse.
    in_eclipse : bool
    """
    T_orb   = config["orbital_period"]
    amp     = config["theta_sun_amp"]
    offset  = config["theta_sun_offset"]
    e_frac  = config["eclipse_fraction"]

    phase   = (t % T_orb) / T_orb          # 0 → 1 over one orbit
    angle   = offset + amp * np.sin(2 * np.pi * phase)

    # Eclipse: treat the last e_frac of each orbit as shadow
    in_eclipse = phase > (1.0 - e_frac)

    return float(angle), bool(in_eclipse)


# ============================================================================
# Stateful SMA wire — wraps step3 logic into a step-able object
# ============================================================================

class SMAWireState:
    """
    Holds the state of one SMA wire and advances it by a fixed sub-timestep.
    Wraps the functions from step3_sma_actuator.py.
    """

    def __init__(self, wire_diam=None, T_init=21.0, pretension_MPa=20.0,
                 p=SMA_PARAMS, p_boom=BOOM_PARAMS):
        self.p      = p
        self.p_boom = p_boom
        self.d      = wire_diam or p_boom["wire_diam"]
        self.A      = 0.25 * np.pi * self.d**2
        self.state  = initial_wire_state(T_init, pretension_MPa, p)

    def step(self, I_current, dt, v_tip=0.0):
        """Advance wire by dt seconds at current I_current [A]."""
        self.state["v"] = v_tip

        T       = self.state["T"]
        sigma   = self.state["sigma"]
        epsilon = self.state["epsilon"]
        T_dot   = self.state["T_dot"]

        # Phase fraction
        xi_ma, xi_am = phase_fraction(T, self.p)
        xi_ma_dot, xi_am_dot = phase_fraction_dot(T_dot, self.p)
        if T_dot > 1e-6:
            xi, xi_dot = xi_ma, xi_ma_dot
        elif T_dot < -1e-6:
            xi, xi_dot = xi_am, xi_am_dot
        else:
            xi, xi_dot = xi_ma, 0.0

        # Constitutive
        sigma_dot, epsilon_dot = constitutive_model(epsilon, xi, xi_dot,
                                                     v_tip, self.p)

        # Thermal (radiative)
        # W_mech = σ × A × v_tip  (external power output only — avoids σ_dot feedback)
        W_mech  = sigma * self.A * abs(v_tip)
        rho_e   = (xi * (self.p["rho_eM"] + self.p["mu_eM"] * T)
                   + (1 - xi) * (self.p["rho_eA"] + self.p["mu_eA"] * T))
        R_wire  = rho_e * self.p_boom["L_boom"] / self.A
        from step3_sma_actuator import SIGMA_SB, T_SPACE_K
        A_surf  = np.pi * self.d * self.p_boom["L_boom"]
        T_K     = T + 273.15
        Q_j     = I_current**2 * R_wire
        Q_rad   = SIGMA_SB * self.p["emissivity"] * A_surf * (
                      np.clip(T_K, 0, 3000)**4 - T_SPACE_K**4)
        Q_lat   = self.p["rho"] * self.A * self.p_boom["L_boom"] * self.p["deltaH"] * xi_dot
        thermal_mass = self.p["rho"] * self.p["cp"] * self.A * self.p_boom["L_boom"]
        T_dot_new = (Q_j - Q_rad + Q_lat - W_mech) / thermal_mass

        # Clamp T_dot to prevent runaway (physics limiter, not model fudge)
        T_dot_new = float(np.clip(T_dot_new, -50.0, 50.0))

        # Euler forward
        self.state["T"]           = float(np.clip(T + dt * T_dot_new, -100, 3000))
        self.state["sigma"]       = float(max(sigma + dt * sigma_dot, 0.0))
        self.state["epsilon"]     = float(max(epsilon + dt * epsilon_dot, 0.0))
        self.state["xi"]          = float(xi)
        self.state["T_dot"]       = T_dot_new
        self.state["sigma_dot"]   = float(sigma_dot)
        self.state["epsilon_dot"] = float(epsilon_dot)

        return self.state.copy()

    def advance(self, I_current, total_dt, sub_dt=0.5, v_tip=0.0):
        """Advance wire by total_dt using sub_dt sub-steps."""
        n = max(1, int(round(total_dt / sub_dt)))
        actual_dt = total_dt / n
        for _ in range(n):
            self.step(I_current, actual_dt, v_tip)
        return self.state.copy()

    def reset(self, T_init=21.0, pretension_MPa=20.0):
        self.state = initial_wire_state(T_init, pretension_MPa, self.p)

    @property
    def T(self):       return self.state["T"]
    @property
    def sigma(self):   return self.state["sigma"]
    @property
    def xi(self):      return self.state["xi"]
    @property
    def F(self):       return max(self.state["sigma"], 0.0) * self.A


# ============================================================================
# Stateful boom thermal model — wraps ThermalSolver into a step-able object
# ============================================================================

class BoomThermalState:
    """
    Holds the spatial temperature field T(x) of the SMPC boom and advances
    it one control step at a time using the Step 2 finite-difference solver.
    """

    def __init__(self, smpc_model, n_nodes=30, boom_params=None,
                 sub_dt=1.0, T_init=20.0):
        self.smpc    = smpc_model
        self.sub_dt  = sub_dt
        self.T_field = np.full(n_nodes, T_init)   # spatial temp field [°C]

        # Build the thermal solver (Step 2)
        boom = boom_params or {
            "length":    BOOM_PARAMS["L_boom"],
            "width":     0.030,
            "thickness": 0.0012,
        }
        self.solver = ThermalSolver(smpc_model, boom_params=boom,
                                    n_nodes=n_nodes)
        self.T_field = np.full(n_nodes, T_init)

    def advance(self, P_joule, total_dt):
        """
        Advance the boom temperature field by total_dt seconds
        with Joule heating power P_joule [W].

        Uses internal sub-stepping at self.sub_dt for stability.
        """
        n = max(1, int(round(total_dt / self.sub_dt)))
        actual_dt = total_dt / n
        q_joule_vol = P_joule / (self.solver._A_cross * self.solver.L)
        for _ in range(n):
            self.T_field = self.solver._step(self.T_field, actual_dt, q_joule_vol)

    def reset(self, T_init=20.0):
        self.T_field = np.full(len(self.T_field), T_init)

    @property
    def T_mean(self):
        return float(np.mean(self.T_field))

    @property
    def T_tip(self):
        return float(self.T_field[-1])

    @property
    def EI(self):
        """Bending stiffness using mean boom temperature."""
        return float(self.smpc.EI(self.T_mean))

    @property
    def EI_spatial_mean(self):
        """More accurate: average EI over spatial nodes."""
        return float(np.mean(self.smpc.EI(self.T_field)))


# ============================================================================
# The Gymnasium Environment
# ============================================================================

class SMPCBoomEnv(gym.Env):
    """
    Gymnasium environment for SMPC boom solar tracking.

    The agent controls three actuators:
      - I_top   [A]  : SMA wire current (bends boom up)
      - I_bot   [A]  : SMA wire current (bends boom down)
      - P_joule [W]  : Joule heating of SMPC boom (softens stiffness)

    And observes:
      theta_boom, theta_sun, theta_error,
      T_wire_top, T_wire_bot, xi_top, xi_bot,
      T_boom_mean, EI_norm, t_orbital_norm

    The reward maximises solar flux captured (cos of pointing error)
    while penalising energy use, actuation rate, and wire overheating.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, config=None, render_mode=None):
        super().__init__()

        self.cfg  = {**ENV_CONFIG, **(config or {})}
        self.render_mode = render_mode

        # ── Build physics objects ─────────────────────────────────────
        self.smpc   = SMPCModel()
        self.boom   = BoomThermalState(
            self.smpc,
            n_nodes=self.cfg["n_boom_nodes"],
            sub_dt=self.cfg["boom_thermal_dt"],
        )
        self.wire_top = SMAWireState()
        self.wire_bot = SMAWireState()

        # ── Normalisation references ──────────────────────────────────
        EI_stiff = self.smpc.EI(25.0)
        EI_soft  = self.smpc.EI(100.0)
        self.cfg["EI_ref_stiff"] = EI_stiff
        self.cfg["EI_ref_soft"]  = EI_soft
        self._EI_range = EI_stiff - EI_soft   # for normalisation

        # Max possible energy per step (for reward normalisation)
        R_ref = 1.5    # Ω  approximate wire resistance
        self._power_max = (2 * self.cfg["I_max"]**2 * R_ref
                           + self.cfg["P_joule_max"])

        # ── Gymnasium spaces ──────────────────────────────────────────
        # Action: [I_top, I_bot, P_joule] — all normalised to [0, 1]
        # We use normalised actions so the RL algorithm doesn't need to
        # know the physical units; we scale inside step().
        self.action_space = spaces.Box(
            low=np.zeros(3, dtype=np.float32),
            high=np.ones(3, dtype=np.float32),
            dtype=np.float32,
        )

        # Observation: 10 features, all scaled to roughly [-1, 1] or [0, 1]
        obs_low  = np.array([-1, -1, -2,   0,   0, 0, 0,  0, 0, 0], dtype=np.float32)
        obs_high = np.array([ 1,  1,  2,   1,   1, 1, 1,  1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        # ── Episode state ─────────────────────────────────────────────
        self.t          = 0.0
        self.theta_boom = 0.0
        self.prev_action = np.zeros(3, dtype=np.float32)
        self._episode_reward = 0.0
        self._step_count = 0

        # Render buffer
        self._render_history = []

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Randomise episode start time within orbit for curriculum diversity
        if options and options.get("fixed_start", False):
            self.t = 0.0
        else:
            self.t = float(self.np_random.uniform(0, self.cfg["orbital_period"]))

        # Reset physics
        self.boom.reset(T_init=20.0)
        self.wire_top.reset()
        self.wire_bot.reset()

        self.theta_boom  = 0.0
        self.prev_action = np.zeros(3, dtype=np.float32)
        self._episode_reward = 0.0
        self._step_count = 0
        self._render_history = []

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action):
        action = np.clip(action, 0.0, 1.0).astype(np.float32)

        # Scale normalised actions to physical units
        I_top   = float(action[0] * self.cfg["I_max"])
        I_bot   = float(action[1] * self.cfg["I_max"])
        P_joule = float(action[2] * self.cfg["P_joule_max"])

        dt      = self.cfg["control_dt"]
        sub_sma = self.cfg["sma_dt"]

        # ── 1. Advance boom thermal state ────────────────────────────
        self.boom.advance(P_joule, dt)
        EI = self.boom.EI   # uses mean boom temperature

        # ── 2. Advance SMA wires ─────────────────────────────────────
        # Tip velocity feedback: rate of change of boom arc length at wire
        # attachment point ≈ dθ/dt × d_arm (small-angle, in m/s)
        v_tip = 0.0   # updated below after angle is known
        self.wire_top.advance(I_top, dt, sub_dt=sub_sma, v_tip= v_tip)
        self.wire_bot.advance(I_bot, dt, sub_dt=sub_sma, v_tip=-v_tip)

        # ── 3. Compute boom tip angle ─────────────────────────────────
        theta_net, _, _ = boom_tip_angle(
            self.wire_top.sigma,
            self.wire_bot.sigma,
            EI,
        )
        # Clamp to physical limits
        theta_net     = float(np.clip(theta_net, -89.9, 89.9))
        self.theta_boom = theta_net

        # ── 4. Advance time and get sun angle ─────────────────────────
        self.t += dt
        theta_sun, in_eclipse = sun_angle(self.t, self.cfg)

        # ── 5. Reward ─────────────────────────────────────────────────
        reward = self._compute_reward(
            theta_net, theta_sun, in_eclipse,
            I_top, I_bot, P_joule, action
        )
        self._episode_reward += reward
        self._step_count += 1

        # ── 6. Termination ────────────────────────────────────────────
        T_max_wire = max(self.wire_top.T, self.wire_bot.T)
        terminated = bool(T_max_wire > self.cfg["T_wire_hard_limit"])
        truncated  = bool(self.t >= self.cfg["orbital_period"] * 2)
        # One full orbital period per episode; allow up to 2× for recovery

        # Store for render
        self._render_history.append({
            "t": self.t, "theta_boom": theta_net, "theta_sun": theta_sun,
            "in_eclipse": in_eclipse, "EI": EI,
            "T_top": self.wire_top.T, "T_bot": self.wire_bot.T,
            "xi_top": self.wire_top.xi, "xi_bot": self.wire_bot.xi,
            "I_top": I_top, "I_bot": I_bot, "P_joule": P_joule,
            "reward": reward,
        })

        self.prev_action = action.copy()

        info = {
            "theta_error_deg": theta_sun - theta_net,
            "in_eclipse": in_eclipse,
            "EI": EI,
            "T_wire_max": T_max_wire,
            "episode_reward": self._episode_reward,
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self):
        """
        Build the 10-element normalised observation vector.

        All values scaled to approximately [-1, 1] or [0, 1] so the
        neural network doesn't have to deal with wildly different magnitudes.
        """
        theta_sun, _ = sun_angle(self.t, self.cfg)
        theta_err    = theta_sun - self.theta_boom
        amp          = self.cfg["theta_sun_amp"]

        # Angle features: normalised by amplitude
        obs_theta_boom = np.clip(self.theta_boom / 90.0, -1, 1)
        obs_theta_sun  = np.clip(theta_sun / 90.0,       -1, 1)
        obs_theta_err  = np.clip(theta_err / 180.0,      -1, 1)

        # Wire temperatures: normalised to [0, 1] over [0, 300°C]
        obs_T_top = np.clip(self.wire_top.T / 300.0, 0, 1)
        obs_T_bot = np.clip(self.wire_bot.T / 300.0, 0, 1)

        # Phase fraction already in [0, 1]
        obs_xi_top = float(self.wire_top.xi)
        obs_xi_bot = float(self.wire_bot.xi)

        # Boom temperature: [0, 1] over [0, 150°C]
        obs_T_boom = np.clip(self.boom.T_mean / 150.0, 0, 1)

        # EI norm: 1 = stiff (room temp), 0 = soft (100°C)
        EI_now = self.boom.EI
        obs_EI = np.clip(
            (EI_now - self.cfg["EI_ref_soft"]) / max(self._EI_range, 1e-10),
            0, 1
        )

        # Orbital phase [0, 1]
        obs_t_orb = (self.t % self.cfg["orbital_period"]) / self.cfg["orbital_period"]

        return np.array([
            obs_theta_boom, obs_theta_sun, obs_theta_err,
            obs_T_top, obs_T_bot, obs_xi_top, obs_xi_bot,
            obs_T_boom, obs_EI, obs_t_orb,
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, theta_boom, theta_sun, in_eclipse,
                         I_top, I_bot, P_joule, action):
        cfg = self.cfg

        # ── Tracking reward ──────────────────────────────────────────
        # cos(error): 1.0 = perfect, 0 = 90° off, -1 = facing away
        # During eclipse: no sun → zero tracking reward (but also zero penalty
        # for pointing error — the agent should park in a neutral position)
        if in_eclipse:
            r_track = 0.0
        else:
            err_rad = np.radians(theta_sun - theta_boom)
            r_track = float(np.cos(err_rad))

        # ── Energy penalty ────────────────────────────────────────────
        # Approximate wire resistance for power estimate
        xi_top  = self.wire_top.xi
        xi_bot  = self.wire_bot.xi
        T_top   = self.wire_top.T
        T_bot   = self.wire_bot.T
        p = SMA_PARAMS
        rho_e_top = (xi_top*(p["rho_eM"]+p["mu_eM"]*T_top)
                     + (1-xi_top)*(p["rho_eA"]+p["mu_eA"]*T_top))
        rho_e_bot = (xi_bot*(p["rho_eM"]+p["mu_eM"]*T_bot)
                     + (1-xi_bot)*(p["rho_eA"]+p["mu_eA"]*T_bot))
        A_w   = 0.25 * np.pi * BOOM_PARAMS["wire_diam"]**2
        L     = BOOM_PARAMS["L_boom"]
        R_top = rho_e_top * L / A_w
        R_bot = rho_e_bot * L / A_w
        power_sma   = I_top**2 * R_top + I_bot**2 * R_bot
        power_total = power_sma + P_joule
        r_energy = power_total / self._power_max   # [0, 1]

        # ── Action-rate penalty ───────────────────────────────────────
        r_rate = float(np.mean(np.abs(action - self.prev_action)))

        # ── Overtemperature penalty ───────────────────────────────────
        T_lim = cfg["T_wire_limit"]
        r_temp = (max(0.0, T_top - T_lim) + max(0.0, T_bot - T_lim)) / T_lim

        # ── Combine ───────────────────────────────────────────────────
        reward = (r_track
                  - cfg["lambda_energy"] * r_energy
                  - cfg["lambda_rate"]   * r_rate
                  - cfg["lambda_temp"]   * r_temp)

        return float(reward)

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self):
        if not self._render_history:
            return None

        h   = self._render_history
        t   = np.array([s["t"]          for s in h])
        th  = np.array([s["theta_boom"] for s in h])
        ts  = np.array([s["theta_sun"]  for s in h])
        ec  = np.array([s["in_eclipse"] for s in h])
        EI  = np.array([s["EI"]         for s in h])
        Tt  = np.array([s["T_top"]      for s in h])
        Tb  = np.array([s["T_bot"]      for s in h])
        xit = np.array([s["xi_top"]     for s in h])
        xib = np.array([s["xi_bot"]     for s in h])
        It  = np.array([s["I_top"]      for s in h])
        Ib  = np.array([s["I_bot"]      for s in h])
        Pj  = np.array([s["P_joule"]    for s in h])
        rw  = np.array([s["reward"]     for s in h])

        fig = plt.figure(figsize=(16, 10))
        fig.suptitle("SMPCBoomEnv — Episode Rollout", fontsize=13, fontweight="bold")
        gs  = gridspec.GridSpec(3, 3, hspace=0.50, wspace=0.35)

        def shade_eclipse(ax):
            in_e = False
            for i, e in enumerate(ec):
                if e and not in_e:  x0 = t[i]; in_e = True
                elif not e and in_e: ax.axvspan(x0, t[i], color='gray', alpha=0.15); in_e = False
            if in_e: ax.axvspan(x0, t[-1], color='gray', alpha=0.15, label='Eclipse')

        # [0,0:2] Pointing performance
        ax = fig.add_subplot(gs[0, :2])
        ax.plot(t/60, ts, 'gold',   lw=1.5, ls='--', label='Sun angle θ_sun')
        ax.plot(t/60, th, 'darkviolet', lw=2, label='Boom angle θ_boom')
        ax.fill_between(t/60, ts, th, alpha=0.15, color='red', label='Pointing error')
        shade_eclipse(ax)
        ax.set_xlabel("Time [min]"); ax.set_ylabel("Angle [°]")
        ax.set_title("Solar tracking performance — boom tip angle vs sun angle")
        ax.legend(fontsize=8, loc='upper right'); ax.grid(alpha=0.3)

        # [0,2] Cumulative reward
        ax = fig.add_subplot(gs[0, 2])
        ax.plot(t/60, np.cumsum(rw), 'steelblue', lw=2)
        ax.set_xlabel("Time [min]"); ax.set_ylabel("Cumulative reward")
        ax.set_title("Cumulative reward"); ax.grid(alpha=0.3)

        # [1,0] Actions
        ax = fig.add_subplot(gs[1, 0])
        ax.step(t/60, It, color='tomato',       lw=1.5, where='post', label='I_top [A]')
        ax.step(t/60, Ib, color='steelblue',    lw=1.5, where='post', label='I_bot [A]')
        ax2 = ax.twinx()
        ax2.step(t/60, Pj, color='darkorange', lw=1.5, where='post', ls='--', label='P_joule [W]')
        ax2.set_ylabel("P_joule [W]", color='darkorange')
        ax.set_xlabel("Time [min]"); ax.set_ylabel("Current [A]")
        ax.set_title("RL agent actions"); ax.legend(fontsize=7); ax2.legend(fontsize=7, loc='center right')
        ax.grid(alpha=0.3)

        # [1,1] Wire temperatures
        ax = fig.add_subplot(gs[1, 1])
        ax.plot(t/60, Tt, color='tomato',    lw=2, label='T_top')
        ax.plot(t/60, Tb, color='steelblue', lw=2, label='T_bot')
        ax.axhline(ENV_CONFIG["T_as"] if "T_as" in ENV_CONFIG else 70,
                   ls='--', color='gray', lw=1, label='T_as (70°C)')
        ax.axhline(ENV_CONFIG["T_wire_limit"],
                   ls='--', color='tomato', lw=1, alpha=0.7, label='T_limit (200°C)')
        ax.set_xlabel("Time [min]"); ax.set_ylabel("T [°C]")
        ax.set_title("SMA wire temperatures"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

        # [1,2] Martensite fraction
        ax = fig.add_subplot(gs[1, 2])
        ax.plot(t/60, xit, color='tomato',    lw=2, label='ξ_top')
        ax.plot(t/60, xib, color='steelblue', lw=2, label='ξ_bot')
        ax.set_ylim(-0.05, 1.10)
        ax.set_xlabel("Time [min]"); ax.set_ylabel("ξ [—]")
        ax.set_title("Martensite fraction (1=cold, 0=hot)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # [2,0:2] EI vs time
        ax = fig.add_subplot(gs[2, :2])
        ax.semilogy(t/60, EI, color='mediumseagreen', lw=2)
        ax.axhline(self.cfg["EI_ref_stiff"], ls='--', color='gray', lw=1,
                   label=f"EI_stiff = {self.cfg['EI_ref_stiff']:.3f} N·m²")
        ax.axhline(self.cfg["EI_ref_soft"],  ls=':', color='gray', lw=1,
                   label=f"EI_soft  = {self.cfg['EI_ref_soft']:.4f} N·m²")
        shade_eclipse(ax)
        ax.set_xlabel("Time [min]"); ax.set_ylabel("EI [N·m²]")
        ax.set_title("Boom bending stiffness EI(t) — controlled by P_joule")
        ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')

        # [2,2] Pointing error distribution
        ax = fig.add_subplot(gs[2, 2])
        err = ts - th
        err_sunlit = err[~ec]
        ax.hist(err_sunlit, bins=30, color='darkviolet', alpha=0.7, edgecolor='white')
        ax.axvline(0,   color='gray', lw=1, ls='--')
        ax.axvline( 5,  color='tomato', lw=1, ls=':', label='±5°')
        ax.axvline(-5,  color='tomato', lw=1, ls=':')
        ax.set_xlabel("Pointing error [°]")
        ax.set_ylabel("Count")
        ax.set_title(f"Pointing error distribution (sunlit)\nRMS = {np.sqrt(np.mean(err_sunlit**2)):.2f}°")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        fig.tight_layout()

        if self.render_mode == "rgb_array":
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return img
        else:
            return fig

    def close(self):
        pass


# ============================================================================
# Sanity checks — run before training to verify the environment is valid
# ============================================================================

def run_sanity_checks(env, n_steps=200, verbose=True):
    """
    Run the Gymnasium environment checker plus a few physics sanity checks.
    Call this before training to catch issues early.
    """
    from gymnasium.utils.env_checker import check_env
    if verbose:
        print("\n── Gymnasium API check ──")
    check_env(env, warn=True)
    if verbose:
        print("  ✓  Gymnasium API valid")

    # Random rollout
    obs, _ = env.reset(options={"fixed_start": True})
    assert obs.shape == env.observation_space.shape
    total_r = 0.0
    for i in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_r += reward
        assert env.observation_space.contains(obs), f"obs out of bounds at step {i}: {obs}"
        if terminated or truncated:
            obs, _ = env.reset()

    if verbose:
        print(f"  ✓  {n_steps}-step random rollout complete")
        print(f"     Mean reward per step: {total_r/n_steps:.4f}")
        print(f"     Final obs:            {obs}")
    return True


# ============================================================================
# Quick training demo with Stable-Baselines3 SAC
# ============================================================================

def train_demo(total_timesteps=10_000, save_path=None):
    """
    Demonstrate training with SAC (Soft Actor-Critic).
    SAC is recommended for this environment because:
      - Continuous action space (currents, power)
      - Significant thermal lag (delayed reward signal)
      - Off-policy: sample-efficient
      - Entropy regularisation helps explore the I/EI coupling

    For serious training, increase total_timesteps to 500_000+
    and use vectorised environments (make_vec_env).
    """
    from stable_baselines3 import SAC
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback

    print("\n── Training with SAC ──")
    env      = SMPCBoomEnv()
    eval_env = SMPCBoomEnv()

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=50_000,
        batch_size=256,
        gamma=0.99,           # discount — long time-horizon for orbital tracking
        tau=0.005,
        ent_coef="auto",      # automatic entropy tuning
        policy_kwargs=dict(
            net_arch=[256, 256],   # two hidden layers of 256
        ),
        tensorboard_log=os.path.join(save_path or "/mnt/user-data/outputs", "tb_logs"),
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_path or "/mnt/user-data/outputs", "sac_model"),
        log_path=os.path.join(save_path or "/mnt/user-data/outputs", "sac_logs"),
        eval_freq=2000,
        n_eval_episodes=3,
        deterministic=True,
        verbose=0,
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_cb, progress_bar=False)

    print(f"  Training done. Model saved to {save_path or './models/'}")
    return model, env


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    OUTPUT = "/mnt/user-data/outputs"
    os.makedirs(OUTPUT, exist_ok=True)

    print("="*60)
    print("  Step 4 — SMPCBoomEnv Gymnasium Environment")
    print("="*60)

    # ── 1. Build and check the environment ───────────────────────────
    print("\n[1] Building environment...")
    env = SMPCBoomEnv()
    print(f"  Action space:      {env.action_space}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Control timestep:  {env.cfg['control_dt']} s")
    print(f"  Episode length:    {env.cfg['orbital_period']/env.cfg['control_dt']:.0f} steps "
          f"({env.cfg['orbital_period']/60:.0f} min)")
    print(f"  EI stiff: {env.cfg['EI_ref_stiff']:.4f} N·m²  "
          f"EI soft: {env.cfg['EI_ref_soft']:.5f} N·m²  "
          f"ratio: {env.cfg['EI_ref_stiff']/env.cfg['EI_ref_soft']:.0f}×")

    # ── 2. Sanity check ───────────────────────────────────────────────
    print("\n[2] Running sanity checks...")
    run_sanity_checks(env, n_steps=100, verbose=True)

    # ── 3. Heuristic rollout — demonstrate the environment behaviour ──
    # A simple rule-based policy: activate top wire when sun is above
    # horizon, bottom wire when below; apply P_joule whenever tracking
    # error exceeds 10°. This should produce a sensible (not optimal)
    # episode that we can render.
    print("\n[3] Running heuristic policy rollout (1 orbital period)...")
    obs, _ = env.reset(options={"fixed_start": True})
    n_steps_episode = int(env.cfg["orbital_period"] / env.cfg["control_dt"])
    total_reward = 0.0

    for i in range(n_steps_episode):
        theta_sun_now, in_eclipse = sun_angle(env.t, env.cfg)
        theta_error = theta_sun_now - env.theta_boom
        abs_err = abs(theta_error)

        # Heuristic: proportional to error, with P_joule if large error
        I_base  = np.clip(abs_err / 45.0, 0, 1)
        P_base  = np.clip((abs_err - 10.0) / 35.0, 0, 1) if abs_err > 10 else 0.0

        if in_eclipse:
            action = np.array([0, 0, 0], dtype=np.float32)
        elif theta_error > 0:
            action = np.array([I_base, 0.0, P_base], dtype=np.float32)
        else:
            action = np.array([0.0, I_base, P_base], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated:
            print(f"  Episode terminated at step {i} (wire overtemp)")
            break

    print(f"  Episode complete: {i+1} steps, total reward = {total_reward:.2f}")
    print(f"  Mean reward/step: {total_reward/(i+1):.4f}")

    # ── 4. Render and save ────────────────────────────────────────────
    print("\n[4] Rendering episode...")
    fig = env.render()
    if fig is not None:
        path = os.path.join(OUTPUT, "step4_heuristic_rollout.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved → {path}")

    # ── 5. Short SAC training demo ────────────────────────────────────
    print("\n[5] Short SAC training demo (2000 steps — illustrative only)...")
    print("    For real training, run with total_timesteps=500_000+")
    model, train_env = train_demo(
        total_timesteps=2000,
        save_path=os.path.join(OUTPUT, "sac_model")
    )

    # Evaluate trained policy for one episode
    print("\n[6] Evaluating trained policy...")
    obs, _ = train_env.reset(options={"fixed_start": True})
    eval_reward = 0.0
    for i in range(n_steps_episode):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = train_env.step(action)
        eval_reward += reward
        if terminated or truncated:
            break

    fig_eval = train_env.render()
    if fig_eval is not None:
        path_eval = os.path.join(OUTPUT, "step4_trained_policy_rollout.png")
        fig_eval.savefig(path_eval, dpi=150, bbox_inches='tight')
        plt.close(fig_eval)
        print(f"  Eval reward (2000-step model): {eval_reward:.2f}")
        print(f"  Saved → {path_eval}")

    print("\n" + "="*60)
    print("  Environment ready. Next steps for real training:")
    print("  1. Increase total_timesteps to 500_000–2_000_000")
    print("  2. Use make_vec_env(SMPCBoomEnv, n_envs=8) for parallelism")
    print("  3. Tune reward weights (lambda_energy, lambda_rate) in ENV_CONFIG")
    print("  4. Replace synthetic sun angle with real orbital ephemeris (poliastro)")
    print("  5. Swap n_boom_nodes=30 → 60 for more accurate thermal solve")
    print("="*60)
