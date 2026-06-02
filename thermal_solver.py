"""
thermal_solver.py
-----------------
Step 2: 1D finite-difference thermal model for the SMPC boom.

Solves the heat equation along the boom axis:

    ρ c_p ∂T/∂t = k_th ∂²T/∂x² + q_joule(x,t) - q_rad(x,t)

Boundary conditions (spacecraft / vacuum environment):
  - Root (x=0): fixed temperature (spacecraft bus, T_bus ≈ 20°C)
  - Tip  (x=L): radiation-only (adiabatic approximation or radiative BC)

Joule heating:
    q_joule(x,t) = P(t) / (A_cross * L_boom)   [W/m³]
    (assumes uniform resistive element; extend to spatially varying if needed)

Radiative cooling (vacuum — no convection):
    q_rad(x,t) = σ_sb * ε * (T(x,t)⁴ - T_space⁴) * (P_surface / A_cross)

where P_surface/A_cross is the surface-area-to-cross-sectional-area ratio
of the boom, approximated geometrically.

Output at each timestep:
  - T(x, t)      : temperature field [°C]
  - E(x, t)      : Young's modulus field from constitutive model [Pa]
  - EI(x, t)     : Bending stiffness field [N·m²]

References:
  - Liu et al. 2023 (SMPC boom geometry)
  - Cengel, Heat Transfer (radiation in vacuum)
  - White et al. (2001) – SMP thermal modelling
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import warnings


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
SIGMA_SB = 5.670374419e-8   # W/(m²·K⁴)  Stefan-Boltzmann constant
T_SPACE_K = 4.0              # K           deep space background


# ---------------------------------------------------------------------------
# Default thermal material properties
# SMPC composite: epoxy-based SMP matrix with CFRP/glass reinforcement
# Adjust to match your specific layup from materials testing.
# ---------------------------------------------------------------------------
DEFAULT_THERMAL = {
    "rho":       1400.0,   # kg/m³   — composite density
    "c_p":        900.0,   # J/(kg·K) — specific heat capacity
    "k_th":         0.5,   # W/(m·K)  — axial thermal conductivity
    "emissivity":   0.85,  # —         surface emissivity (typical CFRP)
    "T_bus":       20.0,   # °C        spacecraft bus (root BC)
    "T_init":      20.0,   # °C        initial temperature (isothermal)
}


# ---------------------------------------------------------------------------
# Boom geometry for thermal model
# ---------------------------------------------------------------------------
DEFAULT_BOOM = {
    "length":     1.5,    # m    — deployed boom length
    "width":      0.030,  # m    — chord width
    "thickness":  0.0012, # m    — wall thickness
}


def _boom_geometric_ratios(width, thickness, boom_type="lenticular"):
    """
    Returns:
      A_cross   : cross-sectional area [m²]
      P_surface : perimeter (surface per unit length) [m]
    """
    if boom_type == "lenticular":
        # Two thin arcs → approximate total perimeter ≈ pi * width
        P_surface = np.pi * width
        A_cross   = 2.0 * width * thickness  # approximate
    else:
        P_surface = 2.0 * (width + thickness)
        A_cross   = width * thickness
    return A_cross, P_surface


# ---------------------------------------------------------------------------
# Thermal Solver
# ---------------------------------------------------------------------------

class ThermalSolver:
    """
    1D finite-difference thermal solver for the SMPC boom.

    Parameters
    ----------
    constitutive : SMPCModel
        Calibrated SMPC constitutive model (provides E(T) and EI(T)).
    boom_params : dict, optional
        Boom geometry: length, width, thickness.
    thermal_params : dict, optional
        Material thermal properties.
    n_nodes : int
        Number of spatial nodes along the boom axis.

    Example
    -------
    from smpc_constitutive import SMPCModel
    from thermal_solver import ThermalSolver

    model = SMPCModel()
    solver = ThermalSolver(model, n_nodes=50)

    # Simulate 300 s of Joule heating at 30 W then free cooling
    def power_schedule(t):
        return 30.0 if t < 150 else 0.0

    T_hist, t_hist = solver.simulate(t_end=300, dt=0.5,
                                     power_fn=power_schedule)
    solver.plot_history(T_hist, t_hist)
    """

    def __init__(self, constitutive, boom_params=None, thermal_params=None,
                 n_nodes=60):
        from smpc_constitutive import SMPCModel  # local import for standalone use

        self.mat   = constitutive
        self.boom  = boom_params   or DEFAULT_BOOM.copy()
        self.therm = thermal_params or DEFAULT_THERMAL.copy()
        self.n     = n_nodes

        # Spatial grid
        self.L  = self.boom["length"]
        self.dx = self.L / (self.n - 1)
        self.x  = np.linspace(0, self.L, self.n)

        # Geometric ratios
        self._A_cross, self._P_surf = _boom_geometric_ratios(
            self.boom["width"], self.boom["thickness"]
        )

        # Volume-specific heat capacity [J/(m³·K)]
        self._rho_cp = self.therm["rho"] * self.therm["c_p"]

        # Thermal diffusivity [m²/s]
        self._alpha = self.therm["k_th"] / self._rho_cp

        # Radiation prefactor: σ ε (P_surf/A_cross) / (ρ c_p)  [1/(K³·s)]
        self._rad_coeff = (
            SIGMA_SB * self.therm["emissivity"]
            * (self._P_surf / self._A_cross)
            / self._rho_cp
        )

        print(f"ThermalSolver initialised")
        print(f"  Boom length     : {self.L:.2f} m  ({self.n} nodes, dx={self.dx*1e3:.2f} mm)")
        print(f"  Thermal diff.   : {self._alpha:.2e} m²/s")
        print(f"  Rad. coeff.     : {self._rad_coeff:.2e} (K³·s)⁻¹")
        print(f"  CFL limit (cond): dt < {0.4 * self.dx**2 / self._alpha:.4f} s  (capped at 1.0 s)")

    # ------------------------------------------------------------------
    # CFL-safe timestep
    # ------------------------------------------------------------------

    def max_stable_dt(self, safety=0.4):
        """
        Maximum stable timestep for explicit scheme (conduction-limited).
        Also capped at 1.0 s so simulations remain physically meaningful
        for timescales of minutes; override dt in simulate() for longer runs.
        """
        dt_cfl = safety * self.dx**2 / self._alpha
        return min(dt_cfl, 1.0)

    # ------------------------------------------------------------------
    # Single timestep
    # ------------------------------------------------------------------

    def _step(self, T, dt, q_joule_vol):
        """
        Advance temperature field by one timestep using explicit FD.

        T             : (n,) array, temperatures in °C
        dt            : timestep [s]
        q_joule_vol   : volumetric heat source [W/m³] (scalar or (n,) array)
        """
        T_K = T + 273.15   # convert to Kelvin for radiation

        dT = np.zeros(self.n)

        # --- Conduction (interior nodes) ---
        # ∂²T/∂x² ≈ (T[i+1] - 2T[i] + T[i-1]) / dx²
        d2T = np.zeros(self.n)
        d2T[1:-1] = (T[2:] - 2*T[1:-1] + T[:-2]) / self.dx**2
        # Forward/backward difference at interior-adjacent boundaries
        # (boundary nodes handled separately below)

        # --- Volumetric Joule heating ---
        q_j = np.full(self.n, q_joule_vol) if np.isscalar(q_joule_vol) else q_joule_vol

        # --- Radiative cooling to space (linearised T⁴ term is nonlinear,
        #     so we keep it explicit — stable for small dt) ---
        T_space_K = T_SPACE_K
        q_rad_vol = self._rad_coeff * (T_K**4 - T_space_K**4)  # > 0 means heat loss

        # --- Interior update ---
        dT[1:-1] = dt * (
            self._alpha * d2T[1:-1] * (self._rho_cp / self._rho_cp)  # = alpha * d2T
            + q_j[1:-1] / self._rho_cp
            - q_rad_vol[1:-1]
        )
        # Simplify: alpha = k/(rho*cp), so k*d2T/(rho*cp) = alpha*d2T
        dT[1:-1] = dt * (
            self._alpha * d2T[1:-1]
            + q_j[1:-1] / self._rho_cp
            - q_rad_vol[1:-1]
        )

        T_new = T + dT

        # --- Boundary conditions ---
        # Root (x=0): spacecraft bus — Dirichlet
        T_new[0] = self.therm["T_bus"]

        # Tip (x=L): radiation-only (Neumann adiabatic + radiation)
        # Use ghost-node / forward-difference for conduction = 0 at tip,
        # then apply radiation:
        T_new[-1] = T[-1] + dt * (
            + q_j[-1] / self._rho_cp
            - q_rad_vol[-1]
        )

        return T_new

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------

    def simulate(self, t_end, dt=None, power_fn=None,
                 power_spatial_fn=None, store_every=1):
        """
        Simulate thermal response of the boom.

        Parameters
        ----------
        t_end : float
            Simulation end time [s].
        dt : float, optional
            Timestep [s]. Defaults to 80% of CFL limit.
        power_fn : callable(t) -> float, optional
            Total Joule heating power [W] as function of time.
            Assumed uniformly distributed unless power_spatial_fn is given.
            If None, no Joule heating (free cooling / isothermal).
        power_spatial_fn : callable(t, x) -> float, optional
            Spatial power density [W/m³] at position x and time t.
            Overrides power_fn if provided.
        store_every : int
            Store state every N timesteps (reduces memory for long runs).

        Returns
        -------
        T_history : ndarray, shape (n_stored, n_nodes)
            Temperature field at each stored timestep [°C].
        t_history : ndarray, shape (n_stored,)
            Simulation times [s].
        """
        if dt is None:
            dt = self.max_stable_dt(safety=0.4)
            print(f"  Auto dt = {dt:.4f} s  (CFL-safe)")

        if dt > self.max_stable_dt(safety=1.0):
            warnings.warn(
                f"dt={dt:.4f} s exceeds CFL limit ({self.max_stable_dt():.4f} s). "
                "Simulation may be unstable. Consider reducing dt or using implicit scheme.",
                UserWarning
            )

        n_steps   = int(np.ceil(t_end / dt))
        T         = np.full(self.n, self.therm["T_init"], dtype=float)
        T_history = []
        t_history = []

        for i in range(n_steps):
            t = i * dt

            # --- Compute volumetric heat source ---
            if power_spatial_fn is not None:
                q_j_vol = power_spatial_fn(t, self.x)  # W/m³ array
            elif power_fn is not None:
                P_total = power_fn(t)                  # W total
                q_j_vol = P_total / (self._A_cross * self.L)  # uniform W/m³
            else:
                q_j_vol = 0.0

            T = self._step(T, dt, q_j_vol)

            if i % store_every == 0:
                T_history.append(T.copy())
                t_history.append(t)

        T_history = np.array(T_history)
        t_history = np.array(t_history)

        print(f"  Simulation done: {n_steps} steps, "
              f"{len(t_history)} snapshots stored")
        print(f"  Tip T at t_end : {T_history[-1, -1]:.1f} °C")
        print(f"  Max T anywhere : {T_history.max():.1f} °C")

        return T_history, t_history

    # ------------------------------------------------------------------
    # Derived fields: E(x,t) and EI(x,t)
    # ------------------------------------------------------------------

    def compute_stiffness_history(self, T_history):
        """
        From a T(x,t) history, compute E(x,t) and EI(x,t).

        Parameters
        ----------
        T_history : ndarray, shape (n_time, n_nodes)

        Returns
        -------
        E_history  : ndarray, shape (n_time, n_nodes)  [Pa]
        EI_history : ndarray, shape (n_time, n_nodes)  [N·m²]
        """
        E_history  = self.mat.E(T_history)
        EI_history = self.mat.EI(T_history)
        return E_history, EI_history

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_history(self, T_history, t_history, E_history=None,
                     n_snapshots=6, save_path=None):
        """
        Multi-panel summary plot:
          - Temperature profiles along boom at N snapshots
          - Tip temperature vs. time
          - (If E_history provided) EI profile at final time
        """
        n_snap = min(n_snapshots, len(t_history))
        snap_idx = np.linspace(0, len(t_history)-1, n_snap, dtype=int)
        cmap = plt.cm.plasma
        norm = Normalize(vmin=t_history[0], vmax=t_history[-1])

        n_cols = 3 if E_history is not None else 2
        fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 4.5))
        fig.suptitle("SMPC Boom Thermal Simulation", fontsize=13, fontweight='bold')

        # --- Panel 1: T(x) snapshots ---
        ax0 = axes[0]
        for idx in snap_idx:
            color = cmap(norm(t_history[idx]))
            ax0.plot(self.x * 100, T_history[idx], color=color,
                     lw=1.5, label=f"t={t_history[idx]:.0f}s")
        ax0.axhline(self.mat.T_trans, ls='--', color='gray', lw=1,
                    label=f"$T_{{trans}}$={self.mat.T_trans:.0f}°C")
        ax0.set_xlabel("Position along boom  [cm]")
        ax0.set_ylabel("Temperature [°C]")
        ax0.set_title("Temperature profiles T(x, t)")
        ax0.legend(fontsize=7, ncol=2)
        ax0.grid(alpha=0.3)

        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax0, label="Time [s]", shrink=0.85)

        # --- Panel 2: Tip T vs. time ---
        ax1 = axes[1]
        ax1.plot(t_history, T_history[:, -1], 'tomato', lw=2, label="Tip")
        ax1.plot(t_history, T_history[:, self.n//2], 'steelblue', lw=1.5,
                 ls='--', label="Mid-boom")
        ax1.axhline(self.mat.T_trans, ls='--', color='gray', lw=1,
                    label=f"$T_{{trans}}$ = {self.mat.T_trans:.0f}°C")
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Temperature [°C]")
        ax1.set_title("Tip and mid-boom temperature vs. time")
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3)

        # --- Panel 3 (optional): EI(x) at final time ---
        if E_history is not None:
            ax2 = axes[2]
            ax2.plot(self.x * 100, E_history[-1] / 1e6, 'steelblue', lw=2,
                     label=f"t={t_history[-1]:.0f}s (heated)")
            ax2.plot(self.x * 100, E_history[0] / 1e6, 'gray', lw=1.5,
                     ls='--', label=f"t={t_history[0]:.0f}s (initial)")
            ax2.set_xlabel("Position along boom  [cm]")
            ax2.set_ylabel("E(x)  [MPa]")
            ax2.set_title("Stiffness field along boom")
            ax2.set_yscale('log')
            ax2.legend(fontsize=9)
            ax2.grid(alpha=0.3, which='both')

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved → {save_path}")
        return fig

    def plot_stiffness_spacetime(self, T_history, t_history, save_path=None):
        """
        2D colormap: EI(x, t) as a heatmap — shows how stiffness field
        evolves spatially and temporally under Joule heating.
        """
        EI_hist = self.mat.EI(T_history)  # (n_time, n_nodes)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        fig.suptitle("SMPC Boom Stiffness Space–Time Evolution", fontsize=13, fontweight='bold')

        # Temperature heatmap
        im0 = axes[0].pcolormesh(
            self.x * 100, t_history,
            T_history, cmap='inferno', shading='auto'
        )
        axes[0].set_xlabel("Position [cm]")
        axes[0].set_ylabel("Time [s]")
        axes[0].set_title("T(x, t)  [°C]")
        fig.colorbar(im0, ax=axes[0], label="°C")

        # EI heatmap (log scale)
        im1 = axes[1].pcolormesh(
            self.x * 100, t_history,
            np.log10(EI_hist), cmap='viridis', shading='auto'
        )
        axes[1].set_xlabel("Position [cm]")
        axes[1].set_ylabel("Time [s]")
        axes[1].set_title("log₁₀ EI(x, t)  [N·m²]")
        fig.colorbar(im1, ax=axes[1], label="log₁₀(N·m²)")

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved → {save_path}")
        return fig

    def summary(self):
        print("=" * 50)
        print("  ThermalSolver Parameters")
        print("=" * 50)
        print(f"  Boom length        : {self.L:.2f} m")
        print(f"  Nodes              : {self.n}")
        print(f"  dx                 : {self.dx*1e3:.2f} mm")
        print(f"  Density            : {self.therm['rho']:.0f} kg/m³")
        print(f"  Specific heat      : {self.therm['c_p']:.0f} J/(kg·K)")
        print(f"  Axial conductivity : {self.therm['k_th']:.2f} W/(m·K)")
        print(f"  Emissivity         : {self.therm['emissivity']:.2f}")
        print(f"  Cross-section area : {self._A_cross:.4e} m²")
        print(f"  Surface perimeter  : {self._P_surf:.4e} m")
        print(f"  Thermal diffusivity: {self._alpha:.4e} m²/s")
        print(f"  CFL-safe max dt    : {self.max_stable_dt():.4f} s")
        print("=" * 50)
