"""
smpc_constitutive.py
--------------------
Step 1: SMPC temperature-dependent constitutive model.

Fits a smooth E(T) transition curve from DMA test data (storage modulus vs
temperature) and exposes:
  - E(T)        : Young's modulus [Pa] at temperature T [°C]
  - EI(T)       : Bending stiffness for a given cross-section
  - f_shape(T)  : Shape-memory transition function (0=glassy, 1=rubbery)

The transition is modelled with a Boltzmann sigmoid:

    E(T) = E_r + (E_g - E_r) / (1 + exp(k*(T - T_trans)))

where
    E_g    = glassy modulus  (low T, stiff)
    E_r    = rubbery modulus (high T, soft)
    T_trans = midpoint of glass transition [°C]
    k       = steepness of transition [1/°C]

If you have real DMA data, call SMPCModel.fit_from_dma(T_data, E_data) to
calibrate all four parameters via nonlinear least squares.

Reference: Liu et al. (2023), Chinese J. Mech. Eng. 36:67
           Brinson & Huang (1996) – SMP constitutive framework
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ---------------------------------------------------------------------------
# Default material parameters (CFRP/epoxy SMP composite, typical values)
# Swap these out once you have DMA data from your actual layup.
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    "E_g":      2.5e9,   # Pa  — glassy modulus  (~room temp, stiff)
    "E_r":      5.0e6,   # Pa  — rubbery modulus (~above T_g, soft)
    "T_trans":  65.0,    # °C  — glass transition midpoint
    "k":        0.18,    # 1/°C — transition steepness (larger = sharper)
}

# Cross-section geometry for the SMPC boom (lenticular profile from Liu 2023)
# Adjust to match your fabricated boom dimensions.
DEFAULT_GEOMETRY = {
    "boom_type":   "lenticular",  # options: "lenticular", "tape-spring", "circular"
    "width":       0.030,         # m  — chord width
    "thickness":   0.0012,        # m  — wall thickness
    "subtend_deg": 120.0,         # °  — subtended angle (lenticular)
}


def _sigmoid_modulus(T, E_g, E_r, T_trans, k):
    """Boltzmann sigmoid: E(T) transitions from E_g to E_r."""
    return E_r + (E_g - E_r) / (1.0 + np.exp(k * (T - T_trans)))


def _second_moment_lenticular(width, thickness, subtend_deg):
    """
    Approximate second moment of area for a lenticular cross-section.
    Models the boom as two symmetric circular-arc tape-springs.
    I_approx = 2 * (t * R^3 / 12) * (phi - sin(phi)*cos(phi))
    where phi = subtend_deg/2 in radians, R = width / (2*sin(phi)).
    """
    phi = np.radians(subtend_deg / 2.0)
    R = width / (2.0 * np.sin(phi))
    I = 2.0 * thickness * R**3 / 12.0 * (phi - np.sin(phi) * np.cos(phi))
    return I


def _second_moment_circular(radius, thickness):
    """Thin-walled circular tube: I = pi * r^3 * t."""
    return np.pi * radius**3 * thickness


class SMPCModel:
    """
    Temperature-dependent constitutive model for an SMPC composite boom.

    Usage
    -----
    # Use default (literature) parameters
    model = SMPCModel()

    # Or fit from your DMA data
    model = SMPCModel.fit_from_dma(T_array, E_array)

    # Evaluate
    E  = model.E(75.0)          # Young's modulus at 75 °C  [Pa]
    EI = model.EI(75.0)         # Bending stiffness at 75 °C [N·m²]
    f  = model.f_shape(75.0)    # Shape-memory activation (0–1)
    """

    def __init__(self, params=None, geometry=None):
        p = params or DEFAULT_PARAMS
        self.E_g     = p["E_g"]
        self.E_r     = p["E_r"]
        self.T_trans = p["T_trans"]
        self.k       = p["k"]

        g = geometry or DEFAULT_GEOMETRY
        self.geometry = g
        self._compute_I()

    def _compute_I(self):
        g = self.geometry
        if g["boom_type"] == "lenticular":
            self._I = _second_moment_lenticular(
                g["width"], g["thickness"], g["subtend_deg"]
            )
        elif g["boom_type"] == "circular":
            self._I = _second_moment_circular(
                g["width"] / 2.0, g["thickness"]
            )
        else:  # tape-spring fallback: treat as thin flat strip
            b = g["width"]
            t = g["thickness"]
            self._I = b * t**3 / 12.0

    # ------------------------------------------------------------------
    # Core public API
    # ------------------------------------------------------------------

    def E(self, T):
        """
        Young's modulus at temperature T [°C] or array of temperatures.
        Returns value in Pa.
        """
        T = np.asarray(T, dtype=float)
        return _sigmoid_modulus(T, self.E_g, self.E_r, self.T_trans, self.k)

    def EI(self, T):
        """
        Bending stiffness EI [N·m²] at temperature T [°C].
        EI(T) = E(T) * I_cross_section
        """
        return self.E(T) * self._I

    def f_shape(self, T):
        """
        Shape-memory activation fraction (0 = fully glassy/stiff,
        1 = fully rubbery/soft/shape-programmable).
        """
        T = np.asarray(T, dtype=float)
        E_vals = _sigmoid_modulus(T, self.E_g, self.E_r, self.T_trans, self.k)
        return (self.E_g - E_vals) / (self.E_g - self.E_r)

    def T_for_stiffness_fraction(self, fraction):
        """
        Inverse: given a desired softening fraction (0–1), return the
        required temperature. Useful for feedforward control.
        fraction = 0 → T where fully glassy
        fraction = 1 → T where fully rubbery
        """
        fraction = np.clip(fraction, 1e-6, 1.0 - 1e-6)
        # Invert sigmoid analytically
        # fraction = 1/(1+exp(k*(T-T_trans)))  →  T = T_trans - ln(1/f - 1)/k
        return self.T_trans - np.log(1.0 / fraction - 1.0) / self.k

    @property
    def I(self):
        """Second moment of area of the boom cross-section [m^4]."""
        return self._I

    # ------------------------------------------------------------------
    # Fitting from real DMA data
    # ------------------------------------------------------------------

    @classmethod
    def fit_from_dma(cls, T_data, E_data, geometry=None, p0=None):
        """
        Fit the Boltzmann sigmoid to experimental DMA storage modulus data.

        Parameters
        ----------
        T_data : array-like, shape (N,)
            Temperature values from DMA sweep [°C].
        E_data : array-like, shape (N,)
            Storage modulus E' [Pa] at each temperature.
        geometry : dict, optional
            Cross-section geometry dict. Uses DEFAULT_GEOMETRY if None.
        p0 : list, optional
            Initial guess [E_g, E_r, T_trans, k]. Auto-estimated if None.

        Returns
        -------
        SMPCModel with fitted parameters.
        """
        T_data = np.asarray(T_data, dtype=float)
        E_data = np.asarray(E_data, dtype=float)

        if p0 is None:
            # Auto-estimate: E_g from lowest-T point, E_r from highest-T point
            p0 = [
                E_data[np.argmin(T_data)],   # E_g
                E_data[np.argmax(T_data)],   # E_r
                np.median(T_data),            # T_trans
                0.15,                         # k
            ]

        bounds = (
            [1e5,  1e4,  20.0, 0.01],   # lower bounds
            [200e9, 10e9, 150.0, 2.00],  # upper bounds
        )

        popt, pcov = curve_fit(
            _sigmoid_modulus, T_data, E_data,
            p0=p0, bounds=bounds, maxfev=10000
        )

        perr = np.sqrt(np.diag(pcov))
        print("=== DMA Fit Results ===")
        names = ["E_g (Pa)", "E_r (Pa)", "T_trans (°C)", "k (1/°C)"]
        for name, val, err in zip(names, popt, perr):
            print(f"  {name:<18}: {val:.4g}  ±  {err:.4g}")
        print(f"  I cross-section: {_second_moment_lenticular(**{k: v for k, v in (DEFAULT_GEOMETRY if geometry is None else geometry).items() if k in ['width','thickness','subtend_deg']}):.4e} m^4")

        params = {
            "E_g":     popt[0],
            "E_r":     popt[1],
            "T_trans": popt[2],
            "k":       popt[3],
        }
        model = cls(params=params, geometry=geometry)
        model._fit_cov = pcov
        model._T_data  = T_data
        model._E_data  = E_data
        return model

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot(self, T_range=(20, 120), show_data=True, ax=None):
        """
        Plot E(T) and EI(T) curves. If fit_from_dma was called, overlays
        the raw DMA data points.
        """
        T_plot = np.linspace(T_range[0], T_range[1], 400)

        fig = None
        if ax is None:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
            fig.suptitle("SMPC Constitutive Model", fontsize=13, fontweight='bold')
        else:
            axes = ax

        # --- Left: E(T) ---
        ax0 = axes[0]
        ax0.semilogy(T_plot, self.E(T_plot) / 1e6, 'steelblue', lw=2, label="Model E(T)")
        if show_data and hasattr(self, "_T_data"):
            ax0.semilogy(self._T_data, self._E_data / 1e6, 'o', color='tomato',
                         ms=4, zorder=5, label="DMA data")
        ax0.axvline(self.T_trans, ls='--', color='gray', lw=1, label=f"$T_{{trans}}$ = {self.T_trans:.1f} °C")
        ax0.set_xlabel("Temperature [°C]")
        ax0.set_ylabel("E(T)  [MPa]")
        ax0.set_title("Storage modulus vs. temperature")
        ax0.legend(fontsize=9)
        ax0.grid(True, which='both', alpha=0.3)

        # Annotate E_g and E_r
        ax0.annotate(f"$E_g$ = {self.E_g/1e9:.2f} GPa",
                     xy=(T_range[0]+2, self.E_g/1e6), fontsize=8, color='steelblue')
        ax0.annotate(f"$E_r$ = {self.E_r/1e6:.1f} MPa",
                     xy=(T_range[0]+2, self.E_r/1e6*1.4), fontsize=8, color='steelblue')

        # --- Right: EI(T) and f_shape(T) ---
        ax1 = axes[1]
        EI_vals = self.EI(T_plot)
        color_EI = 'steelblue'
        ax1.semilogy(T_plot, EI_vals, color=color_EI, lw=2, label="EI(T)")
        ax1.set_xlabel("Temperature [°C]")
        ax1.set_ylabel("EI(T)  [N·m²]", color=color_EI)
        ax1.tick_params(axis='y', labelcolor=color_EI)

        ax1b = ax1.twinx()
        ax1b.plot(T_plot, self.f_shape(T_plot), color='darkorange', lw=1.5,
                  ls='--', label="$f_{shape}$(T)  (softening fraction)")
        ax1b.set_ylabel("$f_{shape}$ (0=stiff, 1=soft)", color='darkorange')
        ax1b.tick_params(axis='y', labelcolor='darkorange')
        ax1b.set_ylim(-0.05, 1.15)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1b.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
        ax1.set_title("Bending stiffness & activation fraction")
        ax1.grid(True, which='both', alpha=0.3)

        if fig is not None:
            fig.tight_layout()
        return fig

    def summary(self):
        """Print a compact parameter summary."""
        print("=" * 46)
        print("  SMPC Constitutive Model Parameters")
        print("=" * 46)
        print(f"  Glassy modulus  E_g    : {self.E_g/1e9:.3f} GPa")
        print(f"  Rubbery modulus E_r    : {self.E_r/1e6:.2f}  MPa")
        print(f"  Transition temp T_trans: {self.T_trans:.1f} °C")
        print(f"  Transition slope k     : {self.k:.4f} 1/°C")
        print(f"  Cross-section  I       : {self._I:.4e} m⁴")
        print(f"  EI at 25 °C (stiff)   : {self.EI(25):.4f} N·m²")
        print(f"  EI at T_trans          : {self.EI(self.T_trans):.4f} N·m²")
        print(f"  EI at 100°C (soft)    : {self.EI(100):.4f} N·m²")
        print(f"  EI ratio (stiff/soft) : {self.EI(25)/self.EI(100):.1f}×")
        print("=" * 46)


# ---------------------------------------------------------------------------
# Quick synthetic DMA dataset — replace with your real data
# ---------------------------------------------------------------------------

def make_synthetic_dma_data(model=None, noise_frac=0.03, n_points=40, seed=42):
    """
    Generate synthetic DMA storage modulus data for testing the fitting routine.
    Replace this with your actual DMA sweep (pandas CSV or numpy arrays).
    """
    rng = np.random.default_rng(seed)
    if model is None:
        model = SMPCModel()
    T = np.linspace(20, 110, n_points)
    E_clean = model.E(T)
    noise = rng.normal(0, noise_frac, size=n_points)
    E_noisy = E_clean * (1.0 + noise)
    return T, E_noisy


# ---------------------------------------------------------------------------
# Main: demonstrate model, fitting, and plotting
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n── Step 1: Default model (literature parameters) ──\n")
    model_default = SMPCModel()
    model_default.summary()

    print("\n── Step 2: Fit model to synthetic DMA data ──\n")
    T_data, E_data = make_synthetic_dma_data(model_default)
    model_fitted = SMPCModel.fit_from_dma(T_data, E_data)

    print("\n── Step 3: Inverse — temperature needed for 50% softening ──")
    T50 = model_fitted.T_for_stiffness_fraction(0.5)
    T90 = model_fitted.T_for_stiffness_fraction(0.9)
    print(f"  T for 50% softening : {T50:.1f} °C")
    print(f"  T for 90% softening : {T90:.1f} °C")

    print("\n── Step 4: EI table ──")
    for T_eval in [25, 50, 65, 80, 100]:
        print(f"  T={T_eval:3d}°C  →  E={model_fitted.E(T_eval)/1e6:7.2f} MPa"
              f"   EI={model_fitted.EI(T_eval):.5f} N·m²"
              f"   f_shape={model_fitted.f_shape(T_eval):.3f}")

    print("\n── Step 5: Plotting ──")
    fig = model_fitted.plot()
    fig.savefig("/mnt/user-data/outputs/smpc_constitutive_model.png", dpi=150, bbox_inches='tight')
    print("  Saved → smpc_constitutive_model.png")
    plt.close(fig)
