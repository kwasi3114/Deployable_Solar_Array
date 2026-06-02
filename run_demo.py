"""
run_demo.py
-----------
Demonstration and validation runner for Steps 1 & 2:
  - Step 1: SMPC constitutive model (E(T), EI(T))
  - Step 2: 1D Joule heating thermal solver

Runs three scenarios:
  A. Step heating: 30 W for 150 s, then free radiative cooling
  B. Pulsed heating: 10 s on / 20 s off cycles
  C. Spatially-graded heating: more power near the root (mimics a
     resistive element that tapers in resistance along the boom)

Saves all plots to /mnt/user-data/outputs/.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt

from smpc_constitutive import SMPCModel, make_synthetic_dma_data
from thermal_solver import ThermalSolver


#OUTPUT_DIR = "/mnt/user-data/outputs"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper: print section headers
# ---------------------------------------------------------------------------
def banner(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Step 1: Constitutive model
# ---------------------------------------------------------------------------

def run_constitutive_demo():
    banner("Step 1 — SMPC Constitutive Model")

    # 1a. Default (literature) model
    print("\n[1a] Default parameters (literature values)")
    model = SMPCModel()
    model.summary()

    # 1b. Fit to synthetic DMA data (replace with your CSV data)
    print("\n[1b] Fitting to synthetic DMA data")
    print("     → In your workflow: load your DMA CSV here, e.g.")
    print("       import pandas as pd")
    print("       df = pd.read_csv('dma_sweep.csv')")
    print("       T_data = df['Temperature_C'].values")
    print("       E_data = df['Storage_Modulus_Pa'].values")
    print("       model = SMPCModel.fit_from_dma(T_data, E_data)")
    print()

    T_data, E_data = make_synthetic_dma_data(model, noise_frac=0.04)
    model_fitted = SMPCModel.fit_from_dma(T_data, E_data)

    # 1c. Inverse: target temperatures for control
    print("\n[1c] Feedforward control temperatures")
    for frac in [0.1, 0.25, 0.5, 0.75, 0.9]:
        T_req = model_fitted.T_for_stiffness_fraction(frac)
        EI_at = model_fitted.EI(T_req)
        print(f"  {frac*100:4.0f}% softened → T = {T_req:5.1f}°C  "
              f"EI = {EI_at:.5f} N·m²")

    # 1d. Save constitutive plot
    fig = model_fitted.plot()
    path_const = os.path.join(OUTPUT_DIR, "step1_constitutive_model.png")
    fig.savefig(path_const, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Plot saved → {path_const}")

    return model_fitted


# ---------------------------------------------------------------------------
# Step 2: Thermal solver scenarios
# ---------------------------------------------------------------------------

def run_thermal_scenarios(model):
    banner("Step 2 — 1D Joule Heating Thermal Solver")

    solver = ThermalSolver(model, n_nodes=60)
    solver.summary()

    # ----------------------------------------------------------------
    # Scenario A: Step power — 30 W for 150 s, then free cooling
    # ----------------------------------------------------------------
    banner("Scenario A: 30 W step heating → radiative cooling")

    def power_step(t):
        return 30.0 if t < 150 else 0.0

    T_A, t_A = solver.simulate(
        t_end=400, power_fn=power_step, store_every=2
    )
    E_A, EI_A = solver.compute_stiffness_history(T_A)

    fig_A = solver.plot_history(
        T_A, t_A, E_history=E_A, n_snapshots=7,
        save_path=os.path.join(OUTPUT_DIR, "step2A_step_heating.png")
    )
    plt.close(fig_A)

    fig_A2 = solver.plot_stiffness_spacetime(
        T_A, t_A,
        save_path=os.path.join(OUTPUT_DIR, "step2A_spacetime_EI.png")
    )
    plt.close(fig_A2)

    # Quick stats
    print(f"\n  Scenario A summary:")
    print(f"  Peak tip T       : {T_A[:, -1].max():.1f} °C  "
          f"at t={t_A[np.argmax(T_A[:,-1])]:.0f} s")
    print(f"  EI at peak T     : {EI_A[np.argmax(T_A[:,-1]), -1]:.5f} N·m²")
    print(f"  EI at T_trans    : {model.EI(model.T_trans):.5f} N·m²  (reference)")
    print(f"  EI ratio (stiff→soft): {EI_A[0, 0]/EI_A[np.argmax(T_A[:,-1]), -1]:.1f}×")

    # ----------------------------------------------------------------
    # Scenario B: Pulsed heating — 10 s on / 20 s off
    # ----------------------------------------------------------------
    banner("Scenario B: Pulsed heating (10s on / 20s off @ 40 W)")

    def power_pulsed(t):
        period = 30.0
        return 40.0 if (t % period) < 10.0 else 0.0

    T_B, t_B = solver.simulate(
        t_end=240, power_fn=power_pulsed, store_every=2
    )
    E_B, EI_B = solver.compute_stiffness_history(T_B)

    fig_B, axes_B = plt.subplots(1, 2, figsize=(11, 4.5))
    fig_B.suptitle("Scenario B — Pulsed Joule Heating", fontsize=13, fontweight='bold')

    axes_B[0].plot(t_B, T_B[:, -1], 'tomato', lw=2, label="Tip")
    axes_B[0].plot(t_B, T_B[:, solver.n//2], 'steelblue', lw=1.5, ls='--', label="Mid")
    axes_B[0].axhline(model.T_trans, ls='--', color='gray', lw=1,
                      label=f"$T_{{trans}}$={model.T_trans:.0f}°C")
    # Shade heating pulses
    t_eval = np.linspace(0, 240, 1000)
    on_mask = np.array([power_pulsed(t) > 0 for t in t_eval])
    for start, stop in _find_intervals(t_eval, on_mask):
        axes_B[0].axvspan(start, stop, alpha=0.12, color='orange')
    axes_B[0].set_xlabel("Time [s]")
    axes_B[0].set_ylabel("Temperature [°C]")
    axes_B[0].set_title("Tip/mid temperature — pulsed heating")
    axes_B[0].legend(fontsize=9)
    axes_B[0].grid(alpha=0.3)

    axes_B[1].semilogy(t_B, EI_B[:, -1], 'tomato', lw=2, label="EI tip")
    axes_B[1].semilogy(t_B, EI_B[:, solver.n//2], 'steelblue', lw=1.5,
                       ls='--', label="EI mid")
    axes_B[1].axhline(model.EI(model.T_trans), ls='--', color='gray', lw=1,
                      label="EI at $T_{trans}$")
    axes_B[1].set_xlabel("Time [s]")
    axes_B[1].set_ylabel("EI  [N·m²]")
    axes_B[1].set_title("Bending stiffness — pulsed heating")
    axes_B[1].legend(fontsize=9)
    axes_B[1].grid(alpha=0.3, which='both')

    fig_B.tight_layout()
    path_B = os.path.join(OUTPUT_DIR, "step2B_pulsed_heating.png")
    fig_B.savefig(path_B, dpi=150, bbox_inches='tight')
    plt.close(fig_B)
    print(f"\n  Plot saved → {path_B}")

    # ----------------------------------------------------------------
    # Scenario C: Spatially graded heating
    # (higher power density near root — simulates tapered resistive element)
    # ----------------------------------------------------------------
    banner("Scenario C: Spatially graded heating (root-concentrated)")

    def power_graded(t, x):
        """
        Gaussian power density centred at root (x=0), decaying along boom.
        Total power ≈ 25 W; sigma controls the heated length.
        """
        if t > 180:
            return np.zeros_like(x)
        sigma = solver.L * 0.25            # heat concentrated in first 25% of boom
        A_cross = solver._A_cross
        p_vol   = np.exp(-0.5*(x/sigma)**2)
        # Normalise so integral = P_total
        P_total = 25.0
        norm    = np.trapezoid(p_vol, x) * A_cross
        return P_total * p_vol / norm      # W/m³

    T_C, t_C = solver.simulate(
        t_end=300, power_spatial_fn=power_graded, store_every=2
    )
    E_C, EI_C = solver.compute_stiffness_history(T_C)

    # Plot final spatial profiles (heated vs initial)
    fig_C, axes_C = plt.subplots(1, 2, figsize=(11, 4.5))
    fig_C.suptitle("Scenario C — Spatially Graded Heating (root-concentrated)",
                   fontsize=13, fontweight='bold')

    snap_times = [0, 30, 60, 120, 180, 300]
    snap_idxs  = [np.argmin(np.abs(t_C - ts)) for ts in snap_times]
    cmap_C = plt.cm.plasma
    norm_C = plt.Normalize(vmin=0, vmax=300)

    for idx in snap_idxs:
        c = cmap_C(norm_C(t_C[idx]))
        axes_C[0].plot(solver.x * 100, T_C[idx], color=c, lw=1.5,
                       label=f"t={t_C[idx]:.0f}s")
        axes_C[1].semilogy(solver.x * 100, EI_C[idx], color=c, lw=1.5)

    axes_C[0].axhline(model.T_trans, ls='--', color='gray', lw=1,
                      label=f"$T_{{trans}}$")
    axes_C[0].set_xlabel("Position [cm]")
    axes_C[0].set_ylabel("T  [°C]")
    axes_C[0].set_title("Temperature profile — graded heating")
    axes_C[0].legend(fontsize=7, ncol=2)
    axes_C[0].grid(alpha=0.3)

    axes_C[1].set_xlabel("Position [cm]")
    axes_C[1].set_ylabel("EI  [N·m²]")
    axes_C[1].set_title("Stiffness profile — graded heating")
    axes_C[1].grid(alpha=0.3, which='both')

    sm_C = plt.cm.ScalarMappable(cmap=cmap_C, norm=norm_C)
    sm_C.set_array([])
    fig_C.colorbar(sm_C, ax=axes_C, label="Time [s]", shrink=0.85)

    fig_C.tight_layout()
    path_C = os.path.join(OUTPUT_DIR, "step2C_graded_heating.png")
    fig_C.savefig(path_C, dpi=150, bbox_inches='tight')
    plt.close(fig_C)
    print(f"\n  Plot saved → {path_C}")

    return solver


# ---------------------------------------------------------------------------
# Utility: find contiguous True intervals in a boolean array
# ---------------------------------------------------------------------------
def _find_intervals(t, mask):
    intervals = []
    in_block = False
    for i, m in enumerate(mask):
        if m and not in_block:
            start = t[i]; in_block = True
        elif not m and in_block:
            intervals.append((start, t[i-1])); in_block = False
    if in_block:
        intervals.append((start, t[-1]))
    return intervals


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  SMPC Boom — Steps 1 & 2 Validation Run")
    print("="*60)

    model   = run_constitutive_demo()
    solver  = run_thermal_scenarios(model)

    print("\n" + "="*60)
    print("  All outputs saved to:", OUTPUT_DIR)
    print("  Files generated:")
    import glob
    for f in sorted(glob.glob(os.path.join(OUTPUT_DIR, "step*.png"))):
        print(f"    {os.path.basename(f)}")
    print("="*60)
    print("\n  Next steps:")
    print("  1. Replace make_synthetic_dma_data() with your real DMA CSV data")
    print("  2. Tune DEFAULT_THERMAL / DEFAULT_BOOM to your fabricated boom")
    print("  3. Run smpc_constitutive.py::SMPCModel.fit_from_dma() on real data")
    print("  4. Validate solver against ABAQUS thermal step for a 30W input")
    print("  5. Proceed to Step 3: Brinson SMA actuator model")
