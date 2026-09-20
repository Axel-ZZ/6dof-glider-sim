"""
validate.py  —  6-DOF dynamics model validation (Stage 1)
==========================================================
  1. Trim solve  — find (delta_e, delta_t) s.t. x_dot = 0
  2. Simulate    — run 10s from trim
  3. Plot        — ground track, altitude, Euler angles, airspeed

Run from project root:
    python -m <your_package>.validate
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ── Adjust this import to match your package structure ───────────────────────
from open_loop_tests.open_loop_test import simulate, trim_solve

# ─────────────────────────────────────────────────────────────────────────────
#  Perturbation test
# ─────────────────────────────────────────────────────────────────────────────

def perturbation_test(state_trim, u_ctrl_trim, dt=0.01, t_end=15.0):
    """
    Apply small perturbations from trim, hold trim controls fixed, and
    check that the aircraft returns toward trim (stable) rather than diverging.

    Longitudinal: +5 deg pitch nudge  → expect short-period damping back
    Lateral:      +10 deg roll nudge  → expect dihedral effect rolling back NOT STABLE IN THE ROLL
    Directional:   +5 deg/s yaw-rate nudge  → expect yaw-rate damping
    """
    perturbations = [
        {
            'name'       : 'Pitch perturbation  (+5 deg θ)',
            'state_delta': np.array([0,0,0, 0,0,0, 0, np.radians(1), 0, 0,0,0]),
            'expect'     : 'CM_alpha < 0  →  should damp back to trim pitch',
        },
        {
            'name'       : 'Roll perturbation  (+10 deg φ)',
            'state_delta': np.array([0,0,0, 0,0,0, np.radians(1), 0, 0, 0,0,0]),
            'expect'     : 'CL_beta < 0  →  dihedral effect should roll back',
        },
        {
            'name'       : 'Yaw perturbation  (+5 deg_s r)',
            'state_delta': np.array([0,0,0, 0,0,np.radians(0.1), 0, 0, 0, 0,0,0]),
            'expect'     : 'CN_r < 0 and CN_beta > 0  →  should damp yaw rate',
        },
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Perturbation Recovery — trim controls held fixed", fontsize=13)

    print("="*50)
    print("  PERTURBATION RECOVERY TEST")
    print("="*50)

    trim_phi   = np.degrees(state_trim[6])
    trim_theta = np.degrees(state_trim[7])
    trim_r     = np.degrees(state_trim[5])
    
    trim_Va    = np.sqrt(state_trim[0]**2 + state_trim[2]**2)
    trim_alt   = -state_trim[11]
    trim_psi   = np.degrees(state_trim[8])

    for i, pert in enumerate(perturbations):
        # Adding perturbation to trim state, keeping controls fixed at trim values.
        state_p      = state_trim + pert['state_delta']
        t, states    = simulate(state_p, u_ctrl_trim, dt=dt, t_end=t_end)

        phi   = np.degrees(states[:, 6])
        theta = np.degrees(states[:, 7])
        psi   = np.degrees(states[:, 8])
        r_deg = np.degrees(states[:, 5])

        # if i == 0:  # pitch — plot theta and airspeed
        #     # Only stable perturbation: pitch returns toward trim, and airspeed is stable.
        #     # 1. Longitudinal static stability is strong: CM_alpha is clearly negative.
        #     # 2. Pitch-rate damping is strong: CM_q is large and negative.
        #     # 3. Longitudinal coupling is simpler/weaker than lateral-directional coupling.

        #     name = pert['name']
        #     axes[0, 0].plot(t, theta, label='θ (deg)')
        #     axes[0, 0].axhline(trim_theta, color='k', ls='--', lw=1, label='trim')
        #     axes[0, 0].set_ylabel('Pitch θ (deg)')
        #     axes[0, 0].set_title(pert['name'])
        #     axes[0, 0].legend(fontsize=8)
        #     axes[0, 0].grid(True, alpha=0.3)

        #     # For pe,pn, pd
        #     alt   = -states[:, 11]
        #     pn = states[:, 9]
        #     pe = states[:, 10]
        #     Va    = np.sqrt(states[:, 0]**2 + states[:, 1]**2 + states[:, 2]**2)  

        # if i == 1:  # roll — plot phi and altitude
        #     # Unstable in the roll:
        #     # 1. Dihedral effect is very weak: CL_beta is close to zero, so roll back toward trim is very slow.
        #     # 2. Coupled lateral mode grows and dominates
        
        #     name = pert['name']
        #     axes[0, 0].plot(t, phi, label='φ (deg)')
        #     axes[0, 0].axhline(trim_phi, color='k', ls='--', lw=1, label='trim')
        #     axes[0, 0].set_ylabel('Roll φ (deg)')
        #     axes[0, 0].set_title(pert['name'])
        #     axes[0, 0].legend(fontsize=8)
        #     axes[0, 0].grid(True, alpha=0.3)

        #     # For pe,pn, pd
        #     alt   = -states[:, 11]
        #     pn = states[:, 9]
        #     pe = states[:, 10]
        #     Va    = np.sqrt(states[:, 0]**2 + states[:, 1]**2 + states[:, 2]**2)  

        if i == 2:  # yaw — plot yaw rate and altitude
            # Fast mode is stable: initial correction comes from yaw damping (CN_r < 0), so r drops at first.
            # Slow mode is unstable: a coupled lateral mode (often spiral) grows later and dominates.

            name = pert['name']
            axes[0, 0].plot(t, r_deg, label='r (deg/s)')
            axes[0, 0].axhline(trim_r, color='k', ls='--', lw=1, label='trim')
            axes[0, 0].set_ylabel('Yaw rate r (deg/s)')
            axes[0, 0].set_title(pert['name'])
            axes[0, 0].legend(fontsize=8)
            axes[0, 0].grid(True, alpha=0.3)

            # For pe,pn, pd
            alt   = -states[:, 11]
            pn = states[:, 9]
            pe = states[:, 10]
            Va    = np.sqrt(states[:, 0]**2 + states[:, 1]**2 + states[:, 2]**2)

    # Airspeed
    axes[1, 0].plot(t, Va, label='Va (m/s)')
    axes[1, 0].axhline(trim_Va, color='k', ls='--', lw=1, label='trim')
    axes[1, 0].set_ylabel('Airspeed (m/s)')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylim(trim_Va - 3, trim_Va + 3)
    axes[1, 0].set_xlim(t[0], t[-1])
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    # Ground track
    axes[0, 1].plot(pe, pn, label='Ground track')
    axes[0, 1].set_xlabel("East (m)")
    axes[0, 1].set_ylabel("North (m)")
    axes[0, 1].set_title("Ground track")
    axes[0, 1].axis("equal")
    axes[0, 1].legend(fontsize=8)

    # Altitude
    axes[1, 1].plot(t, alt, label='altitude (m)')
    axes[1, 1].axhline(trim_alt, color='k', ls='--', lw=1, label='trim')
    axes[1, 1].set_ylabel('Altitude (m)')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylim(alt[0] - 5, alt[0] + 5)
    axes[1, 1].set_xlim(t[0], t[-1])
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, name


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Trim
    state_trim, u_ctrl_trim = trim_solve()

    # Simulate 10s from trim
    t, states = simulate(state_trim, u_ctrl_trim, dt=0.01, t_end=10.0)

    # Perturbation test (select which type of pertubation to apply in the function)
    fig_pert, name = perturbation_test(state_trim, u_ctrl_trim)
    plt.savefig(f"plots\\pertubation_tests\\{name}.png", dpi=150)

    plt.show()

    print(f"Done! {len(t)} timesteps,  final altitude: {-states[-1, 11]:.1f} m")