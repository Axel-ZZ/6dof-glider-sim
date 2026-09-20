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
from params import params as P
from params.dynamics import dynamics, update, init_state


# ─────────────────────────────────────────────────────────────────────────────
#  Trim solve
# ─────────────────────────────────────────────────────────────────────────────

def trim_solve(Va=25.0, alpha_guess=np.radians(2.1)):
    """
    Find (delta_e, delta_t, alpha) such that longitudinal x_dot = 0
    for straight-and-level flight (gamma = 0, so theta = alpha).
    Lateral inputs fixed at zero (symmetric flight).
    """
    def residual(x):
        # Unknowns solved by fsolve: elevator, throttle, and trim alpha.
        delta_e, delta_t, alpha = x
        theta = alpha  # level flight: flight path angle gamma = theta - alpha = 0

        # Build a full 12-state vector from the assumed symmetric trim condition.
        state = np.array([
            Va * np.cos(alpha), 0.0, Va * np.sin(alpha),
            0.0, 0.0, 0.0,
            0.0, theta, 0.0,
            0.0, 0.0, -50.0,
        ])

        # Only elevator and throttle are active in this longitudinal trim solve.
        u_ctrl = np.array([delta_e, 0.0, 0.0, delta_t])
        xdot = dynamics(0, state, u_ctrl)
        return [xdot[0], xdot[2], xdot[4]]  # u_dot, w_dot, q_dot (Longitudal dynamics)

    # Initial guess for [delta_e, delta_t, alpha].
    x0  = [0.0, 0.5, alpha_guess]
    sol, _, ier, msg = fsolve(residual, x0, full_output=True)

    delta_e, delta_t, alpha = sol
    theta = alpha  # Needed to go straight (gamma = theta - alpha = 0)

    # Reconstruct trim state/control and compute full-state residuals for reporting.
    state_trim = np.array([
        Va * np.cos(alpha), 0.0, Va * np.sin(alpha),
        0.0, 0.0, 0.0,
        0.0, theta, 0.0,
        0.0, 0.0, -50.0,
    ])
    u_ctrl_trim = np.array([delta_e, 0.0, 0.0, delta_t])
    residual_full = dynamics(0, state_trim, u_ctrl_trim)

    print("\n" + "="*50)
    print("  TRIM REPORT")
    print("="*50)
    print(f"  Converged        : {'YES ✓' if ier == 1 else 'NO ✗  ' + msg}")
    print(f"  Airspeed Va      : {Va:.2f} m/s")
    print(f"  Alpha α          : {np.degrees(alpha):.8f} deg")
    print(f"  Pitch angle θ    : {np.degrees(theta):.8f} deg")
    print(f"  Elevator δe      : {np.degrees(delta_e):.8f} deg")
    print(f"  Throttle δt      : {delta_t:.8f}")
    labels = ['u_dot','v_dot','w_dot','p_dot','q_dot','r_dot',
              'phi_dot','theta_dot','psi_dot','pn_dot','pe_dot','pd_dot']
    print(f"\n  Per-state residuals:")
    for label, val in zip(labels, residual_full):
        flag = "  <- !" if abs(val) > 1e-4 else ""
        print(f"    {label:<12s}: {val:+.4e}{flag}")
    print("="*50 + "\n")

    return state_trim, u_ctrl_trim

# Get trim for Va = 25 m/s
state_trim, u_ctrl_trim = trim_solve(Va=25.0, alpha_guess=np.radians(2.1))
print(f"Trim state: {state_trim}")
print(f"Trim control: {u_ctrl_trim}")

# ─────────────────────────────────────────────────────────────────────────────
#  Simulate
# ─────────────────────────────────────────────────────────────────────────────

def simulate(state0, u_ctrl, dt=0.01, t_end=10.0):
    """Propagate the 12-state model forward in time with fixed controls."""
    n      = int(t_end / dt)
    times  = np.arange(0, t_end, dt)
    states = np.zeros((n, 12))
    # Seed the trajectory with the provided initial condition (typically trim).
    states[0] = state0
    for i in range(1, n):
        # One integration/update step using the model's internal update routine.
        states[i] = update(states[i-1], u_ctrl, dt)
    return times, states


# # ─────────────────────────────────────────────────────────────────────────────
# #  Plot  (same layout as open_loop_test.py)
# # ─────────────────────────────────────────────────────────────────────────────

def plot_results(t, states, title="F5B Glider — Trim Validation"):
    pn   = states[:, 9]
    pe   = states[:, 10]
    alt  = -states[:, 11]          # pd -> altitude
    phi  = np.degrees(states[:, 6])
    theta= np.degrees(states[:, 7])
    psi  = np.degrees(states[:, 8])
    Va   = np.sqrt(states[:, 0]**2 + states[:, 1]**2 + states[:, 2]**2)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=14)

    # Ground track
    axes[0, 0].plot(pe, pn)
    axes[0, 0].set_xlabel("East (m)")
    axes[0, 0].set_ylabel("North (m)")
    axes[0, 0].set_title("Ground track")
    axes[0, 0].axis("equal")

    # Altitude
    axes[0, 1].plot(t, alt)
    axes[0, 1].set_ylim(alt[0] - 5, alt[0] + 5)
    axes[0, 1].set_xlim(t[0], t[-1])
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Altitude (m)")
    axes[0, 1].set_title("Altitude")

    # Euler angles
    axes[1, 0].plot(t, phi,   label="Roll phi")
    axes[1, 0].plot(t, theta, label="Pitch theta")
    axes[1, 0].plot(t, psi,   label="Yaw psi")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Angle (deg)")
    axes[1, 0].set_title("Euler angles")
    axes[1, 0].legend()

    # Airspeed
    axes[1, 1].plot(t, Va)
    axes[1, 1].set_ylim(Va[0] - 3, Va[0] + 3)
    axes[1, 1].set_xlim(t[0], t[-1])
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Airspeed (m/s)")
    axes[1, 1].set_title("Airspeed")

    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Trim
    state_trim, u_ctrl_trim = trim_solve()

    """
    Trim conditions:
        Alpha α       : 9.040 deg
        Pitch angle θ : 9.040 deg
        Elevator δe   : -24.238 deg
        Throttle δt   : 0.4620

    Sign conventions follow Beard & McLain:

    Elevator δe:
            +Δ (trailing edge down) → nose pitches UP
            -Δ (trailing edge up)   → nose pitches DOWN

        Aileron δa:
            +Δ (right aileron down, left up) → roll RIGHT (right wing down)
            -Δ (right aileron up, left down) → roll LEFT
            Note: also causes small adverse yaw opposite to roll (CN_delta_a < 0)

        Rudder δr:
            +Δ → yaw LEFT   (CN_delta_r = -0.069, negative yaw moment)
            -Δ → yaw RIGHT
            Note: this is B&M convention — opposite to some other textbooks

        Throttle δt:
            +Δ → increased thrust → accelerates (↑ airspeed, possible climb)
            -Δ → decreased thrust → decelerates (↓ airspeed, possible descent)
    """
    # TODO: Adjust trim inputs to see that it behaves as it should

    # Up or down from trim
    # u_ctrl_trim[0] += -0.05 # Elevator deflection  (rad)      UP
    # u_ctrl_trim[0] += 0.05  # Elevator deflection  (rad)      DOWN

    # Left and right roll and hence turn
    # u_ctrl_trim[1] += -0.005 # Aileron deflection  (rad)      LEFT
    # u_ctrl_trim[1] += 0.005  # Aileron deflection  (rad)      RIGHT

    # Left or right turn
    # u_ctrl_trim[2] += -0.005 # Rudder deflection   (rad)      RIGHT
    # u_ctrl_trim[2] += 0.005  # Rudder deflection   (rad)      LEFT

    # Increase or decrease airspeed
    # u_ctrl_trim[3] += -0.005 # Throttle setting    (0 to 1)   DECENT
    # u_ctrl_trim[3] += 0.005  # Throttle setting    (0 to 1)   CLIMB

    # Simulate 10s from trim
    t, states = simulate(state_trim, u_ctrl_trim, dt=0.01, t_end=10.0)

    # Plot
    fig = plot_results(t, states)
    plt.savefig("plots/pertubation_tests/trim_validation.png", dpi=150)
    plt.show()

    print(f"Done! {len(t)} timesteps,  final altitude: {-states[-1, 11]:.1f} m")