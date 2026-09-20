import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ── Adjust this import to match your package structure ───────────────────────
from params import params as P
from params.dynamics import dynamics, update
# prop

def trim_solve(Va=25.0, alpha_guess=np.radians(2.1)):
    """
    Find (delta_e, alpha) such that longitudinal we are slowing down in the 
    horizontal plane but staying at constant altitude in the vertical plane
    for straight-and-level flight (gamma = 0, so theta = alpha).
    Lateral inputs fixed at zero (symmetric flight).
    """
    def straight_level_residual(x):
        # Unknowns solved by fsolve: elevator, throttle, and trim alpha.
        delta_e, alpha = x
        theta = alpha  # level flight: flight path angle gamma = theta - alpha = 0

        # Build a full 12-state vector from the assumed symmetric trim condition.
        state = np.array([
            Va * np.cos(alpha), 0.0, Va * np.sin(alpha),
            0.0, 0.0, 0.0,
            0.0, theta, 0.0,
            0.0, 0.0, -50.0,
        ])

        # Only elevator and throttle are active in this longitudinal trim solve.
        u_ctrl = np.array([delta_e, 0.0, 0.0, 0.0])
        xdot = dynamics(0, state, u_ctrl)
        return [xdot[2], xdot[4]]  # u_dot, w_dot, q_dot (Longitudal dynamics)

    x0  = [0.0, alpha_guess]
    sol, _, ier, msg = fsolve(straight_level_residual, x0, full_output=True)

    delta_e, alpha = sol
    theta = alpha  # Needed to go straight (gamma = theta - alpha = 0)

    # Reconstruct trim state/control and compute full-state residuals for reporting.
    state_trim = np.array([
        Va * np.cos(alpha), 0.0, Va * np.sin(alpha),
        0.0, 0.0, 0.0,
        0.0, theta, 0.0,
        0.0, 0.0, -150.0,
    ])
    u_ctrl_trim = np.array([delta_e, 0.0, 0.0, 0.0])
    residual_full = dynamics(0, state_trim, u_ctrl_trim)

    print("\n" + "="*50)
    print("  TRIM REPORT")
    print("="*50)
    print(f"  Converged        : {'YES ✓' if ier == 1 else 'NO ✗  ' + msg}")
    print(f"  Airspeed Va      : {Va:.2f} m/s")
    print(f"  Alpha α          : {np.degrees(alpha):.8f} deg")
    print(f"  Pitch angle θ    : {np.degrees(theta):.8f} deg")
    print(f"  Elevator δe      : {np.degrees(delta_e):.8f} deg")
    labels = ['u_dot','v_dot','w_dot','p_dot','q_dot','r_dot',
              'phi_dot','theta_dot','psi_dot','pn_dot','pe_dot','pd_dot']
    print(f"\n  Per-state residuals:")
    for label, val in zip(labels, residual_full):
        flag = "  <- !" if abs(val) > 1e-4 else ""
        print(f"    {label:<12s}: {val:+.4e}{flag}")
    print("="*50 + "\n")

    return state_trim, u_ctrl_trim, alpha, delta_e

# Get trim for Va = 25 m/s
state_trim, u_ctrl_trim, _, _ = trim_solve(Va=25.0, alpha_guess=np.radians(2.1))
print(f"Trim state: {state_trim}")
print(f"Trim control: {u_ctrl_trim}")