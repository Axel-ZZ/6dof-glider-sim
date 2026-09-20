import numpy as np
from scipy.optimize import least_squares
import os
import sys

# Ensure parent directory is in path for params import
sys.path.insert(0, os.getcwd())

from params.dynamics import dynamics

def trim_residual(x, Va):
    # x = [delta_e, delta_f, alpha]
    delta_e, delta_f, alpha = x
    theta = alpha  # Level flight assumption: gamma = 0 => theta = alpha
    
    # State: [u, v, w, p, q, r, phi, theta, psi, pn, pe, pd]
    state = np.array([
        Va * np.cos(alpha), 0.0, Va * np.sin(alpha), # u, v, w
        0.0, 0.0, 0.0,                              # p, q, r
        0.0, theta, 0.0,                            # phi, theta, psi
        0.0, 0.0, -100.0                            # position
    ])
    
    # Control: [delta_e, delta_a, delta_r, delta_f]
    u_ctrl = np.array([delta_e, 0.0, 0.0, delta_f])
    
    xdot = dynamics(0, state, u_ctrl)
    
    # Residuals: [u_dot, w_dot, q_dot]
    return [xdot[0], xdot[2], xdot[4]]

Va = 25.0
x0 = [0.0, 0.0, np.radians(2.0)]
bounds = (
    [np.radians(-35), np.radians(-35), np.radians(-10)], # lower bounds
    [np.radians(35), np.radians(35), np.radians(20)]    # upper bounds
)

res = least_squares(trim_residual, x0, bounds=bounds, args=(Va,))

print(f"Success: {res.success}")
print(f"Message: {res.message}")
print(f"Solution (delta_e, delta_f, alpha) in deg: {np.degrees(res.x)}")
print(f"Residual norm: {np.linalg.norm(res.fun)}")
print(f"Residuals [u_dot, w_dot, q_dot]: {res.fun}")
