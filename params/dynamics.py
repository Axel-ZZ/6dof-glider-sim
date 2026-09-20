import os
import sys
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from params import params as P


# ─────────────────────────────────────────────────────────────────────────────
#  State vector (12 states):
#   x = [u, v, w,           body-frame velocities        (m/s)
#         p, q, r,           body-frame angular rates     (rad/s)
#         phi, theta, psi,   Euler angles                 (rad)
#         pn, pe, pd]        inertial position (NED)      (m)
#
#  Control input vector (4 inputs):
#   u_ctrl = [delta_e, delta_a, delta_r, delta_t]
#             elevator aileron  rudder   throttle
#             (rad)    (rad)    (rad)    (0 to 1)
# ─────────────────────────────────────────────────────────────────────────────


def init_state():
    """
    Initialize a trimmed straight-and-level flight state.
    Va = 17 m/s, altitude = 20m.
    """
    alpha0 = np.radians(2.1)
    Va0 = 17.0
    return np.array([
        Va0 * np.cos(alpha0), 0.0, Va0 * np.sin(alpha0), # u, v, w
        0.0, 0.0, 0.0,                                   # p, q, r
        0.0, alpha0, 0.0,                                # phi, theta, psi
        0.0, 0.0, -20.0                                  # pn, pe, pd (20m altitude)
    ])


# ── Nonlinear lift model with stall (Beard & McLain eq. 4.9 / 4.10) ──────────
def _CL_nonlinear(alpha):
    """
    Blended lift coefficient: linear at low AoA, flat-plate at stall.
    Eq. (4.9) and (4.10) from Beard & McLain.
    """
    # Correct
    M  = P.M_SIGMOID
    a0 = P.ALPHA0_STALL

    sigma = ((1 + np.exp(-M * (alpha - a0)) + np.exp(M * (alpha + a0)))
             / ((1 + np.exp(-M * (alpha - a0))) * (1 + np.exp(M * (alpha + a0)))))

    CL_linear     = P.CL_0 + P.CL_alpha * alpha
    CL_flat_plate = 2.0 * np.sign(alpha) * np.sin(alpha)**2 * np.cos(alpha)

    return (1 - sigma) * CL_linear + sigma * CL_flat_plate


# ── Nonlinear drag model (Beard & McLain eq. 4.11) ───────────────────────────
def _CD_nonlinear(alpha):
    """
    Quadratic (polar) drag model.
    CD(alpha) = CDp + (CL0 + CLalpha * alpha)^2 / (pi * e * AR)
    """
    # Correct
    return P.CD_p + (P.CL_0 + P.CL_alpha * alpha)**2 / (np.pi * P.E * P.AR)


# ── Aerodynamic forces and moments ───────────────────────────────────────────
def aero_forces_moments(state, u_ctrl):
    """
    Compute aerodynamic forces [fx, fy, fz] and moments [l, m, n]
    in the body frame, following Beard & McLain Chapter 4 summary
    (equations 4.19 and 4.20).
    """
    u, v, w = state[0], state[1], state[2]
    p, q, r = state[3], state[4], state[5]

    delta_e, delta_a, delta_r, _ = u_ctrl

    # ── Airspeed, angle of attack, sideslip ──────────────────────────────────
    # Correct
    Va    = np.sqrt(u**2 + v**2 + w**2)
    Va    = max(Va, 0.1)
    alpha = np.arctan2(w, u)
    beta  = np.arcsin(np.clip(v / Va, -1.0, 1.0))

    q_bar = 0.5 * P.RHO * Va**2

    # ── Nonlinear CL and CD (with stall) ─────────────────────────────────────
    CL_a = _CL_nonlinear(alpha)
    CD_a = _CD_nonlinear(alpha)

    # ── CX and CZ: body-frame force coefficients (B&M eq. 4.17 / 4.18) ──────
    # Rotate lift/drag from stability frame into body frame via alpha
    # Correct
    CX    = -CD_a * np.cos(alpha)         + CL_a * np.sin(alpha)
    CX_q  = -P.CD_q * np.cos(alpha)       + P.CL_q * np.sin(alpha)
    CX_de = -P.CD_delta_e * np.cos(alpha) + P.CL_delta_e * np.sin(alpha)

    CZ    = -CD_a * np.sin(alpha)         - CL_a * np.cos(alpha)
    CZ_q  = -P.CD_q * np.sin(alpha)       - P.CL_q * np.cos(alpha)
    CZ_de = -P.CD_delta_e * np.sin(alpha) - P.CL_delta_e * np.cos(alpha)

    # ── Total forces in body frame (B&M eq. 4.19) ────────────────────────────
    # Correct
    fx = q_bar * P.S * (CX
                        + CX_q  * (P.C / (2 * Va)) * q
                        + CX_de * delta_e)

    fy = q_bar * P.S * (P.CY_0
                        + P.CY_beta    * beta
                        + P.CY_p       * (P.B / (2 * Va)) * p
                        + P.CY_r       * (P.B / (2 * Va)) * r
                        + P.CY_delta_a * delta_a
                        + P.CY_delta_r * delta_r)

    fz = q_bar * P.S * (CZ
                        + CZ_q  * (P.C / (2 * Va)) * q
                        + CZ_de * delta_e)

    # ── Total moments in body frame (B&M eq. 4.20) ───────────────────────────
    # Correct
    l = q_bar * P.S * P.B * (P.CL_0_moment
                              + P.CL_beta    * beta
                              + P.CL_p       * (P.B / (2 * Va)) * p
                              + P.CL_r       * (P.B / (2 * Va)) * r
                              + P.CL_delta_a * delta_a
                              + P.CL_delta_r * delta_r)

    m = q_bar * P.S * P.C * (P.CM_0
                              + P.CM_alpha   * alpha
                              + P.CM_q       * (P.C / (2 * Va)) * q
                              + P.CM_delta_e * delta_e)

    n = q_bar * P.S * P.B * (P.CN_0
                              + P.CN_beta    * beta
                              + P.CN_p       * (P.B / (2 * Va)) * p
                              + P.CN_r       * (P.B / (2 * Va)) * r
                              + P.CN_delta_a * delta_a
                              + P.CN_delta_r * delta_r)

    return np.array([fx, fy, fz]), np.array([l, m, n])


# ── Propulsion forces and moments (B&M eq. 4.15 / 4.16) ─────────────────────
def propulsion_forces_moments(state, u_ctrl):
    """
    Bernoulli-based propeller thrust and propeller torque.
    Thrust acts along body x-axis only.
    Propeller torque opposes direction of rotation (roll moment).
    """
    # Correct
    u, v, w = state[0], state[1], state[2]
    Va      = np.sqrt(u**2 + v**2 + w**2)
    delta_t = float(np.clip(u_ctrl[3], 0.0, 1.0)) # unsure about this line, but it seems prudent to ensure throttle is in [0, 1]

    # Thrust (B&M eq. 4.15)
    Fx_prop = (0.5 * P.RHO * P.S_PROP * P.C_PROP
               * ((P.K_MOTOR * delta_t)**2 - Va**2))

    # Propeller torque (B&M eq. 4.16)
    l_prop = 0 # -P.K_TP * (P.K_OMEGA * delta_t)**2

    return np.array([Fx_prop, 0.0, 0.0]), np.array([l_prop, 0.0, 0.0])


# ── Full 6-DOF equations of motion ───────────────────────────────────────────
def dynamics(t, state, u_ctrl):
    """
    6-DOF rigid body equations of motion following Beard & McLain Ch. 3 & 4.
    Returns state derivative x_dot (12,).
    Pass directly to scipy.integrate.solve_ivp.
    """
    u, v, w = state[0], state[1], state[2]
    p, q, r = state[3], state[4], state[5]
    phi, theta, psi = state[6], state[7], state[8]

    m   = P.MASS
    g   = P.GRAVITY

    # ── Forces and moments ───────────────────────────────────────────────────
    F_aero, M_aero = aero_forces_moments(state, u_ctrl)
    F_prop, M_prop = propulsion_forces_moments(state, u_ctrl)

    # Gravity in body frame (B&M eq. 4.1)
    F_grav = np.array([
        -m * g * np.sin(theta),
         m * g * np.cos(theta) * np.sin(phi),
         m * g * np.cos(theta) * np.cos(phi),
    ])

    F   = F_grav + F_aero + F_prop
    tau = M_aero + M_prop

    fx, fy, fz       = F
    l_tot, m_tot, n_tot = tau

    # ── Translational dynamics ───────────────────────────────────────────────
    u_dot = (fx / m) + r * v - q * w
    v_dot = (fy / m) - r * u + p * w
    w_dot = (fz / m) + q * u - p * v
    
    p_dot = P.Gamma1 * p * q - P.Gamma2 * q * r + P.Gamma3 * l_tot + P.Gamma4 * n_tot
    q_dot = P.Gamma5 * p * r - P.Gamma6 * (p**2 - r**2) + m_tot / P.I_YY
    r_dot = P.Gamma7 * p * q - P.Gamma1 * q * r + P.Gamma4 * l_tot + P.Gamma8 * n_tot

    # ── Kinematic equations (body rates → Euler angle rates) ─────────────────
    phi_dot   = p + np.sin(phi) * np.tan(theta) * q + np.cos(phi) * np.tan(theta) * r
    theta_dot = np.cos(phi) * q - np.sin(phi) * r
    psi_dot   = (np.sin(phi) / np.cos(theta)) * q + (np.cos(phi) / np.cos(theta)) * r

    # ── Position update (body velocity → inertial NED) ───────────────────────
    cphi,   sphi   = np.cos(phi),   np.sin(phi)
    ctheta, stheta = np.cos(theta), np.sin(theta)
    cpsi,   spsi   = np.cos(psi),   np.sin(psi)

    R = np.array([
        [ctheta*cpsi,  sphi*stheta*cpsi - cphi*spsi,  cphi*stheta*cpsi + sphi*spsi],
        [ctheta*spsi,  sphi*stheta*spsi + cphi*cpsi,  cphi*stheta*spsi - sphi*cpsi],
        [-stheta,      sphi*ctheta,                    cphi*ctheta                 ],
    ])

    pn_dot, pe_dot, pd_dot = R @ np.array([u, v, w])

    return np.array([
        u_dot, v_dot, w_dot,
        p_dot, q_dot, r_dot,
        phi_dot, theta_dot, psi_dot,
        pn_dot, pe_dot, pd_dot,
    ])


def update(state, u_ctrl, dt):
    """
    Step the 6-DOF dynamics forward in time by dt using RK4 integration.
    state: (12,) array
    u_ctrl: (4,) array [delta_e, delta_a, delta_r, delta_t]
    dt: time step (s)
    """
    k1 = dynamics(0, state, u_ctrl)
    k2 = dynamics(0, state + k1 * (dt / 2), u_ctrl)
    k3 = dynamics(0, state + k2 * (dt / 2), u_ctrl)
    k4 = dynamics(0, state + k3 * dt, u_ctrl)

    return state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

if __name__ == "__main__":
    # Test nonlinear CL and CD models
    alpha_test = range(-50, 50, 2)  # Test angles of attack from -20 to +20 degrees
    CL_test = [_CL_nonlinear(np.radians(a)) for a in alpha_test]
    CD_test = [_CD_nonlinear(np.radians(a)) for a in alpha_test]

    plt.plot(alpha_test, CL_test, label='CL')
    plt.xlabel('Angle of Attack (degrees)')
    plt.ylabel('Coefficient')
    plt.legend()
    plt.title('Nonlinear Drag Coefficients vs Angle of Attack')
    plt.savefig('CD_nonlinear.png', dpi=150)

    plt.plot(alpha_test, CD_test, label='CD')
    plt.xlabel('Angle of Attack (degrees)')
    plt.ylabel('Coefficient')
    plt.legend()
    plt.title('Nonlinear Lift Coefficient vs Angle of Attack')
    plt.savefig('CL_nonlinear.png', dpi=150)

    plt.show()
