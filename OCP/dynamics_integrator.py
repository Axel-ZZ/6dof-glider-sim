import os
import sys
import casadi as ca
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ── Adjust this import to match your package structure ───────────────────────
# from simulation import params_perch_modified as P
from params import params as P

"""
This file contains the dynamics integrator for the 6-DOF glider model without propulsion effects.
The propulsion effects are removed in the dynamics equations, allowing for a pure glider model.
"""

def build_dynamics():

    x = ca.MX.sym("x", 12)
    u = ca.MX.sym("u", 4)

    # -----------------States----------------
    u_b, v_b, w_b = x[0], x[1], x[2]
    p, q, r       = x[3], x[4], x[5]
    phi, theta, psi = x[6], x[7], x[8]
    pn, pe, pd    = x[9], x[10], x[11]

    # -----------------Controls----------------
    delta_e, delta_a, delta_r, delta_f = u[0], u[1], u[2], u[3]

    # -----------------Aerodynamic angles and dynamic pressure----------------
    Va = ca.sqrt(u_b**2 + v_b**2 + w_b**2)
    Va = ca.fmax(Va, 0.1)

    alpha = ca.atan2(w_b, u_b)
    beta  = ca.asin(ca.fmin(ca.fmax(v_b / Va, -1.0), 1.0))

    q_bar = 0.5 * P.RHO * Va**2

    # -----------------Sigmoid stall model----------------
    # NOTE: REPLACE WITH NEURAL FOIL
    M  = P.M_SIGMOID
    a0 = P.ALPHA0_STALL

    sigma = ((1 + ca.exp(-M * (alpha - a0)) + ca.exp(M * (alpha + a0))) /
            ((1 + ca.exp(-M * (alpha - a0))) * (1 + ca.exp(M * (alpha + a0)))))

    CL_linear = P.CL_0 + P.CL_alpha * alpha
    CL_flat   = 2.0 * ca.sign(alpha) * ca.sin(alpha)**2 * ca.cos(alpha)

    CL = (1 - sigma) * CL_linear + sigma * CL_flat

    CD = P.CD_p + (P.CL_0 + P.CL_alpha * alpha)**2 / (ca.pi * P.E * P.AR)

    # -----------------Force coefficients----------------
    CX = -CD * ca.cos(alpha) + CL * ca.sin(alpha) # REPLACE
    CZ = -CD * ca.sin(alpha) - CL * ca.cos(alpha) # REPLACE

    CX_q  = -P.CD_q * ca.cos(alpha) + P.CL_q * ca.sin(alpha)
    CX_de = -P.CD_delta_e * ca.cos(alpha) + P.CL_delta_e * ca.sin(alpha)
    CX_df = -P.CD_delta_f * ca.cos(alpha) + P.CL_delta_f * ca.sin(alpha)

    CZ_q  = -P.CD_q * ca.sin(alpha) - P.CL_q * ca.cos(alpha)
    CZ_de = -P.CD_delta_e * ca.sin(alpha) - P.CL_delta_e * ca.cos(alpha)
    CZ_df = -P.CD_delta_f * ca.sin(alpha) - P.CL_delta_f * ca.cos(alpha)

    # -----------------Forces----------------
    fx = q_bar * P.S * (
        CX
        + CX_q * (P.C / (2 * Va)) * q
        + CX_de * delta_e
        + CX_df * delta_f
    )

    fy = q_bar * P.S * (
        P.CY_0
        + P.CY_beta * beta
        + P.CY_p * (P.B / (2 * Va)) * p
        + P.CY_r * (P.B / (2 * Va)) * r
        + P.CY_delta_a * delta_a
        + P.CY_delta_r * delta_r
    )

    fz = q_bar * P.S * (
        CZ
        + CZ_q * (P.C / (2 * Va)) * q
        + CZ_de * delta_e
        + CZ_df * delta_f
    )

    # -----------------Moments----------------
    # KEEP
    l = q_bar * P.S * P.B * (
        P.CL_0_moment                       # 
        + P.CL_beta * beta
        + P.CL_p * (P.B / (2 * Va)) * p
        + P.CL_r * (P.B / (2 * Va)) * r
        + P.CL_delta_a * delta_a
        + P.CL_delta_r * delta_r
    )
    # The moment arm would be fixed
    # moment_arm = (P.X_AC_WING - P.X_CG) / P.C 
    # CM_wing = neuralfoil(psi, alpha, Re) * moment_arm
    # CM_tail = P.CM_tail_0 + P.CM_tail_alpha * alpha
    m = q_bar * P.S * P.C * (
        P.CM_0                              # Replace with CM_WING
        + P.CM_alpha * alpha                # Replace with CM_TAIL
        + P.CM_q * (P.C / (2 * Va)) * q     
        + P.CM_delta_e * delta_e
        + P.CM_delta_f * delta_f
    )
    # KEEP
    n = q_bar * P.S * P.B * (
        P.CN_0
        + P.CN_beta * beta
        + P.CN_p * (P.B / (2 * Va)) * p
        + P.CN_r * (P.B / (2 * Va)) * r
        + P.CN_delta_a * delta_a
        + P.CN_delta_r * delta_r
    )

    # -----------------Gravity----------------
    g = P.GRAVITY
    m_aircraft = P.MASS

    fx += -m_aircraft * g * ca.sin(theta)
    fy +=  m_aircraft * g * ca.cos(theta) * ca.sin(phi)
    fz +=  m_aircraft * g * ca.cos(theta) * ca.cos(phi)

    # -----------------Equations of motion----------------
    u_dot = fx / m_aircraft + r * v_b - q * w_b
    v_dot = fy / m_aircraft - r * u_b + p * w_b
    w_dot = fz / m_aircraft + q * u_b - p * v_b
    # Rotational
    p_dot = P.Gamma1 * p * q - P.Gamma2 * q * r + P.Gamma3 * l + P.Gamma4 * n
    q_dot = P.Gamma5 * p * r - P.Gamma6 * (p**2 - r**2) + m / P.I_YY
    r_dot = P.Gamma7 * p * q - P.Gamma1 * q * r + P.Gamma4 * l + P.Gamma8 * n

    # ------------------Kinematics----------------
    phi_dot   = p + ca.sin(phi)*ca.tan(theta)*q + ca.cos(phi)*ca.tan(theta)*r
    theta_dot = ca.cos(phi)*q - ca.sin(phi)*r
    psi_dot   = (ca.sin(phi)/ca.cos(theta))*q + (ca.cos(phi)/ca.cos(theta))*r

    # ------------------Position kinematics----------------
    cphi, sphi = ca.cos(phi), ca.sin(phi)
    cth, sth   = ca.cos(theta), ca.sin(theta)
    cpsi, spsi = ca.cos(psi), ca.sin(psi)

    R = ca.vertcat(
        ca.horzcat(cth*cpsi, sphi*sth*cpsi - cphi*spsi, cphi*sth*cpsi + sphi*spsi),
        ca.horzcat(cth*spsi, sphi*sth*spsi + cphi*cpsi, cphi*sth*spsi - sphi*cpsi),
        ca.horzcat(-sth,     sphi*cth,                    cphi*cth)
    )

    vel = ca.vertcat(u_b, v_b, w_b)
    pos_dot = R @ vel

    # ------------------Final Assembly----------------
    x_dot = ca.vertcat(
        u_dot, v_dot, w_dot,
        p_dot, q_dot, r_dot,
        phi_dot, theta_dot, psi_dot,
        pos_dot[0], pos_dot[1], pos_dot[2]
    )

    return ca.Function("f", [x, u], [x_dot])

def RK4(x, u, dt, f=None):

    if f is None:
        f = build_dynamics()

    k1 = f(x, u)
    k2 = f(x + dt/2 * k1, u)
    k3 = f(x + dt/2 * k2, u)
    k4 = f(x + dt * k3, u)

    x_next = x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

    return x_next