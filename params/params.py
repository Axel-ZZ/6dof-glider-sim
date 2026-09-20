import numpy as np

"""
_delta_ - control derivatives
Based on M. Beard and T. McLain, "Small Unmanned Aircraft: Theory and Practice"
Aero coefficients from their github:
https://github.com/byu-magicc/mavsim_public.git
mavsim_python/parameters/aerosonde_parameters.py
"""


# ─────────────────────────────────────────
#  Physical constants
# ─────────────────────────────────────────
GRAVITY = 9.81          # m/s^2
RHO     = 1.2682        # kg/m^3

# ─────────────────────────────────────────
#  Inertial parameters  (F5B-like glider)
# ─────────────────────────────────────────
MASS = 11.0             # kg
I_XX = 0.8244           # kg·m²  (roll)
I_YY = 1.135            # kg·m²  (pitch)
I_ZZ = 1.759            # kg·m²  (yaw)
I_XZ = 0.1204           # kg·m²  (cross product of inertia)

# ── Rotational dynamics (Beard & McLain, symmetric aircraft Ixy=Iyz=0) ───
Gamma = I_XX * I_ZZ - I_XZ**2
Gamma1 = I_XZ * (I_XX - I_YY + I_ZZ) / Gamma
Gamma2 = (I_ZZ * (I_ZZ - I_YY) + I_XZ**2) / Gamma
Gamma3 = I_ZZ / Gamma
Gamma4 = I_XZ / Gamma
Gamma5 = (I_ZZ - I_XX) / I_YY
Gamma6 = I_XZ / I_YY
Gamma7 = ((I_XX - I_YY) * I_XX + I_XZ**2) / Gamma
Gamma8 = I_XX / Gamma

# ─────────────────────────────────────────
#  Geometric parameters
# ─────────────────────────────────────────
S    = 0.55             # m²   wing reference area
B    = 2.8956           # m    wingspan
C    = 0.18994          # m    mean aerodynamic chord
AR   = B**2 / S         # aspect ratio
E    = 0.9              # Oswald efficiency factor

# ─────────────────────────────────────────
#  Stall model parameters (B&M eq. 4.10)
# ─────────────────────────────────────────
M_SIGMOID    = 50.0                     # sigmoid transition rate
ALPHA0_STALL = 0.47                     # stall onset angle (rad)

# ─────────────────────────────────────────
#  Longitudinal aerodynamic coefficients
# ─────────────────────────────────────────
CL_0          =  0.23   # lift at zero AoA
CL_alpha      =  5.61   # lift curve slope (1/rad)
CL_q          =  7.95   # pitch rate contribution to lift
CL_delta_e    =  0.13   # elevator contribution to lift

CD_0          =  0.0424 # drag at zero AoA
CD_alpha      =  0.132  # drag slope vs AoA
CD_p          =  0.043  # parasitic drag (zero-lift drag)
CD_q          =  0.0    # pitch rate contribution to drag
CD_delta_e    =  0.0135 # elevator contribution to drag

CM_0          =  0.0135 # pitching moment at zero AoA
CM_alpha      = -2.74   # longitudinal static stability (must be < 0)
CM_q          = -38.21  # pitch damping derivative
CM_delta_e    = -0.99   # elevator control power (negative by convention)

# ─────────────────────────────────────────
#  Lateral-directional aerodynamic coefficients
# ─────────────────────────────────────────
CY_0          =  0.00   # side force at zero sideslip (zero for symmetric a/c)
CY_beta       = -0.98   # side force due to sideslip
CY_p          =  0.0    # side force due to roll rate
CY_r          =  0.0    # side force due to yaw rate
CY_delta_a    =  0.075  # aileron contribution to side force
CY_delta_r    =  0.19   # rudder contribution to side force

CL_0_moment   =  0.00   # roll moment at zero (zero for symmetric a/c)
CL_beta       = -0.13   # dihedral effect (roll stability) REALLY LOW APPARENTLY
CL_p          = -0.51   # roll damping
CL_r          =  0.25   # roll due to yaw rate
CL_delta_a    =  0.17   # aileron control power
CL_delta_r    =  0.0024 # rudder contribution to roll

CN_0          =  0.00   # yaw moment at zero (zero for symmetric a/c)
CN_beta       =  0.073  # weathercock stability (must be > 0)
CN_p          =  0.069  # yaw due to roll rate
CN_r          = -0.095  # yaw damping
CN_delta_a    = -0.011  # adverse yaw from aileron
CN_delta_r    = -0.069  # rudder control power

# ─────────────────────────────────────────
#  Propulsion parameters (B&M eq. 4.15 / 4.16)
# ─────────────────────────────────────────
S_PROP  = 0.2027        # m²   propeller disk area
C_PROP  = 1.0           # propeller efficiency constant
K_MOTOR = 40.0          # motor constant: Vexit = K_MOTOR * delta_t  (m/s)
K_TP    = 1e-4          # propeller torque constant
K_OMEGA = 150.0         # propeller speed constant (rad/s per throttle unit)

# ─────────────────────────────────────────
#  Flap parameters (for future 4-control OCP)
# ─────────────────────────────────────────
# Sign convention: positive flap means trailing-edge down.
# Longitudinal flap control derivatives (same naming convention as delta_e)
CL_delta_f    =  0.13   # flap contribution to lift
CD_delta_f    =  0.0135   # flap contribution to drag
CM_delta_f    = -0.99   # flap contribution to pitching moment