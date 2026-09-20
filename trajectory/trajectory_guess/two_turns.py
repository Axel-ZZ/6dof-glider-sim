import numpy as np

N_1 = 60
N_2 = 90
N_3 = 60

waypoint_1 = np.array([150])
waypoint_2 = np.array([0])
waypoint_3 = np.array([150])

# Trim at Va = 25 m/s
alpha0  = np.radians(2.0)
Va0     = 25.0

x_guess = np.zeros(12 * (N_1 + N_2 + N_3 + 1))

pe_amp = 20.0  # lateral extent of each turn [m]

for k in range(N_1 + N_2 + N_3 + 1):
    if k < N_1:
        # Phase 1: fly north from pn=0 to pn=80
        tau = k / N_1
        pn_k = waypoint_1[0] * tau
        pe_k = -pe_amp/3 * np.sin(np.pi * tau)
        alt_k = np.interp(tau, [0, 1], [150.0, 140.0])
        psi_k = 0.0
    elif k < N_1 + N_2:
        # Phase 2: first turn — pn 80→20, sweeping east, heading rotates north→east→south
        tau = (k - N_1) / N_2
        pn_k = waypoint_1[0] - (waypoint_1[0] - waypoint_2[0]) * tau
        pe_k = pe_amp * np.sin(np.pi * tau)
        alt_k = np.interp(tau, [0, 1], [140.0, 130.0])
        psi_k = np.pi * tau
    else:
        # Phase 3: second turn — pn 20→80, sweeping west, heading rotates south→west→north
        tau = (k - N_1 - N_2) / N_3
        pn_k = waypoint_2[0] + (waypoint_3[0] - waypoint_2[0]) * tau
        pe_k = -pe_amp * np.sin(np.pi * tau)
        alt_k = np.interp(tau, [0, 1], [130.0, 120.0])
        psi_k = np.pi + np.pi * tau

    x_guess[12*k : 12*k+12] = [
        Va0*np.cos(alpha0), 0, Va0*np.sin(alpha0),  # u,v,w
        0, 0, 0,                                            # p,q,r
        0, alpha0, psi_k,                               # phi,theta,psi
        pn_k, pe_k, -alt_k                                  # pn,pe,pd
    ]