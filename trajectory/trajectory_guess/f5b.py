import numpy as np
import matplotlib.pyplot as plt

def init_trajectory(nodes: list[int], waypoints: list[float], 
                   altitude_profile: list[float] = None,
                   pe_amp: float = 20.0, alpha0: float = 2.0, Va0: float = 25.0):
    """
    Generate an initial trajectory guess across multiple phases.
    
    Parameters
    ----------
    nodes : list[int]
        Number of discretization steps per phase. len(nodes) defines the number of phases.
    waypoints : list[float]
        North position at each phase boundary (length = len(nodes) + 1 or len(nodes)).
        If length == len(nodes), the first waypoint is implicitly 0.0.
    altitude_profile : list[float], optional
        Altitude at each phase boundary (length = len(nodes) + 1).
        If None, linearly interpolates from 150 m to 100 m.
    pe_amp : float
        Lateral (east) amplitude for sinusoidal turns. Default: 20.0 m.
    alpha0 : float
        Angle of attack (rad). Default: 2.0 rad.
    Va0 : float
        Airspeed (m/s). Default: 25.0 m/s.
    
    Returns
    -------
    x_guess : np.ndarray
        State guess of shape (12 * (N + 1),) where N = sum(nodes).
    """
    nodes = np.asarray(nodes, dtype=int)
    waypoints = np.asarray(waypoints, dtype=float).flatten()
    n_phases = len(nodes)
    n_nodes = int(np.sum(nodes))
    
    # Handle waypoints: if not provided explicitly at start, insert implicit 0.0
    if len(waypoints) == n_phases:
        waypoints = np.concatenate(([0.0], waypoints))
    elif len(waypoints) != n_phases + 1:
        raise ValueError(
            f"waypoints must have length {n_phases} or {n_phases + 1}; got {len(waypoints)}"
        )
    
    # Default altitude profile: linear from 150 m to 100 m
    if altitude_profile is None:
        altitude_profile = np.linspace(150.0, 100.0, n_phases + 1)
    else:
        altitude_profile = np.asarray(altitude_profile, dtype=float).flatten()
        if len(altitude_profile) != n_phases + 1:
            raise ValueError(
                f"altitude_profile must have length {n_phases + 1}; got {len(altitude_profile)}"
            )
    
    x_guess = np.zeros(12 * (n_nodes + 1))
    
    cumulative_nodes = np.cumsum([0] + list(nodes))
    
    for k in range(n_nodes + 1):
        # Determine which phase k belongs to
        phase_idx = n_phases - 1  # default to last phase (handles terminal node)
        for i, cum_k in enumerate(cumulative_nodes[:-1]):
            if k >= cum_k and k < cumulative_nodes[i + 1]:
                phase_idx = i
                break
        
        # Local node index within the phase
        k_local = k - cumulative_nodes[phase_idx]
        N_phase = nodes[phase_idx]
        tau = k_local / N_phase
        
        # North position: linear interpolation from start to end waypoint of phase
        wp_start = waypoints[phase_idx]
        wp_end = waypoints[phase_idx + 1]
        pn_k = (1.0 - tau) * wp_start + tau * wp_end
        
        # Altitude: linear interpolation across phase
        alt_start = altitude_profile[phase_idx]
        alt_end = altitude_profile[phase_idx + 1]
        alt_k = (1.0 - tau) * alt_start + tau * alt_end
        
        # East position: alternating sine waves for turns (odd phases turn east, even turn west)
        # First phase
        if phase_idx == 0:
            pe_k = pe_amp/3 * np.sin(np.pi * tau)
        elif phase_idx % 2 == 0:
            # Even phase: turn east (positive sine)
            pe_k = pe_amp * np.sin(np.pi * tau)
        else:
            # Odd phase: turn west (negative sine)
            pe_k = -pe_amp * np.sin(np.pi * tau)
        
        # Yaw: accumulate heading rotation across phases
        psi_k = phase_idx * np.pi + np.pi * tau
        
        x_guess[12*k : 12*k+12] = [
            Va0*np.cos(alpha0), 0, Va0*np.sin(alpha0),  # u,v,w
            0, 0, 0,                                     # p,q,r
            0, alpha0, psi_k,                            # phi,theta,psi
            pn_k, pe_k, -alt_k                           # pn,pe,pd (pd is negative down)
        ]
    
    return x_guess


# N_1 = 60
# N_2 = 90
# N_3 = 60

# waypoint_1 = np.array([150])
# waypoint_2 = np.array([0])
# waypoint_3 = np.array([150])

# trajectory = init_trajectory([N_1, N_2, N_3], [waypoint_1, waypoint_2, waypoint_3])

# # 3D plot of the trajectory
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(trajectory[9::12], trajectory[10::12], -trajectory[11::12])  # pn, pe, alt
# ax.set_xlabel('North (m)')
# ax.set_ylabel('East (m)')
# ax.set_zlabel('Altitude (m)')
# ax.set_title('Initial Trajectory Guess')
# ax.grid()
# plt.show()