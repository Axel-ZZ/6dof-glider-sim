import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.cost_breakdown import CostBreakdown
from utils.export_czml import export_czml
from utils.live_plot import live_plot_3d


# Read the data from the file
file = r"analysis\f5b_four_turns_w_sol_analysis.npy"
cost_path = r"plots/f5b/f5b_four_turns_cost_breakdown.png"
title = "f5b Four Turns gif side view"
path = "plots/f5b/"
path_3D = "f5b/"
w_sol = np.load(file)
w_sol  = np.array(w_sol).flatten()
save_angels = False
save_controls = False
cost_analysis = True
create_czml = False
save_gif = False

N_1 = 60
N_2 = 90
N_3 = 90
N_4 = 90
N_5 = 60
N_total = N_1 + N_2 + N_3 + N_4 + N_5
N = N_total

waypoint_1 = np.array([150])
waypoint_2 = np.array([0])
waypoint_3 = np.array([150])
waypoint_4 = np.array([0])
waypoint_5 = np.array([150])

# Use if warmstart is available
w_north = 0.1 / 4
w_angular = 0.025 / (2 * 100)
w_altitude_drop = 0.2 / 2
w_input_rate = 0.001 / (2 * 10) # 0.00005
w_terminal_north = 50.0
w_terminal_angular = 10.0
w_time = 0.50

N_u = np.arange(N_total)
N_x = np.arange(N_total + 1)

n_x     = 12 * (N_total + 1)
n_u     = 4  * N_total

# Extract the relevant data
X_sol = w_sol[:n_x].reshape((12, N_total + 1), order='F').T  # (N+1, 12)
U_sol = w_sol[n_x:n_x + n_u].reshape((4, N_total), order='F').T  # (N, 4)

T_1_sol = w_sol[n_x + n_u]
T_2_sol = w_sol[n_x + n_u + 1]
T_3_sol = w_sol[n_x + n_u + 2]
T_4_sol = w_sol[n_x + n_u + 3]
T_5_sol = w_sol[n_x + n_u + 4]

total_time = T_1_sol + T_2_sol + T_3_sol + T_4_sol + T_5_sol

dt1_opt = (T_1_sol) / N_1
dt2_opt = (T_2_sol) / N_2
dt3_opt = (T_3_sol) / N_3
dt4_opt = (T_4_sol) / N_4
dt5_opt = (T_5_sol) / N_5

p = X_sol[:, 3]
q = X_sol[:, 4]
r = X_sol[:, 5]

phi_sol   = X_sol[:, 6]
theta_sol = X_sol[:, 7]
psi_sol   = X_sol[:, 8]
pn_sol    = X_sol[:, 9]
pe_sol    = X_sol[:, 10]
alt_sol   = -X_sol[:, 11]

# Controls in the OCP are ordered as [delta_e, delta_a, delta_r, delta_f]
elevator_sol = U_sol[:, 0]
aileron_sol = U_sol[:, 1]
rudder_sol = U_sol[:, 2]
flap_sol = U_sol[:, 3]

live_plot_3d(
    pn_sol,
    pe_sol,
    alt_sol,
    X_sol[:, 6],
    X_sol[:, 7],
    X_sol[:, 8],
    title=title,
    save_animation=save_gif,
    sample_time= (dt1_opt + dt2_opt + dt3_opt + dt4_opt + dt5_opt) / (N),
    path_3D=path_3D,
    stl_scale=10.0,
)
plt.show()

# print(f"dt1_opt: {dt1_opt:.4f} s, dt2_opt: {dt2_opt:.4f} s, dt3_opt: {dt3_opt:.4f} s, dt4_opt: {dt4_opt:.4f} s, dt5_opt: {dt5_opt:.4f} s")

# t_grid = np.linspace(0, T_1_sol + T_2_sol + T_3_sol + T_4_sol + T_5_sol, N_total + 1)
# t_mesh_1 = np.arange(0, N_1 + 1, dtype=float) * dt1_opt
# t_mesh_2 = t_mesh_1[-1] + np.arange(1, N_2 + 1, dtype=float) * dt2_opt
# t_mesh_3 = t_mesh_2[-1] + np.arange(1, N_3 + 1, dtype=float) * dt3_opt
# t_mesh_4 = t_mesh_3[-1] + np.arange(1, N_4 + 1, dtype=float) * dt4_opt
# t_mesh_5 = t_mesh_4[-1] + np.arange(1, N_5 + 1, dtype=float) * dt5_opt
# t_mesh = np.concatenate([t_mesh_1, t_mesh_2, t_mesh_3, t_mesh_4, t_mesh_5])

# plt.plot(p, label='p')
# plt.plot(q, label='q')
# plt.plot(r, label='r')
# plt.xlabel('Time step')
# plt.ylabel('Angular rates [rad/s]')
# plt.title('Optimal angular rates over time')
# plt.legend()
# plt.grid()
# plt.show()

# # Plot the angels
# plt.plot(N_x, np.degrees(phi_sol), label='phi')
# plt.plot(N_x, np.degrees(theta_sol), label='theta')
# plt.plot(N_x, np.degrees(psi_sol), label='psi')

# plt.hlines(90, xmin=0, xmax=N_total, colors='gray', linestyles='dashed', label='pi / 2')
# plt.vlines([N_1, N_1 + N_2, N_1 + N_2 + N_3, N_1 + N_2 + N_3 + N_4, N_total], ymin=-5, ymax=720, colors='gray', linestyles='dashed', label='Phase transitions')
# plt.xlabel('Time step')
# plt.ylabel('Angle [degrees]')
# plt.title('Optimal angles over time')
# plt.legend()
# plt.grid()
# if save_angels:
#     plt.savefig(r"analysis\four_turns_angles.png", dpi=300)
# plt.show()

# # plot controls
# plt.plot(N_u, np.degrees(elevator_sol), label='elevator')
# plt.plot(N_u, np.degrees(flap_sol), label='flap')
# plt.plot(N_u, np.degrees(rudder_sol), marker='o', markersize=3, label='rudder')
# plt.plot(N_u, np.degrees(aileron_sol), marker='o', markersize=3, label='aileron')

# plt.vlines([N_1, N_1 + N_2, N_1 + N_2 + N_3, N_1 + N_2 + N_3 + N_4, N_total], ymin=-40, ymax=40, colors='gray', linestyles='dashed', label='Phase transitions')
# plt.xlabel('Time step')
# plt.ylabel('Control input [degrees]')
# plt.ylim(-40, 40)
# plt.title('Optimal control inputs over time')
# plt.legend()
# plt.grid()
# if save_controls:
#     plt.savefig(r"analysis\four_turns_controls.png", dpi=300)
# plt.show()

# # Calculate the total distance traveled
# distance = np.sqrt(np.diff(pn_sol)**2 + np.diff(pe_sol)**2 + np.diff(alt_sol)**2)
# distance_traveled = np.sum(distance)

# speed = distance / np.diff(t_mesh)

# plt.plot(N_u, speed, label='speed')
# plt.vlines([N_1, N_1 + N_2, N_1 + N_2 + N_3, N_1 + N_2 + N_3 + N_4, N_total], ymin=21, ymax=26, colors='gray', linestyles='dashed', label='Phase transitions')
# plt.xlabel('Time step')
# plt.ylabel('Speed [m/s]')
# plt.title('Optimal speed over time')
# plt.legend()
# plt.grid()
# plt.savefig(r"analysis\four_turns_speed.png", dpi=300)
# plt.show()

# print(f"Total distance traveled: {distance_traveled:.2f} m")

# avg_speed = distance_traveled / (T_1_sol + T_2_sol + T_3_sol + T_4_sol + T_5_sol)
# print(f"Average speed: {avg_speed:.2f} m/s")

if cost_analysis:
    cost_breakdown = CostBreakdown(
        X_sol,
        [T_1_sol, T_2_sol, T_3_sol, T_4_sol, T_5_sol],
        [N_1, N_2, N_3, N_4, N_5],
        [waypoint_1, waypoint_2, waypoint_3, waypoint_4, waypoint_5],
        w_north, w_angular, w_altitude_drop, w_terminal_north, w_terminal_angular, w_time,
        U_sol=U_sol, w_input_rate=w_input_rate, path=cost_path
    )
    cost_breakdown.print()
    fig, axes = cost_breakdown.plot_costs_over_time()
    plt.show()

# if create_czml:
#     export_czml(
#         pn=pn_sol,
#         pe=pe_sol,
#         alt=alt_sol,
#         phi=X_sol[:, 6],
#         theta=X_sol[:, 7],
#         psi=X_sol[:, 8],
#         t_mesh=t_mesh,
#         output_path="trajectory_four_turns.czml",
#         label="F5B Four Turn Slalom",
#         waypoints=[0.0, 150.0],   # your two waypoint planes in pn [m]
#     )

# print("====================================================================")
# print(f"Total distance traveled: {distance_traveled:.2f} m")
# print(f"Average speed: {avg_speed:.2f} m/s")
# print(f"Total time: {total_time:.2f} seconds")
# print(f"Time breakdown:")
# print(f"  Phase 1: N = {N_1} , Time: {T_1_sol:.2f} s, dt: {dt1_opt:.3f} s")
# print(f"  Phase 2: N = {N_2} , Time: {T_2_sol:.2f} s, dt: {dt2_opt:.3f} s")
# print(f"  Phase 3: N = {N_3} , Time: {T_3_sol:.2f} s, dt: {dt3_opt:.3f} s")
# print(f"  Phase 4: N = {N_4} , Time: {T_4_sol:.2f} s, dt: {dt4_opt:.3f} s")
# print(f"  Phase 5: N = {N_5} , Time: {T_5_sol:.2f} s, dt: {dt5_opt:.3f} s")
# print("====================================================================")

# print(f"Last altitude: {alt_sol[-1]:.2f} m ")

# # Plot the altitude profile
# plt.plot(N_x, alt_sol, label='Altitude')
# plt.xlabel('Time step')
# plt.ylabel('Altitude [m]')
# plt.title('Optimal altitude profile over time')
# plt.vlines([N_1, N_1 + N_2, N_1 + N_2 + N_3, N_1 + N_2 + N_3 + N_4, N_total], ymin=0, ymax=alt_sol[0] + 10, colors='gray', linestyles='dashed', label='Phase transitions')
# plt.legend()
# plt.grid()
# plt.savefig(r"analysis\four_turns_altitude.png", dpi=300)
# plt.show()