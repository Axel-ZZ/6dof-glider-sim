import os
import sys
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from OCP.dynamics_integrator import RK4, build_dynamics
from utils.live_plot import live_plot_3d
from utils.cost_breakdown import CostBreakdown
from utils.side_top_plot import SideTopPlot
from utils.live_plot_mp4 import LivePlot3DMp4Creator
from trajectory.trajectory_guess.f5b import init_trajectory


"""
Added flaps to the 150m four-turn problem.
"""


title = "f5b Four Turns"
path = "plots/f5b/"
path_3D = "f5b/"
mp4_output_dir = r"videos"
warm_start = r"trajectory\warm_start\f5b_four_turns_w_sol_warm_start.npy"
cost_path = r"plots/f5b/f5b_four_turns_cost_breakdown.png"
save_gif = False
save_mp4 = False
save_warm_start = False

# Decrease
N_1 = 60
N_2 = 90
N_3 = 90
N_4 = 90
N_5 = 60

N = N_1 + N_2 + N_3 + N_4 + N_5

dt = 0.1

F = build_dynamics()

# Decision variables
X = ca.MX.sym("X", 12, N + 1)
U = ca.MX.sym("U", 4, N)
T_1 = ca.MX.sym("T_1") # Time to reach waypoint 1
T_2 = ca.MX.sym("T_2") # Time to reach waypoint 2
T_3 = ca.MX.sym("T_3") # Time to reach waypoint 3
T_4 = ca.MX.sym("T_4") # Time to reach waypoint 4
T_5 = ca.MX.sym("T_5") # Time to reach waypoint 5

h1 = T_1 / N_1
h2 = T_2 / N_2
h3 = T_3 / N_3
h4 = T_4 / N_4
h5 = T_5 / N_5

waypoint_1 = np.array([150])
waypoint_2 = np.array([0])
waypoint_3 = np.array([150])
waypoint_4 = np.array([0])
waypoint_5 = np.array([150])

# Use if no warm start is available
# w_north = 2.0
# w_pitch_rate = 8.0
# w_angular = 0.05
# w_altitude_drop = 0.2
# w_terminal_north = 50.0
# w_terminal_angular = 10.0
# w_input_rate = 0.0001
# w_time = 0.50

# Use if warmstart is available
w_north = 0.025
w_angular = 0.000125
w_altitude_drop = 0.1
w_input_rate = 0.00005
w_terminal_north = 50.0
w_terminal_angular = 10.0
w_time = 0.50

# Initialize cost and constraints
cost = 0
g = []
g_rate = []

PN_SCALE = 150.0
PD_SCALE = 10.0


def stage_cost(x, ref):

    ang_rate_penalty = ca.sumsqr(x[3:6])
    north_tracking = ((x[9] - ref[0]) / PN_SCALE)**2

    return w_north * north_tracking + w_angular * ang_rate_penalty
 
def terminal_cost(x, target):

    north_tracking = (x[9] - target[0])**2
    ang_rate_penalty = ca.sumsqr(x[3:6])

    return w_terminal_north * (north_tracking) + w_terminal_angular * ang_rate_penalty

def control_dt(k):
    if k < N_1:
        return h1
    if k < N_1 + N_2:
        return h2
    if k < N_1 + N_2 + N_3:
        return h3
    if k < N_1 + N_2 + N_3 + N_4:
        return h4
    return h5

# Stage costs and dynamics constraints
for k in range(N):
    xk = X[:, k]
    uk = U[:, k]

    if k < N_1:
        tau = (k + 1) / N_1
        ref_north = waypoint_1 * tau
        h = h1
    elif k < N_1 + N_2:
        tau = (k - N_1 + 1) / N_2
        ref_north = (1.0 - tau) * waypoint_1 + tau * waypoint_2
        h = h2
    elif k < N_1 + N_2 + N_3:
        tau = (k - N_1 - N_2 + 1) / N_3
        ref_north = (1.0 - tau) * waypoint_2 + tau * waypoint_3
        h = h3
    elif k < N_1 + N_2 + N_3 + N_4:
        tau = (k - N_1 - N_2 - N_3 + 1) / N_4
        ref_north = (1.0 - tau) * waypoint_3 + tau * waypoint_4
        h = h4
    else:
        tau = (k - N_1 - N_2 - N_3 - N_4 + 1) / N_5
        ref_north = (1.0 - tau) * waypoint_4 + tau * waypoint_5
        h = h5

    x_next = RK4(xk, uk, h, F)
    cost += stage_cost(xk, ref_north)

    if k > 0:
        dt = control_dt(k - 1)
        altitude_drop_penalty = ca.fmax(0, (xk[11] - X[11, k - 1]) / (PD_SCALE * dt))**2
        cost += w_altitude_drop * altitude_drop_penalty

    if k > 0:
        du = U[:, k] - U[:, k - 1]
        dt = control_dt(k - 1)
        du_scaled = ca.vertcat(du[0] * 3.0, du[1] * 4.0, du[2] * 4.0, du[3]  * 3.0)
        cost += w_input_rate * ca.sumsqr(du_scaled / dt)
        g_rate.append(du / dt)

    g.append(X[:, k+1] - x_next)

# add terminal cost
cost += terminal_cost(X[:, N], waypoint_5) # Final target cost at the end of the fifth phase
cost += w_time * (T_1 + T_2 + T_3 + T_4 + T_5) # Penalize total time to encourage faster solutions.

# ─────────────────────────────────────────────────────────────────────────────
#  NLP Assembly
# ─────────────────────────────────────────────────────────────────────────────

# Stack all dynamics constraints into a single vector
g_dynamics = ca.vertcat(*[ca.vec(gi) for gi in g])   # shape: (12*(N_1+N_2+N_3+N_4),)
g_rate_dynamics = ca.vertcat(*[ca.vec(gi) for gi in g_rate])

g_all = ca.vertcat(g_dynamics, g_rate_dynamics)

# Flatten decision variables into a single NLP vector
# Layout: [x0, x1, ..., xN, u0, u1, ..., u_{N-1}]
w       = ca.vertcat(ca.vec(X), ca.vec(U), T_1, T_2, T_3, T_4, T_5)
n_x     = 12 * (N + 1)
n_u     = 4  * (N)

nlp = {
    'x': w,
    'f': cost,
    'g': g_all,
}

# ─────────────────────────────────────────────────────────────────────────────
#  Bounds
# ─────────────────────────────────────────────────────────────────────────────

# Control surface limits (rad)
delta_e_max = np.radians(35.0)
delta_a_max = np.radians(35.0)
delta_r_max = np.radians(35.0)
delta_f_max = np.radians(35.0)
delta_rate_max = np.radians(35.0)

# Lower/upper bounds on w = [X_flat, U_flat]
# States: no explicit bounds (free) — use large numbers
x_lb = -1e4 * np.ones(n_x)
x_ub =  1e4 * np.ones(n_x)

# Controls: [delta_e, delta_a, delta_r, delta_f] repeated N times
u_lb = np.tile([-delta_e_max, -delta_a_max, -delta_r_max, -delta_f_max], N)
u_ub = np.tile([ delta_e_max,  delta_a_max,  delta_r_max,  delta_f_max], N)

t_1_lb = np.array([4.0])
t_1_ub = np.array([10.0])

t_2_lb = np.array([4.0])
t_2_ub = np.array([10.0])

t_3_lb = np.array([4.0])
t_3_ub = np.array([10.0])

t_4_lb = np.array([4.0])
t_4_ub = np.array([10.0])

t_5_lb = np.array([4.0])
t_5_ub = np.array([10.0])

lbw = np.concatenate([x_lb, u_lb, t_1_lb, t_2_lb, t_3_lb, t_4_lb, t_5_lb])
ubw = np.concatenate([x_ub, u_ub, t_1_ub, t_2_ub, t_3_ub, t_4_ub, t_5_ub])

# Waypoint node constraints as direct bounds on decision variables (pn at node k).
idx_wp1 = 9 + 12 * N_1
idx_wp2 = 9 + 12 * (N_1 + N_2)
idx_wp3 = 9 + 12 * (N_1 + N_2 + N_3)
idx_wp4 = 9 + 12 * (N_1 + N_2 + N_3 + N_4)
idx_wp5 = 9 + 12 * (N_1 + N_2 + N_3 + N_4 + N_5)

# Enforce one-sided waypoint crossing constraints.
# Waypoint 1: only a lower bound at the 80 m north plane (pn >= 80).
lbw[idx_wp1] = waypoint_1[0]
# Waypoint 2: only an upper bound at the 20 m north plane (pn <= 20).
ubw[idx_wp2] = waypoint_2[0]
# Waypoint 3: only a lower bound at the 0 m north plane (pn >= 0).
lbw[idx_wp3] = waypoint_3[0]
# Waypoint 4: only an upper bound at the 150 m north plane (pn <= 150).
ubw[idx_wp4] = waypoint_4[0]
# Waypoint 5: only a lower bound at the 150 m north plane (pn >= 150).
lbw[idx_wp5] = waypoint_5[0]


# Dynamics constraints are all equalities.
lbg = np.concatenate([
    np.zeros(12 * (N)),
    -delta_rate_max * np.ones(4 * (N - 1)),
])
ubg = np.concatenate([
    np.zeros(12 * (N)),
    delta_rate_max * np.ones(4 * (N - 1)),
])

# ─────────────────────────────────────────────────────────────────────────────
#  Initial guess (pin x0 to trim-like starting state)
# Use the trajectory\trajectory_guess\f5b.py function to generate new trajectories
# ─────────────────────────────────────────────────────────────────────────────

w0 = np.load(warm_start)

# Pin x0
lbw[:12] = w0[:12] - 1e-6
ubw[:12] = w0[:12] + 1e-6

# Pin u0
lbw[n_x:n_x + 4] = w0[n_x:n_x + 4] - 1e-6
ubw[n_x:n_x + 4] = w0[n_x:n_x + 4] + 1e-6

print("Initial input", w0[n_x:n_x + 4])

# ─────────────────────────────────────────────────────────────────────────────
#  Solve
# ─────────────────────────────────────────────────────────────────────────────

opts = {
    'ipopt.max_iter':        1000,
    'ipopt.tol':             1e-6,
    'ipopt.print_level':     5,
    'print_time':            True,
}

solver = ca.nlpsol('ocp', 'ipopt', nlp, opts)

sol = solver(
    x0=w0,
    lbx=lbw, ubx=ubw,
    lbg=lbg, ubg=ubg,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Extract solution
# ─────────────────────────────────────────────────────────────────────────────

w_sol  = np.array(sol['x']).flatten()

# CasADi vec() is column-major; reconstruct with order='F'.
X_sol = w_sol[:n_x].reshape((12, N + 1), order='F').T  # (N+1, 12)
U_sol = w_sol[n_x:n_x + n_u].reshape((4, N), order='F').T  # (N, 4)
T_1_sol = w_sol[n_x + n_u]
T_2_sol = w_sol[n_x + n_u + 1]
T_3_sol = w_sol[n_x + n_u + 2]
T_4_sol = w_sol[n_x + n_u + 3]
T_5_sol = w_sol[n_x + n_u + 4]
dt1_opt = (T_1_sol) / N_1
dt2_opt = (T_2_sol) / N_2
dt3_opt = (T_3_sol) / N_3
dt4_opt = (T_4_sol) / N_4
dt5_opt = (T_5_sol) / N_5

t_grid = np.linspace(0, T_1_sol + T_2_sol + T_3_sol + T_4_sol + T_5_sol, N + 1)
t_mesh_1 = np.arange(0, N_1 + 1, dtype=float) * dt1_opt
t_mesh_2 = t_mesh_1[-1] + np.arange(1, N_2 + 1, dtype=float) * dt2_opt
t_mesh_3 = t_mesh_2[-1] + np.arange(1, N_3 + 1, dtype=float) * dt3_opt
t_mesh_4 = t_mesh_3[-1] + np.arange(1, N_4 + 1, dtype=float) * dt4_opt
t_mesh_5 = t_mesh_4[-1] + np.arange(1, N_5 + 1, dtype=float) * dt5_opt
t_mesh = np.concatenate([t_mesh_1, t_mesh_2, t_mesh_3, t_mesh_4, t_mesh_5])

# Convenient state extractions
pn_sol    = X_sol[:, 9]
pe_sol    = X_sol[:, 10]
alt_sol   = -X_sol[:, 11] # NOTE: pd is negative down, so take negative for altitude

# ─────────────────────────────────────────────────────────────────────────────
#  Plotting the side and top views with waypoints
# ─────────────────────────────────────────────────────────────────────────────

side_top_plot = SideTopPlot(title=title, output_path=path)
side_top_plot.create_and_save(
    pe_sol=pe_sol,
    pn_sol=pn_sol,
    alt_sol=alt_sol,
    waypoint_1=waypoint_1,
    waypoint_2=waypoint_2,
)

# ────────────────────────────────────────────────────────────────────────────
# Animation of the 3D trajectory with live plotting
# ────────────────────────────────────────────────────────────────────────────

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

# ────────────────────────────────────────────────────────────────────────────
# Animation of the 3D trajectory with live plotting NOTE The Animation is scaled
# ────────────────────────────────────────────────────────────────────────────

if save_mp4:
    mp4_creator = LivePlot3DMp4Creator(title=title, fps=30)
    mp4_creator.create(
        pn=pn_sol,
        pe=pe_sol,
        alt=alt_sol,
        phi=X_sol[:, 6],
        theta=X_sol[:, 7],
        psi=X_sol[:, 8],
        time_mesh=t_mesh,
        path_3d=mp4_output_dir,
    )

# ────────────────────────────────────────────────────────────────────────────
# Cost breakdown
# ────────────────────────────────────────────────────────────────────────────

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

#────────────────────────────────────────────────────────────────────────────
# Total time and time breakdown
#────────────────────────────────────────────────────────────────────────────
total_time = T_1_sol + T_2_sol + T_3_sol + T_4_sol + T_5_sol
distance_traveled = np.sum(np.sqrt(np.diff(pn_sol)**2 + np.diff(pe_sol)**2))
avg_speed = distance_traveled / (T_1_sol + T_2_sol + T_3_sol + T_4_sol + T_5_sol)
print("====================================================================")
print(f"Total distance traveled: {distance_traveled:.2f} m")
print(f"Average speed: {avg_speed:.2f} m/s")
print(f"Total time: {total_time:.2f} seconds")
print(f"Time breakdown:")
print(f"  Phase 1: N = {N_1} , Time: {T_1_sol:.2f} s, dt: {dt1_opt:.3f} s")
print(f"  Phase 2: N = {N_2} , Time: {T_2_sol:.2f} s, dt: {dt2_opt:.3f} s")
print(f"  Phase 3: N = {N_3} , Time: {T_3_sol:.2f} s, dt: {dt3_opt:.3f} s")
print(f"  Phase 4: N = {N_4} , Time: {T_4_sol:.2f} s, dt: {dt4_opt:.3f} s")
print(f"  Phase 5: N = {N_5} , Time: {T_5_sol:.2f} s, dt: {dt5_opt:.3f} s")
print("====================================================================")

# Save w_sol for future warm start
if save_warm_start:
    np.save(r"trajectory\warm_start\f5b_four_turns_w_sol_warm_start.npy", w_sol)

# Save for analysis
np.save(r"analysis\f5b_four_turns_w_sol_analysis.npy", w_sol)

plt.show()