from os import path

import numpy as np
import matplotlib.pyplot as plt


class CostBreakdown:
    """
    Evaluates and prints a cost breakdown table for the f5b OCP solution.

    Usage:
        from cost_breakdown import CostBreakdown

        cb = CostBreakdown(
            X_sol,
            phase_times,
            nodes,
            waypoints,
            w_north, w_angular, w_altitude_drop,
            w_terminal_north, w_terminal_angular, w_time,
        )
        cb.print()

    If `waypoints` does not include the start waypoint, the initial north
    position from `X_sol[0, 9]` is used as the implicit starting reference.
    """

    # ── Cost weights (mirror your stage_cost exactly) ──────────────────────
    PN_SCALE     = 150.0
    PD_SCALE     = 10.0
    D_REF        = -150.0

    def __init__(self, X_sol, phase_times: list[float],
                 nodes: list[int],
                 waypoints: list[float],
                 w_north, w_angular, w_altitude_drop, w_terminal_north, w_terminal_angular, w_time,
                 U_sol=None, w_input_rate=0.0, path=None):
        self.X_sol = np.asarray(X_sol)
        self.U_sol = None if U_sol is None else np.asarray(U_sol)
        self.phase_times = np.asarray(phase_times, dtype=float).reshape(-1)
        self.nodes = np.asarray(nodes, dtype=int).reshape(-1)
        self.wp = np.asarray(waypoints, dtype=float).reshape(-1)
        self.w_north = w_north
        self.w_angular = w_angular
        self.w_altitude_drop = w_altitude_drop
        self.w_terminal_north = w_terminal_north
        self.w_terminal_angular = w_terminal_angular
        self.w_time = w_time
        self.w_input_rate = w_input_rate
        self.path = path

        self.n_phases = int(self.nodes.size)
        self.n_nodes = int(np.sum(self.nodes))
        self.n_states = self.n_nodes + 1

        self._validate_inputs()

    def _validate_inputs(self):
        if self.phase_times.size != self.n_phases:
            raise ValueError(
                f"phase_times has length {self.phase_times.size}, but nodes has length {self.n_phases}."
            )

        if self.wp.size not in {self.n_phases, self.n_phases + 1}:
            raise ValueError(
                "waypoints must contain either one waypoint per phase endpoint "
                f"({self.n_phases}) or an explicit start plus phase endpoints ({self.n_phases + 1})."
            )

        if self.X_sol.ndim != 2 or self.X_sol.shape[1] != 12:
            raise ValueError(f"X_sol must have shape (N+1, 12); got {self.X_sol.shape}.")

        if self.X_sol.shape[0] != self.n_states:
            raise ValueError(
                f"X_sol has {self.X_sol.shape[0]} state nodes, but nodes sum to {self.n_nodes} "
                f"which requires {self.n_states} states."
            )

        if self.U_sol is not None and (self.U_sol.ndim != 2 or self.U_sol.shape[0] != self.n_nodes):
            if self.U_sol.ndim == 2 and self.U_sol.shape[1] == self.n_nodes and self.U_sol.shape[0] == 4:
                self.U_sol = self.U_sol.T
            else:
                raise ValueError(
                    f"U_sol must have shape (N, 4) with N={self.n_nodes}; got {self.U_sol.shape}."
                )

    def _reference_waypoints(self):
        if self.wp.size == self.n_phases + 1:
            return self.wp

        start_north = float(self.X_sol[0, 9])
        return np.concatenate(([start_north], self.wp))

    def _phase_index_for_k(self, k):
        cumulative = 0
        for phase_index, phase_nodes in enumerate(self.nodes):
            if k < cumulative + phase_nodes:
                return phase_index, k - cumulative
            cumulative += phase_nodes

        return self.n_phases - 1, self.nodes[-1] - 1

    def _phase_boundaries(self):
        return np.cumsum(self.phase_times)
    # ── Internal helpers ───────────────────────────────────────────────────

    @staticmethod
    def _early_switch_tau(tau, switch_point=0.15, sharpness=1.0):
        t  = float(np.clip(tau, 0.0, 1.0))
        s  = 1.0 / (1.0 + np.exp(-sharpness * (t            - switch_point)))
        s0 = 1.0 / (1.0 + np.exp(-sharpness * (0.0          - switch_point)))
        s1 = 1.0 / (1.0 + np.exp(-sharpness * (1.0          - switch_point)))
        u  = np.clip((s - s0) / (s1 - s0 + 1e-12), 0.0, 1.0)
        return 6.0 * u**5 - 15.0 * u**4 + 10.0 * u**3

    def _ref_north(self, k):
        if k >= self.n_nodes:
            return float(self._reference_waypoints()[-1])

        waypoints = self._reference_waypoints()
        cumulative = 0
        for i, phase_nodes in enumerate(self.nodes):
            if k < cumulative + phase_nodes:
                tau = (k - cumulative + 1) / phase_nodes
                wp_start = waypoints[i]
                wp_end = waypoints[i + 1]
                return (1.0 - tau) * wp_start + tau * wp_end
            cumulative += phase_nodes

        return float(waypoints[-1])

    def _control_dt(self, k):
        cumulative = 0
        for phase_time, phase_nodes in zip(self.phase_times, self.nodes):
            if k < cumulative + phase_nodes:
                return phase_time / phase_nodes
            cumulative += phase_nodes

        return self.phase_times[-1] / self.nodes[-1]

    def _control_rate_cost(self, k):
        if self.U_sol is None or k == 0:
            return 0.0
        du = self.U_sol[k] - self.U_sol[k - 1]
        dt = self._control_dt(k - 1)
        return self.w_input_rate * np.sum((du / dt) ** 2)

    def compute(self):
        c_north = c_ang = c_alt = c_rate = 0.0
        N = self.n_nodes

        for k in range(N):
            x   = self.X_sol[k]
            ref = self._ref_north(k)

            phi, q, r, p = x[6], x[4], x[5], x[3]
            pr = np.cos(phi) * q - np.sin(phi) * r

            c_north += self.w_north      * ((x[9]  - ref)          / self.PN_SCALE)**2
            c_ang   += self.w_angular   * (p**2 + q**2 + r**2)
            # Altitude drop penalty (one-sided, only descent k > 0)
            if k > 0:
                dt = self._control_dt(k - 1)
                drop = max(0.0, (x[11] - self.X_sol[k-1, 11]) / (self.PD_SCALE * dt))
                c_alt += self.w_altitude_drop * drop**2
            c_rate  += self._control_rate_cost(k)

        p_f, q_f, r_f = self.X_sol[-1, 3], self.X_sol[-1, 4], self.X_sol[-1, 5]
        c_terminal = self.w_terminal_north * (self.X_sol[-1, 9] - self._reference_waypoints()[-1])**2
        c_terminal_ang = self.w_terminal_angular * (p_f**2 + q_f**2 + r_f**2)
        c_time     = self.w_time     * float(np.sum(self.phase_times))
        c_total    = c_north + c_ang + c_alt + c_rate + c_terminal + c_terminal_ang + c_time

        return dict(
            north    = c_north,
            ang      = c_ang,
            alt      = c_alt,
            control_rate = c_rate,
            terminal = c_terminal,
            terminal_ang = c_terminal_ang,
            time     = c_time,
            total    = c_total,
        )

    def print(self):
        c  = self.compute()
        T  = float(np.sum(self.phase_times))
        W  = 46

        rows = [
            ("North tracking",  c["north"]),
            ("Angular rate",    c["ang"]),
            ("Altitude",        c["alt"]),
            ("Control rate",    c["control_rate"]),
            ("Terminal north",  c["terminal"]),
            ("Terminal ang rate", c["terminal_ang"]),
            ("Time penalty",    c["time"]),
        ]

        print("\n" + "─" * W)
        print(f"  {'COST BREAKDOWN':^{W-4}}")
        print("─" * W)
        print(f"  {'Term':<20} {'Cost':>8}   {'Share':>7}")
        print("─" * W)
        for name, val in rows:
            bar = "█" * int(round(20 * val / c["total"]))
            print(f"  {name:<20} {val:>8.4f}   {100*val/c['total']:>5.1f}%  {bar}")
        print("─" * W)
        print(f"  {'TOTAL':<20} {c['total']:>8.4f}   {'100.0%':>7}")
        print("─" * W)
        phase_times_str = "  ".join(
            f"T{i + 1}={phase_time:.2f}s" for i, phase_time in enumerate(self.phase_times)
        )
        print(f"  {phase_times_str}  Total={T:.2f}s")
        print("─" * W + "\n")
    
    def plot_costs_over_time(self):
        N = self.n_nodes

        c_north = np.zeros(N)
        c_ang   = np.zeros(N)
        c_alt   = np.zeros(N)
        c_rate  = np.zeros(N)
        c_stage = np.zeros(N)

        for k in range(N):
            x   = self.X_sol[k]
            ref = self._ref_north(k)

            q, r, p = x[4], x[5], x[3]

            c_north[k] = self.w_north * ((x[9] - ref) / self.PN_SCALE)**2
            c_ang[k]   = self.w_angular * (p**2 + q**2 + r**2)
            # Altitude drop penalty (one-sided, only descent)
            if k > 0:
                dt = self._control_dt(k - 1)
                drop = max(0.0, (x[11] - self.X_sol[k-1, 11]) / (self.PD_SCALE * dt))
                c_alt[k] = self.w_altitude_drop * drop**2
            c_rate[k]  = self._control_rate_cost(k)
            c_stage[k] = c_north[k] + c_ang[k] + c_alt[k] + c_rate[k]

        p_f, q_f, r_f = self.X_sol[-1, 3], self.X_sol[-1, 4], self.X_sol[-1, 5]
        c_terminal = self.w_terminal_north * (self.X_sol[-1, 9] - self._reference_waypoints()[-1])**2
        c_terminal_ang = self.w_terminal_angular * (p_f**2 + q_f**2 + r_f**2)
        c_time = self.w_time * float(np.sum(self.phase_times))

        t0 = 0.0
        t = np.linspace(0.0, float(np.sum(self.phase_times)), N)

        cumulative_stage = np.cumsum(c_stage)
        total_cumulative = cumulative_stage[-1] + c_terminal + c_terminal_ang + c_time

        phase_boundaries = self._phase_boundaries()[:-1]

        fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

        ax = axes[0]
        ax.plot(t, c_north, label="North", linewidth=1.6)
        ax.plot(t, c_ang, label="Angular rate", linewidth=1.6)
        ax.plot(t, c_alt, label="Altitude", linewidth=1.6)
        ax.plot(t, c_rate, label="Control rate", linewidth=1.6)
        ax.plot(t, c_stage, label="Stage total", linewidth=2.0, color="black", alpha=0.85)
        for boundary in phase_boundaries:
            ax.axvline(boundary, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.set_ylabel("Stage cost")
        ax.set_title("Cost terms over time")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=2)

        ax2 = axes[1]
        ax2.plot(t, cumulative_stage, label="Cumulative stage cost", linewidth=2.0)
        ax2.axhline(cumulative_stage[-1] + c_terminal, color="tab:orange", linestyle=":", linewidth=1.6, label="+ terminal")
        ax2.axhline(cumulative_stage[-1] + c_terminal + c_terminal_ang, color="tab:green", linestyle=":", linewidth=1.6, label="+ terminal ang")
        ax2.axhline(total_cumulative, color="tab:red", linestyle="--", linewidth=1.8, label="Total (+ time)")
        for boundary in phase_boundaries:
            ax2.axvline(boundary, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Accumulated cost")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper left")

        fig.tight_layout()
        if self.path:
            fig.savefig(self.path, dpi=300)
        return fig, axes
