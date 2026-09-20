import numpy as np
import matplotlib.pyplot as plt
import os


class SideTopPlot:
	def __init__(self, title, output_path):
		self.title = title
		self.output_path = output_path

	def create_and_save(self, pe_sol, pn_sol, alt_sol, waypoint_1, waypoint_2):
		fig_simple = plt.figure(figsize=(7, 5))
		ax_simple = fig_simple.add_subplot(111, projection='3d')
		ax_simple.plot(pe_sol, pn_sol, alt_sol, color='royalblue', linewidth=2.0, label='Trajectory')
		ax_simple.scatter(pe_sol[0], pn_sol[0], alt_sol[0], color='green', s=40, label='Start')
		ax_simple.scatter(pe_sol[-1], pn_sol[-1], alt_sol[-1], color='black', s=40, label='End')
		ax_simple.set_xlabel('East [m]')
		ax_simple.set_ylabel('North [m]')
		ax_simple.set_zlabel('Altitude [m]')
		ax_simple.grid(True, alpha=0.25)

		# Plot north-position planes as constant-north surfaces.
		pe_plane = np.linspace(np.min(pe_sol) - 15, np.max(pe_sol) + 15, 15)
		alt_plane = np.linspace(np.min(alt_sol) - 10, np.max(alt_sol) + 10, 15)
		pe_grid, alt_grid = np.meshgrid(pe_plane, alt_plane)

		waypoint_1_plane = np.full_like(pe_grid, waypoint_1[0])
		ax_simple.plot_surface(pe_grid, waypoint_1_plane, alt_grid, color='orange', alpha=0.25, label='Plane pn=80')
		waypoint_2_plane = np.full_like(pe_grid, waypoint_2[0])
		ax_simple.plot_surface(pe_grid, waypoint_2_plane, alt_grid, color='gold', alpha=0.25, label='Plane pn=20')

		pe_min = np.min(pe_sol) - 15
		pe_max = np.max(pe_sol) + 15
		pn_min = min(np.min(pn_sol), waypoint_1[0], waypoint_2[0]) - 15
		pn_max = max(np.max(pn_sol), waypoint_1[0], waypoint_2[0]) + 15
		alt_min = np.min(alt_sol) - 10
		alt_max = np.max(alt_sol) + 10

		ax_simple.set_xlim(pe_min, pe_max)
		ax_simple.set_ylim(pn_min, pn_max)
		ax_simple.set_zlim(alt_min, alt_max)
		ax_simple.set_box_aspect((pe_max - pe_min, pn_max - pn_min, alt_max - alt_min))

		plt.tight_layout()

		# Top view: save tightly and swap figure dimensions briefly to produce a "slimmer" output
		ax_simple.view_init(elev=90, azim=-90)
		top_view_path = self.output_path + f"{self.title.replace(' ', '_').lower()}_top_view.png"
		os.makedirs(os.path.dirname(top_view_path), exist_ok=True)
		orig_size = fig_simple.get_size_inches().copy()
		fig_simple.set_size_inches(orig_size[::-1])
		plt.savefig(
			top_view_path,
			dpi=300,
			bbox_inches='tight',
			pad_inches=0
		)
		fig_simple.set_size_inches(orig_size)

		# Side view (east) with title
		ax_simple.view_init(elev=0, azim=0)
		side_view_path = self.output_path + f"{self.title.replace(' ', '_').lower()}_side_view.png"
		os.makedirs(os.path.dirname(side_view_path), exist_ok=True)
		plt.savefig(
			side_view_path,
			dpi=300,
			bbox_inches='tight',
			pad_inches=0
		)
		# Do a print of what was saved 
		print(f"Saved top view to: {top_view_path}")
		print(f"Saved side view to: {side_view_path}")

		return fig_simple, ax_simple
