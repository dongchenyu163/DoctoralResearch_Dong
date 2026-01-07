from algorithms.points_algo.mls_surface_smooth_numba import mls_smoothing_numba

import open3d as o3d
import numpy as np

if __name__ == "__main__":
	pcd = o3d.io.read_point_cloud("tests/points.pcd")

	# Apply MLS smoothing using the numba-optimized function
	smoothed_pcd = mls_smoothing_numba(pcd, radius=0.01)

	# Save the smoothed point cloud to a new file
	o3d.io.write_point_cloud("debug_pc_calc_data/smoothed_points.pcd", smoothed_pcd)

	print("MLS smoothing completed and saved to 'smoothed_points.pcd'")