import open3d as o3d
import numpy as np

if __name__ == "__main__":
	# Load point cloud from file
	mesh_c1 = o3d.io.read_triangle_mesh("debug_pc_calc_data/step_000_contact_mesh_side0.obj")
	mesh_c2 = o3d.io.read_triangle_mesh("debug_pc_calc_data/step_000_contact_mesh_side1.obj")

	resample_point_count = 2000
	pcd_c1 = o3d.geometry.PointCloud()
	pcd_c1.points = mesh_c1.sample_points_uniformly(number_of_points=resample_point_count).points
	pcd_c2 = o3d.geometry.PointCloud()
	pcd_c2.points = mesh_c2.sample_points_uniformly(number_of_points=resample_point_count).points
	pcd_c1.paint_uniform_color([1.0, 0.0, 0.0])  # Paint points red
	pcd_c2.paint_uniform_color([0.0, 1.0, 0.0])  # Paint points green
	
	pcd_grasp = o3d.io.read_point_cloud("debug_pc_calc_data/step_000_omega_g.ply")
	pcd_grasp.paint_uniform_color([0.0, 0.0, 1.0])  # Paint points blue

	# Visualize the point cloud
	o3d.visualization.draw_geometries([pcd_c1, pcd_c2, pcd_grasp], window_name="Grasp Part Contact Points")