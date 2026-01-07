import open3d as o3d
import numpy as np
from python.utils.py_gpt import compute_mesh, GPTParams
from algorithms.points_algo.mls_surface_smooth_numba import mls_smoothing_numba


def build_greedy_mesh(pcd: o3d.geometry.PointCloud, radius: float, params: GPTParams) -> o3d.geometry.TriangleMesh:
	pcd.estimate_normals(
		search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30)
	)
	pcd.orient_normals_to_align_with_direction(orientation_reference=[0, 0, 1])
	pcd.orient_normals_consistent_tangent_plane(30)
	points = np.asarray(pcd.points, dtype=np.float64)
	normals = np.asarray(pcd.normals, dtype=np.float64)

	if len(points) != len(normals):
		raise ValueError("Points and normals length mismatch.")

	data = np.hstack((points, normals))
	face_indices = compute_mesh(data, params)

	mesh = o3d.geometry.TriangleMesh()
	mesh.vertices = pcd.points
	mesh.triangles = o3d.utility.Vector3iVector(face_indices)
	mesh.orient_triangles()
	mesh.compute_vertex_normals()
	mesh.compute_triangle_normals()

	# if many face normals are pointing inward ([vertex]-[mesh center] dot [face normal] < 0), flip all faces
	center = mesh.get_center()
	vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
	# vertex_centers = np.asarray(mesh.vertices, dtype=np.float64)
	vert = np.asarray(mesh.vertices)
	vectors_to_center = vert - center
	dot_products = np.einsum('ij,ij->i', vectors_to_center, vertex_normals)
	inward_count = np.sum(dot_products < 0)
	if inward_count > len(vertex_normals) / 2:
		mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles)[:, ::-1])
		mesh.compute_vertex_normals()
		mesh.compute_triangle_normals()
	return mesh


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

	pcd_omega_low = o3d.io.read_point_cloud("debug_pc_calc_data/step_-01_omega_high.ply")
	pcd_omega_low.paint_uniform_color([1.0, 1.0, 0.0])  # Paint points yellow
	pcd_omega_low = mls_smoothing_numba(pcd_omega_low, radius=0.005)
	
	pcd_grasp = o3d.io.read_point_cloud("debug_pc_calc_data/step_000_omega_g.ply")
	pcd_grasp.paint_uniform_color([0.0, 0.0, 1.0])  # Paint points blue
	radius = 0.005

	gpt_params = GPTParams()
	gpt_params.search_radius = 0.05
	gpt_params.mu = 2.5
	gpt_params.max_nearest_neighbors = 50
	gpt_params.max_surface_angle = 45.0
	gpt_params.min_angle = 10.0
	gpt_params.max_angle = 120.0
	gpt_params.normal_consistency = False


	gpt_params_high = GPTParams()
	gpt_params_high.search_radius = 0.05
	gpt_params_high.mu = 3.5
	gpt_params_high.max_nearest_neighbors = 180
	gpt_params_high.max_surface_angle = 45.0
	gpt_params_high.min_angle = 10.0
	gpt_params_high.max_angle = 120.0
	gpt_params_high.normal_consistency = False


	greedy_grasp_mesh = build_greedy_mesh(pcd_grasp, radius, gpt_params)
	greedy_omega_low_mesh = build_greedy_mesh(pcd_omega_low, radius, gpt_params_high)
	# greedy_mesh.orient_normals_to_align_with_direction(orientation_reference=[0, 0, 1])

	# # Possion reconstruction for pcd_grasp
	# mesh_grasp, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_grasp, depth=8)
	# mesh_grasp.compute_vertex_normals()

	# Visualize the point cloud
	o3d.visualization.draw_geometries([
									pcd_c1, 
									pcd_c2, 
									pcd_grasp, 
									greedy_grasp_mesh,
									# pcd_omega_low,
									# greedy_omega_low_mesh,
								], window_name="Grasp Part Contact Points")