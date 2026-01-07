import importlib, os
os.environ["PATH"] += os.pathsep + "/home/cookteam/Documents/blender-4.5.3-linux-x64"

import open3d as o3d
import trimesh
import trimesh.interfaces.blender as blend
importlib.reload(blend)


if __name__ == "__main__":
	base_mesh = trimesh.load("debug_pc_calc_data/base_mesh.obj")
	knife_mesh = trimesh.load("debug_pc_calc_data/knife_mesh.obj")

	# Normal & Face orientation must be fixed for boolean to work correctly.
	
	print(f"Is base_mesh watertight? {base_mesh.is_watertight}")
	base_mesh.fix_normals()  # Need networkx package
	trimesh.exchange.export.export_mesh(base_mesh, "debug_pc_calc_data/base_mesh_trimesh_oriented.obj")

	o3dmesh = o3d.io.read_triangle_mesh("debug_pc_calc_data/base_mesh.obj")
	o3dmesh.compute_vertex_normals()
	o3dmesh.compute_triangle_normals()
	o3dmesh.orient_triangles()
	o3d.io.write_triangle_mesh("debug_pc_calc_data/base_mesh_oriented.obj", o3dmesh)

	# make box from base_mesh's aabb.
	box = trimesh.creation.box(bounds=base_mesh.bounds)
	# move box up by 0.01 in z direction
	knife_mesh.apply_translation([0, 0, -0.01])

	merged = trimesh.util.concatenate([base_mesh, knife_mesh])

	intersection = base_mesh.intersection(knife_mesh, engine='blender', check_volume=False, use_exact=True)
	print(intersection.is_empty)
	trimesh.exchange.export.export_mesh(intersection, "debug_pc_calc_data/contact_mesh.obj")
	trimesh.exchange.export.export_mesh(merged, "debug_pc_calc_data/merged.obj")