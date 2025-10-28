import os
import numpy as np
import open3d as o3d
from gamma_visualize import benchmark_HPR_visibility  # assuming gamma_visualize.py defines this
from gamma_visualize import curvature_guided_gamma, refine_R_via_visibility
import trimesh

# --- Parameters ---
meshes = [
    "../data/bunny.obj",
    "../data/buddha.obj",
    "../data/dragon.obj",
    "../data/erato.obj",
    "../data/hairball.obj",
]
densities = [1000, 5000, 10000, 20000]
kernels = ["spherical_flip", "mirror"]
out_root = "../data/benchmarks_density"

os.makedirs(out_root, exist_ok=True)

# --- Main loop ---
for mesh_path in meshes:
    mesh_name = os.path.splitext(os.path.basename(mesh_path))[0]
    

    mesh = trimesh.load(mesh_path, force='mesh')

    # If faces are not triangles, triangulate using built-in utility
    if mesh.faces.shape[1] != 3:
        print("[Info] Triangulating non-triangular mesh...")
        # Use Trimesh's "convex_hull" trick to force triangulation if needed
        # mesh = mesh.convex_hull  # creates a triangulated hull
        # Alternatively, use subdivision to break quads into tris
        mesh = mesh.subdivide_to_size(max_edge=0.01)

    # Verify
    print(f"[Info] Mesh now has {mesh.faces.shape[0]} triangular faces.")
    tri_path = os.path.join(os.path.dirname(mesh_path), f"{mesh_name}_tri.obj")
    if not os.path.exists(tri_path):
        mesh.export(tri_path)
    mesh_path = tri_path  # now benchmark uses the triangulated file


    # Save triangulated copy
    mesh.export(tri_path)
    print(f"[Info] Saved triangulated mesh to {tri_path}")

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    print(f"\n=== {mesh_name} ===")

    for density in densities:
        print(f"  -> Sampling {density} points")
        pcd = mesh.sample_points_poisson_disk(number_of_points=density)
        points = np.asarray(pcd.points)
        camera_origin = np.array([0, 0, 5.0])  # or use mesh.get_center() + offset

        for kernel in kernels:
            print(f"     Kernel: {kernel}")
            outdir = os.path.join(out_root, f"{mesh_name}_{density}_{kernel}")
            os.makedirs(outdir, exist_ok=True)

            # Compute curvature-guided gamma values
            gamma_values, xlabel, R = curvature_guided_gamma(points, camera_origin, kernel)

            # Refine R based on visibility overlap
            R_optimal = refine_R_via_visibility(points, camera_origin, kernel)

            # Run HPR benchmark and generate plots
            benchmark_HPR_visibility(
                points=points,
                camera_origin=camera_origin,
                kernel=kernel,
                gamma_values=gamma_values,
                R_pred=R,
                R_opt=R_optimal,
                mesh_path=mesh_path,
                outdir=outdir
            )
