import open3d as o3d
import numpy as np


import open3d as o3d
import numpy as np

def visualize_hpr_result(
    original_path="data/bridge_pointcloud.npz",
    filtered_path="data/bridge_pointcloud_hull.npz",
    scale=2.0
):
    # Load original point cloud
    original_npz = np.load(original_path)
    original_points = original_npz["verts"]
    original_colors = original_npz["rgb"]
    if original_colors.max() > 1.0:
        original_colors = original_colors / 255.0
    
    print("[Debug] original_points:", original_points.shape, original_points.dtype)
    print("[Debug] original_colors:", original_colors.shape, original_colors.dtype)


    # Create Open3D original point cloud
    pcd_original = o3d.geometry.PointCloud()
    pcd_original.points = o3d.utility.Vector3dVector(original_points)

    if original_colors.shape[1] == 4:
        original_colors = original_colors[:, :3]


    original_colors = original_colors.astype(np.float64)
    pcd_original.colors = o3d.utility.Vector3dVector(original_colors)


    # Load HPR-filtered point cloud
    filtered_npz = np.load(filtered_path)
    filtered_points = filtered_npz["verts"]
    filtered_colors = np.tile(np.array([[1.0, 0.0, 0.0]]), (filtered_points.shape[0], 1))  # red

    pcd_filtered = o3d.geometry.PointCloud()
    pcd_filtered.points = o3d.utility.Vector3dVector(filtered_points)
    filtered_colors = filtered_colors.astype(np.float64)
    pcd_filtered.colors = o3d.utility.Vector3dVector(filtered_colors)


    # Compute camera position used for HPR (same method you used before)
    min_bounds = original_points.min(axis=0)
    max_bounds = original_points.max(axis=0)
    center = (min_bounds + max_bounds) / 2
    diag = np.linalg.norm(max_bounds - min_bounds)
    camera = center + np.array([0, 0, -1]) * diag * scale

    # Show the camera coordinate frame
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=diag * 0.1)
    camera_frame.translate(camera)

    # Show all together
    o3d.visualization.draw_geometries(
        [pcd_original, pcd_filtered, camera_frame],
        window_name="HPR Visualization (Open3D)",
        width=1024,
        height=768,
        point_show_normal=False
    )



import open3d as o3d
import numpy as np
import torch

def debug_indexing_with_open3d(point_cloud_path, hpr_indices):
    """
    Visualize original point cloud and points selected by HPR indices.

    Args:
        point_cloud_path (str): .npz file containing 'verts'
        hpr_indices (torch.Tensor or np.ndarray): indices from ConvexHull.vertices
    """
    data = np.load(point_cloud_path)
    verts = data["verts"]
    
    if isinstance(hpr_indices, torch.Tensor):
        hpr_indices = hpr_indices.cpu().numpy()

    # Original point cloud (gray)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    pcd.colors = o3d.utility.Vector3dVector(np.tile([[0.6, 0.6, 0.6]], (verts.shape[0], 1)))

    # HPR-selected points (red)
    pcd_hpr = o3d.geometry.PointCloud()
    pcd_hpr.points = o3d.utility.Vector3dVector(verts[hpr_indices])
    pcd_hpr.colors = o3d.utility.Vector3dVector(np.tile([[1.0, 0.0, 0.0]], (len(hpr_indices), 1)))

    # Show both
    o3d.visualization.draw_geometries([pcd, pcd_hpr])

# data = np.load("data/bridge_pointcloud.npz")
# verts = data["verts"]
# camera = compute_camera_outside_bounds("data/bridge_pointcloud.npz", scale=2.0)

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(verts)

# camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
# camera_frame.translate(camera)

# o3d.visualization.draw_geometries([pcd, camera_frame])
