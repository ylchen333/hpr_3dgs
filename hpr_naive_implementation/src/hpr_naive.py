
import matplotlib.pyplot as plt
import mcubes
import numpy as np
import pytorch3d
import torch
from tqdm.auto import tqdm
from scipy.spatial import ConvexHull

import argparse
import pickle
from PIL import Image, ImageDraw
import imageio
import json
from simple_interactive_viewer import visualize_pointcloud_interactive
import visualize
from open3d_visualize import visualize_hpr_result, debug_indexing_with_open3d
from HPR import HPR


DEBUG = False

# ------------------------ hpr --------------------------------------------------------------------------

# og author edit
def HPR(pts, viewpoint, gamma, kernel_type="spherical_flip"):
    """
    Hidden Point Removal (HPR)
    --------------------------------------------------------------------
    Implements the algorithm from:
        Katz, S., Tal, A., & Basri, R. (2007).
        "Direct visibility of point sets."
        ACM TOG 26(3), 24.

    Supports multiple kernel types:
        - 'spherical_flip' (default):   linear mirror kernel  p' = (2γ - r) * (p/r)
        - 'mirror':                     simple linear kernel  p' = (γ - r) * (p/r)
        - 'exp_inversion':              exponential inversion p' = (r^γ) * (p/r)
        - 'exp_natural':                natural exponential   p' = exp(-γ * r) * (p/r)

    Parameters
    ----------
    pts : torch.Tensor
        Tensor of shape (3, N) or (1, 3, N) — point cloud coordinates.
    viewpoint : torch.Tensor
        Tensor of shape (3,) or (1, 3) — camera/viewpoint position.
    gamma : float
        Kernel radius parameter (typically ≥ max distance of points from viewpoint).
    kernel_type : str
        One of {'spherical_flip', 'mirror', 'exp_inversion', 'exp_natural'}.

    Returns
    -------
    visible_points : torch.Tensor, shape (3, M)
        Subset of visible points in the original coordinate frame.
    visible_indices : np.ndarray, shape (M,)
        Indices of visible points within the original set.
    """
    # ---------- Input normalization ----------
    if pts.dim() == 2:
        pts = pts.unsqueeze(0)  # (1, 3, N)
    if viewpoint.dim() == 1:
        viewpoint = viewpoint.unsqueeze(0)  # (1, 3)

    B, D, N = pts.shape
    assert D == 3, "Only 3D point clouds supported"

    # Center the points around the viewpoint
    centered_points = pts - viewpoint.unsqueeze(2)
    directions = torch.nn.functional.normalize(centered_points, dim=1)
    radii = torch.norm(centered_points, dim=1, keepdim=True)

    # ---------- Apply kernel transformation ----------
    if kernel_type == "spherical_flip":
        trans_points = (2.0 * gamma - radii) * directions

    elif kernel_type == "mirror":
        trans_points = (gamma - radii) * directions

    elif kernel_type == "exp_inversion":
        # Usually gamma < 0
        trans_points = torch.pow(radii + 1e-8, gamma) * directions

    elif kernel_type == "exp_natural":
        # Usually gamma > 0
        trans_points = torch.exp(-gamma * radii) * directions

    else:
        raise ValueError(
            f"Unknown kernel_type '{kernel_type}'. "
            f"Choose from ['spherical_flip', 'mirror', 'exp_inversion', 'exp_natural']"
        )

    # ---------- Convex hull on transformed points ----------
    trans_np = trans_points.squeeze(0).permute(1, 0).cpu().numpy()
    hull = ConvexHull(trans_np)

    # Visible vertices correspond to points on hull
    visible_indices = np.unique(hull.vertices)
    visible_points = pts[:, :, visible_indices]

    return visible_points, visible_indices



def compute_camera_outside_bounds(point_cloud_path, scale=1.5):
    """
    Returns a camera location outside the bounding box of the point cloud.

    Args:
        point_cloud_path (str): Path to .npz file containing 'verts'
        scale (float): How far outside the bounding box to place the camera

    Returns:
        np.ndarray: A 3D camera position
    """
    verts = np.load(point_cloud_path)["verts"]  # (N, 3)
    min_bounds = verts.min(axis=0)
    max_bounds = verts.max(axis=0)
    center = (min_bounds + max_bounds) / 2
    diag = np.linalg.norm(max_bounds - min_bounds)

    # Position camera along -Z direction, far away from the center
    # camera = center + np.array([0, 0, -1]) * diag * scale # side view
    # camera = center + np.array([1, 1, -1]) * diag * scale # diag down
    camera = center + np.array([0.2, 0.2, 1.0]) * diag * scale # top down ish



    print(f"[Camera] Bounds diag={diag:.2f}, Camera Pos={camera}")
    return camera


def point_transformation(
    point_cloud_path="data/bridge_pointcloud.npz", 
    output_path="images/bridge.jpg", 
    camera_coordinates=np.zeros(3), world_to_cam=None,
    point_color=None,
    background_color=np.zeros(3), image_size=256, device=None
    ):
    # spherical flipping transformation
    # pbi = F(pi) = pi + 2(R − ||pi||) pi / ||pi||

    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    pc_data = np.load(point_cloud_path, mmap_mode='r')
    verts_raw = torch.tensor(pc_data["verts"][::50], dtype=torch.float32, device=device)

    camera_tensor = torch.tensor(camera_coordinates, dtype=torch.float32, device=device)
    verts_translated = verts_raw - camera_tensor
    norms = torch.linalg.norm(verts_translated, dim=1, keepdim=True).clamp(min=1e-6)
    R = norms.max().item() * 1.02
    verts_flipped = verts_translated + 2 * (R - norms) * verts_translated / norms

    return verts_flipped, verts_translated

import torch
import numpy as np
from scipy.spatial import ConvexHull


def apply_inversion_kernel(points, gamma, kernel_type="mirror"):
    """
    Apply inversion or flipping kernel to point distances.

    Args:
        points (torch.Tensor): Nx3 tensor of points in camera coordinates
        gamma (float): user parameter controlling kernel shape
        kernel_type (str): one of
            ['spherical_flip', 'mirror', 'exp_inversion', 'exp_natural']

    Returns:
        torch.Tensor: transformed (flipped) points
    """
    # Distance from camera (origin)
    d = torch.norm(points, dim=1, keepdim=False)
    direction = points / (d.unsqueeze(1) + 1e-8)

    if kernel_type == "spherical_flip":
        # Classic HPR inversion (flip across a sphere)
        # R = 0.5 * gamma, reflection across sphere of radius R
        R = 0.5 * gamma
        transformed_d = (2 * R) - d
        points_flipped = direction * transformed_d.unsqueeze(1)

    elif kernel_type == "mirror":
        # f_mirror(d) = gamma - d
        transformed_d = gamma - d
        points_flipped = direction * transformed_d.unsqueeze(1)

    elif kernel_type == "exp_inversion":
        # f_exponential(d) = d^γ  (γ < 0)
        transformed_d = torch.pow(d, gamma) # gamma < 0
        points_flipped = direction * transformed_d.unsqueeze(1)

    elif kernel_type == "exp_natural":
        # f_natural(d) = e^{-γ d}  (γ > 0)
        transformed_d = torch.exp(-gamma * d)  # gamma > 0
        points_flipped = direction * transformed_d.unsqueeze(1)

    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type}")

    return points_flipped


def hpr(
    point_cloud_path="data/bridge_pointcloud.npz",
    output_path="./data/hull.ply",
    inversion_func="spherical_flip",
    gamma=1.0,
    camera_coordinates=np.zeros(3),
    point_color=None,
    background_color=np.zeros(3),
    image_size=256,
    device=None
):
    """
    Hidden Point Removal (HPR) algorithm supporting multiple inversion kernels.

    Args:
        inversion_func (str): One of
            ['spherical_flip', 'mirror', 'exp_inversion', 'exp_natural']
        gamma (float): Kernel parameter (behavior depends on kernel)
    """
    # Transform points so camera is at origin
    points_flipped, points_translated = point_transformation(
        point_cloud_path=point_cloud_path,
        output_path="images/bridge.jpg",
        camera_coordinates=camera_coordinates,
        point_color=None,
        background_color=background_color,
        image_size=image_size,
        device=device
    )

    # Flatten dimensions if needed
    if points_flipped.ndim == 3 and points_flipped.shape[0] == 1:
        points_flipped = points_flipped.squeeze(0)
        points_translated = points_translated.squeeze(0)

    points_flipped = points_flipped.detach().cpu()
    points_translated = points_translated.detach().cpu()

    mask = torch.isfinite(points_flipped).all(dim=1)
    points_flipped = points_flipped[mask]
    points_translated = points_translated[mask]

    if points_flipped.shape[0] < 4:
        raise ValueError("Need at least 4 non-coplanar points for a 3D hull.")

    # 🌀 Apply inversion kernel (flip)
    points_flipped = apply_inversion_kernel(points_flipped, gamma, kernel_type=inversion_func)

    # Add camera origin to flipped points
    aug = torch.zeros(1, 3, dtype=points_flipped.dtype)
    points_flipped_with_cam = torch.cat([points_flipped, aug], dim=0)
    points_np = points_flipped_with_cam.numpy()

    # Build convex hull
    hull = ConvexHull(points_np)
    print(f"HPR ({inversion_func}) hull volume:", hull.volume)
    print("HPR hull area:", hull.area)

    idx = torch.from_numpy(hull.vertices).long()
    idx = idx[idx < points_flipped.shape[0]]

    hull_points_ogspace = points_translated[idx]

    debug_indexing_with_open3d(point_cloud_path, idx, show=DEBUG)

    return hull_points_ogspace
# ------------------------ main --------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, default=1)
    parser.add_argument("--debug", action='store_true', help="enable debug mode")
    parser.add_argument(
        "--render",
        type=str,
        default="point_cloud",
        choices=["point_cloud", "parametric", "implicit", "fun"],
    )
    parser.add_argument("--output_path", type=str, default="images/bridge.jpg")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=100)
    
    args = parser.parse_args()

    if args.debug: DEBUG = True

    match args.task:
        case 1:
            img = visualize.visualize_pointcloud(point_color=[1.0, 0.0, 0.0])
            plt.imsave("images/bridge.jpg", img)
            visualize_pointcloud_interactive(
                point_cloud_path="data/bridge_pointcloud.npz",
                point_color=[1, 0, 0],
                stride=50,              # subsample if it's huge
                image_size=args.image_size,
                background_color=(0, 0, 0),
            )
        case 2:
            camera_coord = compute_camera_outside_bounds("data/bridge_pointcloud.npz", scale=2.0)
            hpr_pc = hpr(point_cloud_path="data/bridge_pointcloud.npz",
                output_path="./data/bridge_pointcloud_hull.npz",
                inversion_func="spherical_flip",
                point_color=(1, 0, 0), camera_coordinates=camera_coord
                )
            
            visualize.save_tensors_as_point_cloud([hpr_pc], filename="./data/bridge_pointcloud_hull.npz")

            # img = visualize.visualize_pointcloud(point_cloud_path="./data/bridge_pointcloud_hull.npz")
            # plt.imsave("images/bridge_hpr.jpg", img)
            # visualize_pointcloud_interactive(
            #     point_cloud_path="./data/bridge_pointcloud_hull.npz",
            #     point_color=[1, 0, 0],
            #     image_size=args.image_size*2, 
            #     background_color=[0.8, 0.8, 0.8]
            # )

            visualize_hpr_result()


            visualize.render_rotating_pointcloud_gif("data/bridge_pointcloud.npz", "images/original.gif")
            visualize.render_rotating_pointcloud_gif("data/bridge_pointcloud_hull.npz", "images/hpr.gif")

        case 3: # using the implementation of hpr from original authors
            camera_coord = compute_camera_outside_bounds("data/bridge_pointcloud.npz", scale=2.0)
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            pc_data = np.load("data/bridge_pointcloud.npz", mmap_mode='r')
            verts_raw = torch.tensor(pc_data["verts"], dtype=torch.float32, device=device)
            print("verts_raw .size before reshape:", verts_raw.size())

            verts_batched = verts_raw.T.unsqueeze(0)  # (1, 3, N)
            print("verts_batched .size after reshape:", verts_batched.size())
            camera_tensor = torch.tensor(camera_coord, dtype=torch.float32, device=device).view(1, 3, 1)



            visible_points, visible_indices = HPR(verts_batched, camera_tensor, gamma=1.0, use_linear_kernel=False)
            visible_points_np = visible_points.squeeze(0).permute(1, 0).cpu()
            visualize.save_tensors_as_point_cloud([visible_points_np], filename="./data/author_bridge_pointcloud_hull.npz")
            visualize_hpr_result(original_path="data/bridge_pointcloud.npz", filtered_path="./data/author_bridge_pointcloud_hull.npz")


            visualize.render_rotating_pointcloud_gif("data/bridge_pointcloud.npz", "images/author_original.gif")
            visualize.render_rotating_pointcloud_gif("data/author_bridge_pointcloud_hull.npz", "images/author_hpr.gif")

        case _:
            pass