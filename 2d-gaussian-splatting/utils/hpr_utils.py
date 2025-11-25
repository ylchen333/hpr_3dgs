from scipy.spatial import ConvexHull
import numpy as np
import torch


# For a given mesh and sampled points, run HPR with different gamma values and compute FP/FN against groundtruth visibility.
# use yannis's hpr implementation
def HPR_Param(points, view, param, use_linear=False):
    """
    Hidden Point Removal (HPR) operator.
    Reimplementation of Yannis's Matlab implementation of the HPR operator in Python.

    Args:
        points (np.ndarray): Nx3 array of 3D points.
        view (np.ndarray): 3-element camera/viewpoint position.
        param (float): Inversion parameter (0 < param < 1).
        use_linear (bool): If True, use linear inversion; otherwise exponential inversion.

    Returns:
        visible_idx (np.ndarray): Indices of visible points.
    """
    device = points.device

    # --- Input validation ---
    if points.shape[0] == 3 and points.shape[1] != 3:
        points = points.T

    # Convert to NumPy for convex hull
    points = points.detach().cpu().numpy()
    view = view.detach().cpu().numpy().reshape(1, 3)
    # TODO: at some point add gpu based convexhull library

    num_pts, dim = points.shape

    # --- Move points so camera is at the origin ---
    shifted = points - view  # Nx3
    normp = np.linalg.norm(shifted, axis=1)
    directions = shifted / normp[:, None]  # normalized direction vectors

    # --- Inversion step ---
    if use_linear:
        # Linear inversion: R = (1/param)*max(normp)
        R = (1.0 / param) * np.max(normp)
        P = (R - normp)[:, None] * directions
    else:
        # Exponential inversion: points scaled by ||p||^(-param)
        P = (normp ** (-param))[:, None] * directions

    # --- Form augmented set with origin ---
    augmented_points = np.vstack([P, np.zeros((1, dim))])

    # --- Compute convex hull ---
    try:
        hull = ConvexHull(augmented_points)
    except Exception as e:
        raise RuntimeError(f"HPR convex hull failed: {e}")

    # Extract visible vertices
    visible_idx = np.unique(hull.vertices)
    visible_idx = visible_idx[visible_idx < num_pts]

    visible_pts = torch.from_numpy(points[visible_idx].T).float().to(device)
    visible_idx_torch = torch.from_numpy(visible_idx).long().to(device)

    return visible_pts, visible_idx_torch