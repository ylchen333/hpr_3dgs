
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
    # if device is None:
    #     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    # pc_data = np.load(point_cloud_path, mmap_mode='r')
    # verts_raw = torch.tensor(pc_data["verts"][::50], dtype=torch.float32, device=device)



    # print("point_transform verts sshape:" + str(verts_raw.shape))
    

    # # normalize coordinates s.t. C is origin
    # if world_to_cam is not None:
    #     W2C = torch.as_tensor(world_to_cam, dtype=torch.float32, device=device)
    #     Rm = W2C[:3, :3]                     # (3,3)
    #     t  = W2C[:3,  3]                     # (3,)
    #     verts_translated = verts_raw @ Rm.T + t                 # (N,3)
    #     camera_coordinates = torch.zeros(3, device=device)
    # else:
    #     verts_translated = verts_raw - torch.tensor(camera_coordinates, dtype=torch.float32, device=device)



    # # decide an R for the sphere we flip to (simple: max dist away from Camera, just approx using min max)
    # norms = torch.linalg.norm(verts_translated, dim=1, keepdim=True).clamp(min=1e-6)
    # R = norms.max().item() * 1.02
    # verts_flipped = verts_translated + 2 * (R - norms) * verts_translated / norms

    # print("point_transform verts flipped shape:", verts_flipped.shape)
    # return verts_flipped, verts_raw  # flipped and original space



def hpr(
    point_cloud_path="data/bridge_pointcloud.npz", output_path="./data/hull.ply", 
    camera_coordinates=np.zeros(3),
    point_color=None,
    background_color=np.zeros(3), image_size=256, device=None
):
    '''
    input a point cloud, make a point cloud with only visible points
    '''
    # transform the points s.t. C is the origin, where C is the camera
    points_flipped, points_translated  = point_transformation(
                    point_cloud_path="data/bridge_pointcloud.npz", output_path="images/bridge.jpg", 
                    camera_coordinates=camera_coordinates,
                    point_color=None,
                    background_color=background_color, image_size=256, device=None)

    # apply convexhull algo to the transformed points
    # according to file:///home/lorie/Downloads/cgf70046.pdf, we can approximate the CH differently:
    # compute a visibility indicator (where visible points maximize the projection in the direction d_i = {some math from the paper})
    # 
    if points_flipped.ndim == 3 and points_flipped.shape[0] == 1:
        points_flipped = points_flipped.squeeze(0)
        points_translated = points_translated.squeeze(0)

    points_flipped = points_flipped.detach().cpu()
    points_translated = points_translated.detach().cpu()

    mask = torch.isfinite(points_flipped).all(dim=1)
    points_flipped = points_flipped[mask]
    points_translated = points_translated[mask]  # 👈 align filtered original points
    if points_flipped.shape[0] < 4:
        raise ValueError("Need at least 4 non-coplanar points for a 3D hull.")
    
    print("verts_raw shape:", points_translated.shape)
    print("verts_flipped shape:", points_flipped.shape)
    print("finite_mask sum:", mask.sum())


    # Append camera origin to flipped only
    aug = torch.zeros(1, 3, dtype=points_flipped.dtype)
    points_flipped_with_cam = torch.cat([points_flipped, aug], dim=0)
    points_np = points_flipped_with_cam.cpu().numpy()

    hull = ConvexHull(points_np)
    print("HPR hull volume:", hull.volume)
    print("HPR hull area:", hull.area)

    idx = torch.from_numpy(hull.vertices).long()

    # ⚠️ We need to exclude the camera point index (last point)
    idx = idx[idx < points_flipped.shape[0]]

    # ✅ Select the correct points from original space
    hull_points_ogspace = points_translated[idx]

    debug_indexing_with_open3d(point_cloud_path, idx)

    #  project points back to original space
    # hull_points_ogspace = points_og[idx] + torch.tensor(camera_coordinates, device=hull_points.device, dtype=hull_points.dtype)


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
            verts_raw = torch.tensor(pc_data["verts"][::50], dtype=torch.float32, device=device)  # (N, 3)
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