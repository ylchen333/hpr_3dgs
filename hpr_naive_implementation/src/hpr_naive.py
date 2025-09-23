
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

from utils import get_device, get_points_renderer, save_tensors_as_point_cloud
from viewer import visualize_pointcloud_interactive

DEBUG = False

# ------------------------ hpr --------------------------------------------------------------------------

def visualize_pointcloud(
    point_cloud_path="data/bridge_pointcloud.npz", output_path="images/bridge.jpg", 
    point_color=None,
    background_color=np.zeros(3), image_size=256, device=None):
    
    if device is None:
        device = get_device()

    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )

    point_cloud = np.load(point_cloud_path)
    verts = torch.Tensor(point_cloud["verts"][::50]).to(device).unsqueeze(0)
    rgb = torch.Tensor(point_cloud["rgb"][::50]).to(device).unsqueeze(0)
    if point_color is not None: 
        color = torch.tensor(point_color, dtype=torch.float32, device=device) 
        B, N, _ = verts.shape
        rgb = color.view(1, 1, 3).expand(B, N, 3)
    point_cloud = pytorch3d.structures.Pointclouds(points=verts, features=rgb)
    R, T = pytorch3d.renderer.look_at_view_transform(4, 10, 0)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(point_cloud, cameras=cameras)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    return rend

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
    verts = torch.Tensor(pc_data["verts"][::50]).to(device) #.unsqueeze(0)

    print("point_transform verts sshape:" + str(verts.shape))
    

    # normalize coordinates s.t. C is origin
    if world_to_cam is not None:
        W2C = torch.as_tensor(world_to_cam, dtype=torch.float32, device=device)
        Rm = W2C[:3, :3]                     # (3,3)
        t  = W2C[:3,  3]                     # (3,)
        verts = verts @ Rm.T + t                 # (N,3)
        camera_coordinates = torch.zeros(3, device=device)
    else:
        verts = verts - torch.as_tensor(camera_coordinates, dtype=torch.float32, device=device)

    # decide an R for the sphere we flip to (simple: max dist away from Camera, just approx using min max)
    norms = torch.linalg.norm(verts, dim=1, keepdim=True)
    R = norms.max().item() * 1.02

    print("point_transform norms shape:" + str(norms.shape))


    # spherical flipping
    verts = verts + 2 *(R - norms) * verts / norms
    verts.unsqueeze(1)
    print("point_transform verts transformed shape:" + str(verts.shape))
    return verts



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
    points = point_transformation(
    point_cloud_path="data/bridge_pointcloud.npz", output_path="images/bridge.jpg", 
    camera_coordinates=camera_coordinates,
    point_color=None,
    background_color=background_color, image_size=256, device=None)

    # apply convexhull algo to the transformed points
    # according to file:///home/lorie/Downloads/cgf70046.pdf, we can approximate the CH differently:
    # compute a visibility indicator (where visible points maximize the projection in the direction d_i = {some math from the paper})
    # 
    if points.ndim == 3 and points.shape[0] == 1:
        points = points.squeeze(0)   

    points = points.detach().cpu()
    mask = torch.isfinite(points).all(dim=1)         # drop any NaN/Inf rows
    points = points[mask]
    if points.shape[0] < 4:
        raise ValueError("Need at least 4 non-coplanar points for a 3D hull.")

    aug = torch.zeros(1, 3, dtype=points.dtype)         # the camera point C=0
    points = torch.vstack([points, aug]) 
    points_np = points.numpy().astype(np.float64, copy=False)
    # points_np = points.detach().cpu().numpy()
    hull = ConvexHull(points_np)
    # hull_pc_file = "./data/hull.ply"
    idx = torch.from_numpy(hull.vertices).long()             # (K,)
    hull_points = points[idx]
    save_tensors_as_point_cloud([hull_points], filename=output_path)

    return 
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
            img = visualize_pointcloud(point_color=[1.0, 0.0, 0.0])
            plt.imsave("images/bridge.jpg", img)
            visualize_pointcloud_interactive(
                point_cloud_path="data/bridge_pointcloud.npz",
                point_color=[1, 0, 0],
                stride=50,              # subsample if it's huge
                image_size=args.image_size,
                background_color=(0, 0, 0),
            )
        case 2:
            hpr(point_cloud_path="data/bridge_pointcloud.npz",
                output_path="./data/bridge_pointcloud_hull.npz",
                point_color=(1, 0, 0)
                )
            img = visualize_pointcloud(point_cloud_path="./data/bridge_pointcloud_hull.npz")
            plt.imsave("images/bridge_hpr.jpg", img)
            visualize_pointcloud_interactive(
                point_cloud_path="./data/bridge_pointcloud_hull.npz",
                point_color=[1, 0, 0],
                image_size=args.image_size*2, 
                background_color=[0, 0, 0]
            )
        case _:
            pass