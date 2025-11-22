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
from utils import get_points_renderer
import imageio
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras
import pytorch3d
import torch
import numpy as np
from utils import get_device, get_points_renderer, save_tensors_as_point_cloud

# ------------------------ hpr --------------------------------------------------------------------------
# for creating visualization assets


def render_rotating_pointcloud_gif(
    point_cloud_path,
    output_gif_path="images/rotating.gif",
    num_frames=36,
    elev=10.0,
    distance=4.0,
    image_size=512,
    background_color=(0, 0, 0),
    point_color=None,
):

    device = get_device()
    renderer = get_points_renderer(image_size=image_size, device=device, background_color=background_color)

    # Load point cloud
    npz = np.load(point_cloud_path)
    verts = torch.tensor(npz["verts"], dtype=torch.float32).to(device)
    rgb = torch.tensor(npz["rgb"], dtype=torch.float32).to(device)
    if rgb.max() > 1.0:
        rgb /= 255.0

    if point_color is not None:
        color = torch.tensor(point_color, dtype=torch.float32, device=device).view(1, 3)
        rgb = color.expand(verts.shape[0], 3)

    pc = pytorch3d.structures.Pointclouds(points=[verts], features=[rgb])
    
    # Generate frames
    frames = []
    for azim in np.linspace(0, 360, num_frames, endpoint=False):
        R, T = pytorch3d.renderer.look_at_view_transform(dist=distance, elev=elev, azim=azim, device=device)
        cameras = FoVPerspectiveCameras(R=R, T=T, device=device)
        with torch.no_grad():
            img = renderer(pc, cameras=cameras)[0, ..., :3].cpu().numpy()
        frames.append((img * 255).astype(np.uint8))

    # Save as gif
    imageio.mimsave(output_gif_path, frames, duration=0.1)
    print(f"Saved rotating gif to {output_gif_path}")


def compare_pointclouds_side_by_side(
    original_path="data/bridge_pointcloud.npz",
    filtered_path="data/bridge_pointcloud_hull.npz",
    image_size=256
):
    from hpr_naive import visualize_pointcloud

    # Render both images
    original_img = visualize_pointcloud(point_cloud_path=original_path, image_size=image_size)
    filtered_img = visualize_pointcloud(point_cloud_path=filtered_path, image_size=image_size)

    # Create a side-by-side plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(original_img)
    axes[0].set_title("Original Point Cloud")
    axes[0].axis("off")

    axes[1].imshow(filtered_img)
    axes[1].set_title("After HPR Filtering")
    axes[1].axis("off")
    plt.savefig("images/comparison.jpg", dpi=300)


    plt.tight_layout()
    plt.show()


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