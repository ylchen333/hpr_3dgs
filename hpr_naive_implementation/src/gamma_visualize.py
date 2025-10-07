
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
import hpr_naive


DEBUG = False
MESH_PATH = "../data/bunny.obj"

# Set up a benchmark where you can take a mesh, sample random point clouds of different densities (controlled using the number of points) from that mesh, 
# and use the mesh and ray-mesh intersections to compute groundtruth visibility values for each point. 
# Then once this groundtruth is available, use HPR with different param values and inversion kernels to assess accuracy 
# (e.g., number of false negatives and false positives).

# ------------------------ hpr --------------------------------------------------------------------------

def visualize_gamma(mesh_path, output_folder_path, image_size=256, gamma_min=-1, gamma_max=1, step_size=0.1):
    """
    for range [gamma_min, gamma_max] with step_size, run hpr and visualize the results, and compute the error (wrt the mesh)
    name the files with the following convention <mesh_path without file type ext>_<gamma>.png

    Args:
        mesh_path (str): Path to .obj file
        

    Returns:
        nothing
    """
    verts = np.load(mesh_path)

    return 



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, default=1)
    parser.add_argument("--debug", action='store_true', help="enable debug mode")
    parser.add_argument("--input_path", type=str, default="../data/bunny.obj")
    parser.add_argument("--output_folder_path", type=str, default="./data/gamma_vis/")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=100)
    
    args = parser.parse_args()
    if args.debug: DEBUG = True
    visualize_gamma(args.input_path, args.output_folder_path, args.image_size, )
