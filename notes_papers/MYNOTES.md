# My Notes for 2dgs with HPR

chatgpt:
"HPR is too aggressive, even though your camera is now placed outside. Here's why this still happens:

Spherical flipping (HPR core step) folds the points around the sphere centered at the camera.

If most points are on the ground plane, HPR sees only the lower convex shell (sides and bottom), and ignores all top-surface points.

This is a known issue with single-view HPR — it returns only the convex subset of the visible contour from that camera direction."
^ is this a possible hyperparameter?

things:
camera dist from point cloud
density of points
angle of view of incidence (idk how to phrase this, its like are you looking a book from its spine vs cover)

9.30.25 yannis meeting notes
todo:
- figure out relationship between hyperparameter gamme/R and the HPR operator
    - take meshes from stanford mesh repo
    - sample points on mesh with density ro (using gpytoolbox)
    - greate a GT of visibility via raycasting on the mesh itself (using gpytoolbox ray_mesh_intersect)
    - plot errors as a function of different gamma, diff densities, viewpoints (look at false positives/negatives and overall error)

explore gamme wrt curvature (determined via area thing yannis send?)

from yannis:
"And to summarize what we discussed today: We need to understand and automate the selection of param to ensure robust performance.
We can try to come up with some heuristic that takes into account point cloud density, curvature, bounding box, and distance, following the insights in the 2007 paper.
We can potentially consider per-point values of param, though it's unclear whether that's theoretically sound.
We can train a neural network that predicts a good param from some input point cloud and viewpoint statistics.
To start examining this question in a more systematic way, the plan of action is as follows: Set up a benchmark where you can take a mesh, sample random point clouds of different densities (controlled using the number of points) from that mesh, and use the mesh and ray-mesh intersections to compute groundtruth visibility values for each point. Then once this groundtruth is available, use HPR with different param values and inversion kernels to assess accuracy (e.g., number of false negatives and false positives).
To create this benchmark, you can use gpytoolbox (linked above) to load meshes, sample point clouds on them, and do ray-mesh intersections---the toolbox has functions for all these operations. You then apply this benchmarking procedure to various meshes from the casual-effects website (also linked above)."

HPR implementation in open3d: [open3d doc](https://github.com/cmu-ci-lab/fast_dipole_sums/blob/2e8b763f610f5431a107757c433c554029490ad0/util/point_cloud_util.py#L14)



convexhull on gpu
only cpu rn
dont need to make hpr differentiable, just need the visibility
furthest point query(?)