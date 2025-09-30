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