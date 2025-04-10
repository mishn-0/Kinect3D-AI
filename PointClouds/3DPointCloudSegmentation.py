import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt  # Fixed the import name

# Point Cloud Data Preparation

DATANAME = "appartment_cloud.ply"  # Change this to the data you want

pcd = o3d.io.read_point_cloud("../DATA/" + DATANAME)  # Point Cloud Data is an open3d object, we use a relative path

# PRE PROCESSING

# We need to translate everything so that Open3D can visualize it correctly
pcd_center = pcd.get_center()  # Get the center of the data
pcd.translate(-pcd_center)  # we don't need to translate it in this case but we're doing it just in case

# STATISTICAL OUTLIER FILTER
# Some point clouds may have some noise

nn = 16  # 16 is the number of nearest neighbors

std_multiplier = 2.0  # We use the STD of our Point distribution and apply that on a multiplier, a PC above this number it's an outlier

filtered_pcd, inlier_indices = pcd.remove_statistical_outlier(nn, std_multiplier)  # Now we can use this function to remove statistical outliers
# After filtering we keep the outliers elsewhere

outliers = pcd.select_by_index(inlier_indices, invert=True)
outliers.paint_uniform_color([1, 0, 0])  # We paint the outliers in RED so we can see them more clearly

filtered_pcd = filtered_pcd

o3d.visualization.draw_geometries([filtered_pcd, outliers])

# VOXEL DOWNSAMPLING

# The size of the little cube, it determines the level of detail and amount of downsampling,
# smaller means more points are preserved but it costs more computationally
voxel_size = 0.01  # 1 cm, the best candidate every 1 cm Not too harsh

pcd_downsampled = filtered_pcd.voxel_down_sample(voxel_size=voxel_size)
o3d.visualization.draw_geometries([pcd_downsampled])  # Visualize it again but downsampled

# ESTIMATING NORMALS

# Using heuristics
nn_distance = np.mean(pcd_downsampled.compute_nearest_neighbor_distance())  # The mean of the distance of each point it will average distance

# 4 times the average distance
radius_normals = nn_distance * 4

pcd_downsampled.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normals, max_nn=16), fast_normal_computation=True)

pcd_downsampled.paint_uniform_color([0.6, 0.6, 0.6])
o3d.visualization.draw_geometries([pcd_downsampled, outliers])

# EXTRACTING AND SETTING PARAMETERS

# CTRL V - To get what you got from the visualization and copy paste front, lookat and up

front = [0.96819, 0.24767, -0.035419]
lookat = [0.00849, -0.208787, -0.4890704]
up = [0.0118675, 0.095549553, 0.995353928]
zoom = 0.239999999999

pcd = pcd_downsampled  # Set the current working point cloud

o3d.visualization.draw_geometries([pcd], zoom=zoom, front=front, lookat=lookat, up=up)

# RANSAC PLANAR SEGMENTATION

# How far should a point be to be considered an outlier to a specific planar shape
# for planes
pt_to_plane_dist = 0.02  # 2 cm

plane_model, inliers = pcd.segment_plane(distance_threshold=pt_to_plane_dist, ransac_n=3, num_iterations=1000)  # FIXED

# The minimum number of points to be used are 3 for plane
[a, b, c, d] = plane_model  # The first variable it outputs is the plane model

print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

inlier_cloud = pcd.select_by_index(inliers)
outlier_cloud = pcd.select_by_index(inliers, invert=True)
inlier_cloud.paint_uniform_color([1.0, 0, 0])
outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])

# Now we do the visualization with the parameters we set up
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], zoom=zoom, front=front, lookat=lookat, up=up)

# MULTI-ORDER RANSAC

max_plane_idx = 6  # How many planes are we expecting to find? 4 walls, 1 ceiling, 1 floor
pt_to_plane_dist = 0.02

segment_models = {}
segments = {}
rest = pcd
# We do it for every point in the iteration, 1,2,3,4,5,6

for i in range(max_plane_idx):  # Fixed missing colon
    colors = plt.get_cmap("tab20")(i)
    segment_models[i], inliers = rest.segment_plane(distance_threshold=pt_to_plane_dist, ransac_n=3, num_iterations=1000)
    segments[i] = rest.select_by_index(inliers)
    segments[i].paint_uniform_color(list(colors[:3]))
    rest = rest.select_by_index(inliers, invert=True)
    print("pass", i, "/", max_plane_idx, "done")

o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)] + [rest], zoom=zoom, front=front, lookat=lookat, up=up)

# DBSCAN sur rest

labels = np.array(rest.cluster_dbscan(eps=0.05, min_points=5))
max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")

colors = plt.get_cmap("tab10")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
rest.colors = o3d.utility.Vector3dVector(colors[:, :3])

o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)] + [rest], zoom=zoom, front=front, lookat=lookat, up=up)
