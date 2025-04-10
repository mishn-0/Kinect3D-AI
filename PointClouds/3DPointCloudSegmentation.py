import numpy as np
import open3d as o3d
import matplotlib as plt

# Point Cloud Data Preparation

DATANAME = "appartment_cloud.ply"  #Change this to the data you want

pcd = o3d.io.read_point_cloud("../DATA/" + DATANAME)       #Point Cloud Data is an open 3d object, we use a relative path

# PRE PROCESSING

#We need to translate everything so that Open3D can visualize it correctly
pcd_center = pcd.get_center #Get the center of the data
pcd.translate(-pcd_center)  #we don't need to translate it in this case but we're doing it just in case


## STATISTICAL OUTLIER FILTER
#Some point clouds may have some noise

nn = 16        # 16 is the number of nearest neighbors

std_multiplier =            # We use the STD of our Point distribution and apply that on a multiplier, a PC above this number it's an outlier

filtered_pcd = pcd.remove_statistical_outlier(nn,std_multiplier)    #Now we can use this function to remove statistical outliers
#After filtering we keep the outliers elsewhere

outliers = pcd.select_by_index(filtered_pcd[1], invert=True)
outliers.paint_uniform.color(1,0,0) #We paint the outliers in RED so we can see them more clearly

filtered_pcd = filtered_pcd[1]

o3d.visualization.draw_geometries([filtered_pcd, outliers])

#VOXEL DOWNSAMPLING

# The size of the little cube, it determines the level of detail and amount of downsampling, smaller means more points are preserved but it costs more computationally
voxel_size = 0.01   # 1 cm, the best candidate every 1 cm Not too harsh

pcd_downsampled = filtered_pcd.voxel_down_sample(voxel_size = voxel_size)
o3d.visualization.draw_geometries([pcd_downsampled])    # Visualize it again but downsampled


# ESTIMATING NORMALS

#Using heuristics
nn_distance = np.mean(pcd.compute_nearest_neighbor_distance())  #The mean of the distance of each point it will average distance

# 4 times the average distance
radius_normals = nn_distance*4

pcd_downsampled.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normals, max_nn=16), fast_normal_computation=True)

pcd.downsampled.paint_uniform_color([0.6,0.6,0.6])
o3d.visualization.draw_geometries(pcd_downsampled,outliers)

#EXTRACTING AND SETTING PARAMETERS

#CTRL V    - To get what you got from the visualitzation anc opy paste front, look at and up


front = [0.96819, 0.24767, -0.035419]
lookat = [0.00849, -0.208787, -0.4890704]
up = [0.0118675, 0.095549553, 0.995353928]
zoom = 0.239999999999

pcd = 

o3d.visualization.draw_geometries([pcd],zoom=zoom, front=front, lookat=lookat,up=up)



#RANSAC PLANAR SEGMENTATION

# How far should a point be to be considered an outlier to a specific planar shape
# for planes 
pt_to_plane_dist - 0.02 # 2 cm


plane_model, inliers =          #H IDK WHY ITS EMPTY

# The minimum number of points to be used are 3 for plane
pcd.segment_plane(distance_threshold=pt_to_plane_dist, ransac_n=3,num_iterations=1000)

[a,b,c,d] = plane_model #The first variable it outputs is the plane model

print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

inlier_cloud = pcd.select_by_index(inliers)
outlier_cloud = pcd.select_by_index(inliers, invert = True)
inlier_cloud.paint_uniform_color([1.0, 0, 0])
outlier_cloud.paint_uniform_color([0.6,0.6,0.6])


# Now we do the visualization with the parameters we set up
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], zoom=zoom, front=front, lookat=lookat, up=up)

# MULTI-ORDER RANSAC

max_plane_idx = 6 # How many planes are we expecting to find? 4 walls, 1 ceiling, 1 floor
