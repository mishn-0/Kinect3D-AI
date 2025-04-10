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

voxel_size = 1   # The size of the little cube, it determines the level of detail and amount of downsampling, smaller means more points are preserved but it costs more computationally

pcd_downsampled = 


# ESTIMATING NORMALS

nn_distance =

radius_normals =

pcd_downsampled.estimate_normals()

#EXTRACTING AND SETTING PARAMETERS



