import numpy as np
import torch
import torch.nn as nn
import open3d as o3d
from pyk4a import PyK4A, Config, DepthMode, ColorResolution

# ---- PointNet Model (simplified version) ----
class PointNet(nn.Module):
    def __init__(self, input_dim=3, output_dim=40):
        super(PointNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1024)
        self.relu = nn.ReLU()
        self.fc_final = nn.Linear(1024, output_dim)

    def forward(self, x):
        # x shape: (B, N, 3)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = torch.max(x, dim=1)[0]  # Global feature: max pooling
        x = self.fc_final(x)
        return x

# ---- Start Kinect ----
k4a = PyK4A(Config(color_resolution=ColorResolution.RES_720P,
                  depth_mode=DepthMode.NFOV_UNBINNED,
                  synchronized_images_only=True))
k4a.start()

print("Capturing frame...")
capture = k4a.get_capture()
k4a.stop()

# ---- Convert depth + color to point cloud using Open3D ----
depth_image = o3d.geometry.Image(capture.transformed_depth)
color_image = o3d.geometry.Image(capture.color)

rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_image, depth_image,
    convert_rgb_to_intensity=False,
    depth_scale=1000.0,
    depth_trunc=3.0  # Meters
)

intrinsic = o3d.camera.PinholeCameraIntrinsic(
    o3d.camera.PinholeCameraIntrinsicParameters.Kinect2DepthCameraDefault
)

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
pcd = pcd.voxel_down_sample(voxel_size=0.02)

# ---- Visualize point cloud (optional) ----
o3d.visualization.draw_geometries([pcd])

# ---- Convert point cloud to tensor (N, 3) ----
points = np.asarray(pcd.points)
if points.shape[0] > 1024:
    idx = np.random.choice(points.shape[0], 1024, replace=False)
    points = points[idx]
else:
    pad = 1024 - points.shape[0]
    points = np.pad(points, ((0, pad), (0, 0)))

points_tensor = torch.tensor(points, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 1024, 3)

# ---- Run through PointNet ----
model = PointNet(input_dim=3, output_dim=10)  # Assume 10 classes for example
output = model(points_tensor)
predicted_class = torch.argmax(output, dim=1)

print(f"Predicted class: {predicted_class.item()}")
