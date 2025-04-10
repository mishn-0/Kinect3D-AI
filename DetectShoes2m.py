#Kinda works, still need to manually adjust parameters

import numpy as np
import cv2
from pyk4a import PyK4A, Config, ColorResolution, DepthMode

# Initialize Kinect
k4a = PyK4A(
    Config(
        color_resolution=ColorResolution.RES_720P,
        depth_mode=DepthMode.NFOV_UNBINNED,
        synchronized_images_only=True,
    )
)
k4a.start()

# Define depth threshold for segmentation (e.g., 1 meter to 2 meters)
min_depth = 2225  # Minimum depth (1000mm = 1m)
max_depth = 2400  # Maximum depth (2000mm = 2m)

while True:
    capture = k4a.get_capture()
    if capture.depth is not None:
        # Capture depth map
        depth_map = capture.transformed_depth

        # Apply depth threshold to create a mask for valid regions
        valid_depth_mask = (depth_map >= min_depth) & (depth_map <= max_depth)

        # Segment depth map: Keep only valid areas within the specified depth range
        segmented_depth = np.where(valid_depth_mask, depth_map, 0)

        # Normalize depth image for visualization
        depth_normalized = cv2.normalize(segmented_depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_normalized = np.uint8(depth_normalized)

        # Visualize the segmented depth image
        cv2.imshow("Segmented Depth Image", depth_normalized)

        # Optionally, segment the RGB image in the same way (optional)
        if capture.color is not None:
            rgb = capture.color
            rgb_segmented = cv2.bitwise_and(rgb, rgb, mask=valid_depth_mask.astype(np.uint8))

            # Display RGB with segmentation
            cv2.imshow("Segmented RGB Image", rgb_segmented)

        # Check for user input to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

k4a.stop()
cv2.destroyAllWindows()
