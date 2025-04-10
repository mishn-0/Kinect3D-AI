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

# Function to calculate dynamic depth thresholds
def get_dynamic_depth_threshold(depth_map):
    """Calculate dynamic depth range based on depth map statistics."""
    # Get the median depth value
    median_depth = np.median(depth_map)
    
    # Set the min and max depth as ± a percentage of the median depth
    min_depth = int(median_depth - (median_depth * 0.1))  # 10% below median
    max_depth = int(median_depth + (median_depth * 0.1))  # 10% above median
    
    # Ensure min_depth and max_depth are within reasonable bounds
    min_depth = max(1000, min_depth)  # Minimum depth should not be less than 1000 mm
    max_depth = min(4000, max_depth)  # Maximum depth should not exceed 4000 mm
    
    return min_depth, max_depth

while True:
    capture = k4a.get_capture()
    if capture.depth is not None:
        # Capture depth map
        depth_map = capture.transformed_depth

        # Calculate dynamic depth thresholds based on the depth map statistics
        min_depth, max_depth = get_dynamic_depth_threshold(depth_map)

        # Apply depth threshold to create a mask for valid regions
        valid_depth_mask = (depth_map >= min_depth) & (depth_map <= max_depth)

        # Segment depth map: Keep only valid areas within the specified depth range
        segmented_depth = np.where(valid_depth_mask, depth_map, 0)

        # Apply Gaussian blur to reduce noise in the segmented depth image (optional)
        segmented_depth = cv2.GaussianBlur(segmented_depth, (5, 5), 0)

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
