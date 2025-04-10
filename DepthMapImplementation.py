import os
import time
import numpy as np
import cv2
from pyk4a import PyK4A, Config, ColorResolution, DepthMode
from datetime import datetime

# Initialize Kinect
k4a = PyK4A(
    Config(
        color_resolution=ColorResolution.RES_720P,
        depth_mode=DepthMode.NFOV_UNBINNED,
        synchronized_images_only=True,
    )
)
k4a.start()

# Set a threshold for detecting occupied space
depth_threshold = 300  # Depth threshold in mm (for detecting occupied space)
print_interval = 5  # Print occupied space percentage every 5 seconds
last_print_time = time.time()

# Variables for tracking initial depth and ROI
roi = None
initial_depth_map = None
shelf_mask = None
first_depth_value = None
shelf_excluded = False

# Normalize depth map to 8-bit for visualization
def normalize_depth(depth_map):
    depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(depth_normalized)

# Let user select region of interest
def select_roi(depth_map):
    print("Select ROI. The window will close automatically once the ROI is selected.")
    roi = cv2.selectROI("Select ROI", depth_map)
    cv2.destroyWindow("Select ROI")
    return roi

# Output directory for saving depth images
output_dir = "captures"
depth_dir = os.path.join(output_dir, "depth")
os.makedirs(depth_dir, exist_ok=True)

# First scan and ROI selection
print("Scanning the initial scene. Please wait...")
while True:
    capture = k4a.get_capture()
    if capture.depth is not None:
        initial_depth_map = capture.transformed_depth

        if first_depth_value is None:
            first_depth_value = np.median(initial_depth_map)

        min_depth = np.min(initial_depth_map)
        max_depth = np.max(initial_depth_map)
        print(f"Initial Depth Map - Min: {min_depth}mm, Max: {max_depth}mm")

        normalized_depth_map = normalize_depth(initial_depth_map)

        cv2.imshow("Initial Depth Map", normalized_depth_map)
        print("Select ROI by clicking and dragging. Press ENTER when done.")
        roi = select_roi(normalized_depth_map)
        print(f"ROI selected: {roi}")
        break

# Monitoring loop
while True:
    capture = k4a.get_capture()
    if capture.depth is not None:
        depth_map = capture.transformed_depth
        normalized_depth_map = normalize_depth(depth_map)

        x, y, w, h = roi
        depth_map_roi = depth_map[y:y+h, x:x+w]
        depth_map_roi = cv2.GaussianBlur(depth_map_roi, (5, 5), 0)

        # Convert ROI to 8-bit image for edge detection
        depth_map_roi_8u = normalize_depth(depth_map_roi)

        # Apply Canny edge detection
        edges = cv2.Canny(depth_map_roi_8u, 50, 150)
        cv2.imshow("Canny Edges", edges)

        if not shelf_excluded and first_depth_value is not None:
            shelf_margin = (depth_map_roi / first_depth_value) * 100
            shelf_mask = (shelf_margin >= 95) & (shelf_margin <= 105)

            occupied_mask = depth_map_roi < depth_threshold
            occupied_mask = np.logical_and(occupied_mask, ~shelf_mask)
            shelf_excluded = True

            occupied_pixels = np.sum(occupied_mask)
            total_pixels = occupied_mask.size
            occupied_percentage = (occupied_pixels / total_pixels) * 100

            current_time = time.time()
            if current_time - last_print_time >= print_interval:
                print(f"Occupied space in ROI (excluding shelf): {occupied_percentage:.2f}%")
                last_print_time = current_time

            occupied_vis = occupied_mask.astype(np.uint8) * 255
            cv2.imshow("Occupied Space Mask", occupied_vis)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(os.path.join(depth_dir, f"{timestamp}_depth.png"), depth_map_roi)

        else:
            occupied_mask = depth_map_roi < depth_threshold

            occupied_pixels = np.sum(occupied_mask)
            total_pixels = occupied_mask.size
            occupied_percentage = (occupied_pixels / total_pixels) * 100

            current_time = time.time()
            if current_time - last_print_time >= print_interval:
                print(f"Occupied space in ROI: {occupied_percentage:.2f}%")
                last_print_time = current_time

            occupied_vis = occupied_mask.astype(np.uint8) * 255
            cv2.imshow("Occupied Space Mask", occupied_vis)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(os.path.join(depth_dir, f"{timestamp}_depth.png"), depth_map_roi)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cleanup
k4a.stop()
cv2.destroyAllWindows()
