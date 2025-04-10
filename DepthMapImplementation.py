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
roi = None  # To store ROI for segmentation
initial_depth_map = None  # Store depth map of the entire scene
shelf_mask = None  # To store the mask of the shelf area
first_depth_value = None  # Store the first depth value for rule of three calculation
shelf_excluded = False  # Flag to ensure shelf exclusion only happens once

# Function to normalize depth map for better visualization
def normalize_depth(depth_map):
    # Normalize depth map to 8-bit range for visualization
    depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(depth_normalized)

# Function to let the user select the ROI
def select_roi(depth_map):
    print("Select ROI. The window will close automatically once the ROI is selected.")
    roi = cv2.selectROI("Select ROI", depth_map)
    cv2.destroyWindow("Select ROI")  # Automatically close the window after ROI is selected
    return roi

# Output directory for storing images
output_dir = "captures"
depth_dir = os.path.join(output_dir, "depth")
os.makedirs(depth_dir, exist_ok=True)

# First scan and ROI selection
print("Scanning the initial scene. Please wait...")

# STEP 1: Capture "empty" reference depth
print("Capturing reference 'empty' rack image...")
time.sleep(2)  # Let the camera warm up
while True:
    capture = k4a.get_capture()
    if capture.depth is not None:
        ref_depth = capture.transformed_depth
        print("Reference captured.")
        break

# Wait for ROI selection
while True:
    capture = k4a.get_capture()
    if capture.depth is not None:
        # Get depth map from Kinect capture
        initial_depth_map = capture.transformed_depth

        # Normalize the depth map for better visibility
        normalized_depth_map = normalize_depth(initial_depth_map)

        # Display the depth map to select ROI
        cv2.imshow("Initial Depth Map", normalized_depth_map)
        print("Select ROI by clicking and dragging the mouse. Press ENTER when done.")
        roi = select_roi(normalized_depth_map)
        print(f"ROI selected: {roi}")
        break

# Once ROI is selected, focus on the second scan of the ROI
while True:
    capture = k4a.get_capture()
    if capture.depth is not None:
        current_depth = capture.transformed_depth
        # STEP 2: Monitor for changes from reference
        depth_diff = np.abs(current_depth.astype(np.int32) - ref_depth.astype(np.int32))

        # Threshold to detect objects (tune based on setup)
        threshold = 100  # mm
        occupied_mask = (depth_diff > threshold) & (ref_depth != 0)

        # Create mask for detecting occupied space (where depth is less than the threshold)
        occupied_mask = depth_diff > depth_threshold

        # Calculate occupied percentage
        occupied_percent = np.sum(occupied_mask) / occupied_mask.size * 100

        # Print only every 5 seconds
        current_time = time.time()
        if current_time - last_print_time >= print_interval:
            print(f"📦 Occupied space: {occupied_percent:.2f}%")
            last_print_time = current_time

        # Optional visualization
        occupied_vis = occupied_mask.astype(np.uint8) * 255
        cv2.imshow("Occupied Mask", occupied_vis)

        # Save the depth image and mask every interval if you want
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        depth_path = os.path.join(depth_dir, f"{timestamp}_depth.png")
        cv2.imwrite(depth_path, current_depth)

    # Check for user input to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop Kinect and clean up
k4a.stop()
cv2.destroyAllWindows()
