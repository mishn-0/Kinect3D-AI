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
depth_threshold = 2100  # Depth threshold in mm (you may need to adjust this)
print_interval = 5  # Print occupied space percentage every 5 seconds
last_print_time = time.time()

# Variables for tracking initial depth, RGB, and ROI
roi = None  # To store ROI for segmentation
initial_depth_map = None  # Store depth map of the entire scene
initial_rgb_image = None  # Store RGB image for ROI selection

# Function to let the user select the ROI
def select_roi(rgb_image):
    print("Select ROI. Press ENTER to confirm.")
    roi = cv2.selectROI("Select ROI", rgb_image)
    cv2.destroyWindow("Select ROI")
    return roi

# Output directory for storing images
output_dir = "captures"
depth_dir = os.path.join(output_dir, "depth")
rgb_dir = os.path.join(output_dir, "rgb")
os.makedirs(depth_dir, exist_ok=True)
os.makedirs(rgb_dir, exist_ok=True)

# First scan and ROI selection
print("Scanning the initial scene. Please wait...")
while True:
    capture = k4a.get_capture()
    if capture.color is not None and capture.depth is not None:
        # Get RGB and depth maps from Kinect capture
        initial_rgb_image = capture.color
        initial_depth_map = capture.transformed_depth

        # Normalize the depth map for better contrast during ROI selection
        normalized_depth_map = cv2.normalize(initial_depth_map, None, 0, 255, cv2.NORM_MINMAX)
        normalized_depth_map = np.uint8(normalized_depth_map)

        # Display the RGB image to select ROI
        cv2.imshow("Initial RGB Image", initial_rgb_image)
        print("Select ROI by clicking and dragging the mouse. Press ENTER when done.")
        roi = select_roi(initial_rgb_image)
        print(f"ROI selected: {roi}")
        break

# Once ROI is selected, focus on the second scan of the ROI
while True:
    capture = k4a.get_capture()
    if capture.color is not None and capture.depth is not None:
        # Get RGB and depth maps from Kinect capture
        rgb_image = capture.color
        depth_map = capture.transformed_depth

        # Crop both the RGB and depth maps to the selected ROI
        x, y, w, h = roi
        rgb_image_roi = rgb_image[y:y+h, x:x+w]
        depth_map_roi = depth_map[y:y+h, x:x+w]

        # Apply Gaussian blur to reduce noise in the depth map (optional)
        depth_map_roi = cv2.GaussianBlur(depth_map_roi, (5, 5), 0)

        # Create mask for detecting occupied space (where depth is less than the threshold)
        occupied_mask = depth_map_roi < depth_threshold

        # Calculate the percentage of occupied space in the ROI
        occupied_pixels = np.sum(occupied_mask)  # Number of occupied pixels
        total_pixels = occupied_mask.size  # Total number of pixels in the ROI

        # Normalize the percentage: 100% if full, 0% if empty
        occupied_percentage = (occupied_pixels / total_pixels) * 100

        # Get the current time and print the occupied space percentage every print_interval seconds
        current_time = time.time()
        if current_time - last_print_time >= print_interval:
            print(f"Occupied space percentage in ROI: {occupied_percentage:.2f}%")
            last_print_time = current_time

        # Visualize the depth map and the occupied mask (optional)
        empty_space_visual = occupied_mask.astype(np.uint8) * 255
        cv2.imshow("Occupied Space Mask", empty_space_visual)

        # Save the depth image and RGB image every interval if you want
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        depth_path = os.path.join(depth_dir, f"{timestamp}_depth.png")
        rgb_path = os.path.join(rgb_dir, f"{timestamp}_rgb.png")
        cv2.imwrite(depth_path, depth_map_roi)
        cv2.imwrite(rgb_path, rgb_image_roi)

        # Check for user input to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Stop Kinect and clean up
k4a.stop()
cv2.destroyAllWindows()
