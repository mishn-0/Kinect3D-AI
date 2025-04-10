import numpy as np
import cv2
import time
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

# Set a threshold for detecting empty space
depth_threshold = 800  # Adjust based on your setup (depth values are in mm)

print("Press 'q' to quit.")

# Control the printing frequency (in seconds)
print_interval = 5  # Print every 5 seconds

last_print_time = time.time()  # To track when to print

while True:
    capture = k4a.get_capture()
    if capture.depth is not None:
        # Get depth map from Kinect capture
        depth_map = capture.transformed_depth

        # Threshold to detect empty space: areas with depth below the threshold are considered empty
        empty_space_mask = depth_map < depth_threshold

        # Calculate the percentage of empty space
        empty_space_percentage = np.sum(empty_space_mask) / empty_space_mask.size * 100

        current_time = time.time()

        # Only print if enough time has passed
        if current_time - last_print_time >= print_interval:
            print(f"Empty space percentage: {empty_space_percentage:.2f}%")
            last_print_time = current_time  # Update the last print time

        # Visualize the empty space mask (optional)
        empty_space_visual = empty_space_mask.astype(np.uint8) * 255
        cv2.imshow("Empty Space Mask", empty_space_visual)

        # Check for user input to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Stop Kinect and clean up
k4a.stop()
cv2.destroyAllWindows()
