'''Capture RGB and depth data.

Detect empty space in the depth map based on a threshold.

Display both the RGB and depth images in real-time.

Save the images (both RGB and depth) when significant changes are detected using SSIM.

Print the percentage of empty space at regular intervals.'''


import os
import time
from datetime import datetime
import numpy as np
from pyk4a import PyK4A, Config, ColorResolution, DepthMode
import cv2
from skimage.metrics import structural_similarity as ssim

# Configure the Kinect
k4a = PyK4A(
    Config(
        color_resolution=ColorResolution.RES_720P,
        depth_mode=DepthMode.NFOV_UNBINNED,
        synchronized_images_only=True,
    )
)
k4a.start()

# Output directories for storing images
output_dir = "captures"
rgb_dir = os.path.join(output_dir, "rgb")
depth_dir = os.path.join(output_dir, "depth")
os.makedirs(rgb_dir, exist_ok=True)
os.makedirs(depth_dir, exist_ok=True)

# Set thresholds and parameters
SSIM_THRESHOLD = 0.95  # Threshold for SSIM to detect significant changes
depth_threshold = 800  # Depth threshold for detecting empty space (in mm)
SAVE_INTERVAL = 5  # Interval (in seconds) for saving images
print_interval = 5  # Print empty space percentage every 5 seconds
WIN_SIZE = 3  # SSIM window size (should be odd and <= smaller dimension of the image)

print("Displaying images in real-time. Saving if significant changes are detected and printing empty space percentage. Press 'q' to quit.")

last_save_time = time.time()
last_print_time = time.time()
last_rgb = None
last_depth = None

# Loop for capturing and processing data
while True:
    capture = k4a.get_capture()
    if capture.color is not None and capture.depth is not None:
        rgb = capture.color
        depth = capture.transformed_depth

        # Normalize depth image to 8-bit for saving/visualizing (optional 16-bit for more precision)
        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_normalized = np.uint8(depth_normalized)

        # Detect empty space in the depth map
        empty_space_mask = depth < depth_threshold
        empty_space_percentage = np.sum(empty_space_mask) / empty_space_mask.size * 100

        # Display the RGB and depth images in real-time
        cv2.imshow("RGB", rgb)
        cv2.imshow("Depth", depth_normalized)

        # Check if enough time has passed to save images
        current_time = time.time()
        save_image = False

        if current_time - last_save_time >= SAVE_INTERVAL:
            if last_rgb is not None and last_depth is not None:
                # Compare the current and previous RGB and depth images using SSIM
                rgb_ssim_value = ssim(last_rgb, rgb, multichannel=True, win_size=WIN_SIZE)
                depth_ssim_value = ssim(last_depth, depth, data_range=depth.max(), win_size=WIN_SIZE)

                # Save the images if SSIM is below the threshold (indicating a significant change)
                if rgb_ssim_value < SSIM_THRESHOLD or depth_ssim_value < SSIM_THRESHOLD:
                    save_image = True
            else:
                # Save the first image (initial capture)
                save_image = True

            if save_image:
                # Generate a timestamp for the image filenames
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                rgb_path = os.path.join(rgb_dir, f"{timestamp}_rgb.png")
                depth_path = os.path.join(depth_dir, f"{timestamp}_depth.png")

                # Save the RGB and depth images
                cv2.imwrite(rgb_path, rgb)
                cv2.imwrite(depth_path, depth_normalized)
                print(f"⚡ Image saved due to detected change: {timestamp}")
                last_save_time = current_time  # Update the last save time
                last_rgb = rgb.copy()
                last_depth = depth.copy()

        # Print the empty space percentage if enough time has passed
        if current_time - last_print_time >= print_interval:
            print(f"Empty space percentage: {100-empty_space_percentage:.2f}%")
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
