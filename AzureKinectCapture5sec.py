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

# Change detection threshold for SSIM
SSIM_THRESHOLD = 0.95  # You may need to adjust this threshold
SAVE_INTERVAL = 5  # Interval in seconds for saving (to avoid saving too quickly)

print("Displaying images in real-time. Saving if significant changes are detected. Press 'q' to quit.")

last_save_time = time.time()
last_rgb = None
last_depth = None

# SSIM window size
WIN_SIZE = 3  # A smaller window for SSIM (must be odd and <= smaller dimension of image)

while True:
    capture = k4a.get_capture()
    if capture.color is not None and capture.depth is not None:
        rgb = capture.color
        depth = capture.transformed_depth

        # Normalize depth image to 8-bit for saving/visualizing (optional 16-bit for more precision)
        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_normalized = np.uint8(depth_normalized)

        # Display images in real-time
        cv2.imshow("RGB", rgb)
        cv2.imshow("Depth", depth_normalized)

        # Detect changes (RGB and depth)
        save_image = False
        current_time = time.time()

        # Only attempt to save if enough time has passed
        if current_time - last_save_time >= SAVE_INTERVAL:
            if last_rgb is not None and last_depth is not None:
                # Compare RGB images using SSIM with a smaller window size
                rgb_ssim_value = ssim(last_rgb, rgb, multichannel=True, win_size=WIN_SIZE)

                # Compare Depth images using SSIM with a smaller window size
                depth_ssim_value = ssim(last_depth, depth, data_range=depth.max(), win_size=WIN_SIZE)

                # If SSIM is below the threshold, we save the images
                if rgb_ssim_value < SSIM_THRESHOLD or depth_ssim_value < SSIM_THRESHOLD:
                    save_image = True

            else:
                # Save the first image (initial capture)
                save_image = True

            if save_image:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                rgb_path = os.path.join(rgb_dir, f"{timestamp}_rgb.png")
                depth_path = os.path.join(depth_dir, f"{timestamp}_depth.png")
                cv2.imwrite(rgb_path, rgb)
                cv2.imwrite(depth_path, depth_normalized)
                print(f"⚡ Image saved due to detected change: {timestamp}")
                last_save_time = current_time  # Update last save time
                last_rgb = rgb.copy()
                last_depth = depth.copy()

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

k4a.stop()
cv2.destroyAllWindows()
