import os
import time
from datetime import datetime
from pyk4a import PyK4A, Config, ColorResolution, DepthMode
import cv2

# Configuración de la cámara
k4a = PyK4A(
    Config(
        color_resolution=ColorResolution.RES_720P,
        depth_mode=DepthMode.NFOV_UNBINNED,
        synchronized_images_only=True,
    )
)
k4a.start()

# Directorios de guardado
output_dir = "capturas"
rgb_dir = os.path.join(output_dir, "rgb")
depth_dir = os.path.join(output_dir, "depth")

os.makedirs(rgb_dir, exist_ok=True)
os.makedirs(depth_dir, exist_ok=True)

print("Capturando imágenes cada 5 segundos. Presiona Ctrl+C para detener.")

try:
    while True:
        capture = k4a.get_capture()
        if capture.color is not None and capture.depth is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Guardar imagen RGB
            rgb_path = os.path.join(rgb_dir, f"{timestamp}_rgb.png")
            cv2.imwrite(rgb_path, capture.color)

            # Guardar imagen de profundidad como 16 bits
            depth_path = os.path.join(depth_dir, f"{timestamp}_depth.png")
            cv2.imwrite(depth_path, capture.transformed_depth)

            print(f"Imagen guardada: {timestamp}")

        time.sleep(5)

except KeyboardInterrupt:
    print("Captura detenida por el usuario.")

finally:
    k4a.stop()
