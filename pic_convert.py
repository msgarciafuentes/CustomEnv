import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image with original bit depth
img_path = "cam_data/depth_image_20250515-153538-745.png"
img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

print(f"Original dtype: {img.dtype}, min: {img.min()}, max: {img.max()}")

# If image is uint16, convert as suggested
if img.dtype == np.uint16:
    img_float = img.astype(np.float32)
    img_scaled = np.round(img_float / 256.0)
    img_uint8 = img_scaled.astype(np.uint8)
else:
    img_uint8 = img  # No conversion needed

# Optional: show image
plt.imshow(img_uint8, cmap='gray')
plt.title("Converted Grayscale Image (uint8)")
plt.axis('off')
plt.show()
