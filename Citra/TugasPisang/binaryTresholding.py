import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt

# Load the grayscale image
img = imageio.imread('pisang/pisang matang/images (20).jpg')

# Check if the image is already grayscale; if not, convert it
if img.ndim == 3:
    img = np.dot(img[..., :3], [0.299, 0.587, 0.114])  # Convert to grayscale

# Convert to uint8 if necessary
img = img.astype(np.uint8)

# Binary thresholding
threshold = 128  # You can adjust the threshold value as needed
binary_image = (img > threshold).astype(np.uint8) * 255  # 255 for white, 0 for black

# Plot the original grayscale and binary images
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(binary_image, cmap='gray')
plt.title('Binary Thresholding')
plt.axis('off')

plt.show()

# Save the binary image
imageio.imwrite('binary_image.png', binary_image)
