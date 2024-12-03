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

# Define thresholds (example: 3 levels)
thresholds = [64, 128, 192]  # Adjust these values for your specific use case

# Apply multilevel thresholding
output_image = np.zeros_like(img)

# Assign levels based on thresholds
for i, threshold in enumerate(thresholds):
    if i == 0:
        output_image[img <= threshold] = int(255 / len(thresholds)) * i
    else:
        output_image[(img > thresholds[i - 1]) & (img <= threshold)] = int(255 / len(thresholds)) * i

# Assign the highest level to remaining pixels
output_image[img > thresholds[-1]] = 255

# Plot the original grayscale and multilevel thresholded images
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(output_image, cmap='gray')
plt.title('Multilevel Thresholding')
plt.axis('off')

plt.show()

# Save the multilevel thresholded image
imageio.imwrite('multilevel_thresholded_image.png', output_image)
