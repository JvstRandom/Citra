import math

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# Load the image
image = imageio.imread('pisang/pisang matang/3c680633c8138e2856ede6ce0722a17f.jpg')

# Show original image
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')
plt.show()

# Print original RGB pixel values
print("Original Image Pixel Values (RGB):")
print(image)


# Convert RGB to grayscale using the formula
def rgb_to_grayscale(image):
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]
    grayscale = 0.299 * r + 0.587 * g + 0.114 * b
    return grayscale


# Convert RGB to HSI
def rgb_to_hsi(image):
    # Normalize the RGB values to [0, 1]
    image = image / 255.0

    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    # Calculate intensity
    intensity = (r + g + b) / 3.0

    # Calculate saturation
    min_rgb = np.minimum(np.minimum(r, g), b)
    saturation = 1 - (3 / (r + g + b + 1e-10)) * min_rgb

    # Calculate hue
    perhitunganAtas = 2 * r - g - b
    perhitunganBawah = 2 * math.sqrt((r - g) ** 2 + (r - b) * (g - b))
    if perhitunganBawah == 0:
        return 0
    cos_theta = perhitunganAtas / perhitunganBawah
    hueRad = math.acos(cos_theta)  # dalam bentuk radians

    # convert radians jadi degree
    hueDerajat = math.degrees(hueRad)

    # normalisasi hue jadi range [0,1]
    hue = hueDerajat / 360


    return hue, saturation, intensity


# Convert the image to grayscale
grayscale_image = rgb_to_grayscale(image)

# Show the grayscale image
plt.imshow(grayscale_image, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')
plt.show()

# Display grayscale pixel values
print("Grayscale Image Pixel Values:")
print(grayscale_image)

# Convert the image to HSI
hue, saturation, intensity = rgb_to_hsi(image)

# Show the HSI components
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(hue, cmap='hsv')  # Display hue component
plt.title('Hue')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(saturation, cmap='gray')  # Display saturation component
plt.title('Saturation')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(intensity, cmap='gray')  # Display intensity component
plt.title('Intensity')
plt.axis('off')

plt.show()

# Display pixel values of the HSI components
print("Hue Pixel Values:")
print(hue)

print("Saturation Pixel Values:")
print(saturation)

print("Intensity Pixel Values:")
print(intensity)
