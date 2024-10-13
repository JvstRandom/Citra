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


# # Convert RGB to grayscale using the formula
# def rgb_to_grayscale(image):
#     r = image[:, :, 0]
#     g = image[:, :, 1]
#     b = image[:, :, 2]
#     grayscale = 0.299 * r + 0.587 * g + 0.114 * b
#     return grayscale


# Convert RGB to HSI with new normalization method
def rgb_to_hsi(image):
    r = image[:, :, 0].astype(float)
    g = image[:, :, 1].astype(float)
    b = image[:, :, 2].astype(float)

    # Calculate the sum of R, G, and B
    rgb_sum = r + g + b + 1e-10  # Adding a small value to avoid division by zero

    # Normalize each channel according to the new formula
    r_normalized = r / rgb_sum
    g_normalized = g / rgb_sum
    b_normalized = b / rgb_sum

    # Print normalized value
    print("R normalized Pixel Values:")
    print(r_normalized)

    print("G normalized Pixel Values:")
    print(g_normalized)

    print("B normalized Pixel Values:")
    print(b_normalized)

    # Calculate intensity
    intensity = (r + g + b) / 3.0

    # Calculate saturation
    min_rgb = np.minimum(np.minimum(r_normalized, g_normalized), b_normalized)
    saturation = 1 - (3 / (r_normalized + g_normalized + b_normalized + 1e-10)) * min_rgb

    # Calculate hue
    num = 0.5 * ((r_normalized - g_normalized) + (r_normalized - b_normalized))
    den = np.sqrt((r_normalized - g_normalized) ** 2 + (r_normalized - b_normalized) * (g_normalized - b_normalized))
    theta = np.arccos(num / (den + 1e-10))  # Adding a small value to avoid division by zero

    hue = np.where(b_normalized > g_normalized, 2 * np.pi - theta, theta)  # Adjust the hue when blue > green
    hue = hue / (2 * np.pi)  # Normalize hue to the range [0, 1]

    return hue, saturation, intensity


# # Convert the image to grayscale
# grayscale_image = rgb_to_grayscale(image)
#
# # Show the grayscale image
# plt.imshow(grayscale_image, cmap='gray')
# plt.title('Grayscale Image')
# plt.axis('off')
# plt.show()
#
# # Display grayscale pixel values
# print("Grayscale Image Pixel Values:")
# print(grayscale_image)

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
