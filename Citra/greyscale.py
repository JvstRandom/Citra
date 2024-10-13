import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# Load the image (replace 'image.jpg' with your image file path)
image = imageio.imread('pisang/pisang matang/3c680633c8138e2856ede6ce0722a17f.jpg')

# Show original image
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')
plt.show()

print(image)

def rgb_to_grayscale(image):
    # Extract R, G, B channels
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    # print rgb matriks
    print("R= ", r)
    print("G= ", g)
    print("B= ", b)

    # Rumus Grayscale
    grayscale = 0.299 * r + 0.587 * g + 0.114 * b
    return grayscale


# Convert the image to grayscale
grayscale_image = rgb_to_grayscale(image)

# Show the grayscale image
plt.imshow(grayscale_image, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')
plt.show()

# Display pixel values of the grayscale image
print("Grayscale Image Pixel Values:")
print(grayscale_image)
