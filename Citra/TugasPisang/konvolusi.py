import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# Load the image (replace 'image.jpg' with your image file path)
image = imageio.imread('Citra\TugasPisang\pisang\pisang matang\coba.jpg')

# Show original image
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')
plt.show()

# Convert the image to grayscale
def rgb_to_grayscale(image):
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    # Grayscale conversion formula
    grayscale = 0.299 * r + 0.587 * g + 0.114 * b
    return grayscale

# Perform konvolusi
def konvolusi(image, kernel):
    # Get dimensions of the image and kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Define output image size (same as input image)
    output = np.zeros((image_height, image_width))

    # Calculate padding size (assuming kernel is square and odd-sized)
    pad_h = kernel_height // 2
    pad_w = kernel_width // 2

    # Pad the original image to handle edge cases (zero padding)
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

    # Perform convolution
    for i in range(image_height):
        for j in range(image_width):
            # Extract the region of interest (ROI) in the image that corresponds to the kernel
            region = padded_image[i:i + kernel_height, j:j + kernel_width]
            # Perform element-wise multiplication and sum the result
            output[i, j] = np.sum(region * kernel)
    
    return output

# Convert the image to grayscale
grayscale_image = rgb_to_grayscale(image)

# Define a convolution kernel (e.g., edge detection Sobel filter)
# Example kernel for edge detection
kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])

# Apply convolution
convolved_image = konvolusi(grayscale_image, kernel)

# Show the grayscale image
plt.subplot(1, 2, 1)
plt.imshow(grayscale_image, cmap='gray')
plt.title('Citra asli')
plt.axis('off')

# Show the convolved image
plt.subplot(1, 2, 2)
plt.imshow(convolved_image, cmap='gray')
plt.title('Citra Setelah Konvolusi ')
plt.axis('off')

plt.show()

# Display pixel values of the convolved image (optional)
print("Convolved Image Pixel Values:")
print(convolved_image)
