import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# Load image
image_path = 'pisang/gambar/grey.jpeg'
image = imageio.imread(image_path)

# Load images for combination
image_path1 = 'pisang/pisang matang/images (27).jpg'
image11 = imageio.imread(image_path1)

image_path2 = 'pisang/pisang matang/images (28).jpg'
image22 = imageio.imread(image_path2)


# Show original image and its pixel values
def show_image_with_pixels(image, title):
    print(f"{title} Pixel Values:\n", image)
    plt.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
    plt.title(title)
    plt.axis('off')
    plt.show()


# Convert image to grayscale
def rgb_to_grayscale(image):
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]
    grayscale = 0.299 * r + 0.587 * g + 0.114 * b
    return grayscale


# Function to calculate normalized histogram (frequencies)
def calculate_normalized_histogram(image):
    # Get the total number of pixels
    total_pixels = image.size

    # Get the unique grayscale values and their counts
    unique, counts = np.unique(image, return_counts=True)

    # Normalize the counts (frequencies) by dividing by total number of pixels
    normalized_frequencies = counts / total_pixels

    return unique, normalized_frequencies


# Example usage:
if __name__ == "__main__":
    # Convert to grayscale

    show_image_with_pixels(image, 'Gambar Greyscale')

    # Display histogram before normalization
    plt.hist(image.ravel(), bins=256, range=(0, 255), color='gray')
    plt.title('Histogram Gambar Greyscale sebelum Normalisasi')
    plt.show()

    # Calculate normalized histogram (frequencies)
    unique_values, normalized_frequencies = calculate_normalized_histogram(image)

    # Plot normalized histogram using plt.hist() for efficiency
    plt.hist(image.ravel(), bins=256, range=(0, 255), color='gray',
             weights=np.ones_like(image.ravel()) / image.size)
    plt.title('Histogram Setelah Normalisasi (Frekuensi Ter-normalisasi)')
    plt.xlabel('Grayscale Value')
    plt.ylabel('Normalized Frequency')
    plt.show()

    # Show grayscale image after normalization (in this case, same image)
    show_image_with_pixels(image, 'Gambar Greyscale setelah Normalisasi')
