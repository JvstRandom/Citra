import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# Load image
image_path = 'pisang/pisang matang/3c680633c8138e2856ede6ce0722a17f.jpg'
image = imageio.imread(image_path)

# Load image buat combine
image_path1 = 'pisang/pisang matang/images (27).jpg'
image11 = imageio.imread(image_path1)

image_path2 = 'pisang/pisang matang/images (28).jpg'
image22 = imageio.imread(image_path2)

def show_image_with_pixels(image, title):
    print(f"{title} Pixel Values:\n", image)
    plt.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Greyscale
def rgb_to_grayscale(image):
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]
    grayscale = 0.299 * r + 0.587 * g + 0.114 * b
    return grayscale


# Negative
def negative_image(image):
    negative = 255 - image
    return negative


# Brighten
def brighten_image(image, factor=1.5):
    brightened = np.clip(image * factor, 0, 255).astype(np.uint8)
    return brightened


# Kombinasi
def combine_images(image1, image2):
    # Ensure both images have the same size and number of channels
    if image1.shape != image2.shape:
        raise ValueError("Both images must have the same dimensions.")
    combined = (image1 / 2 + image2 / 2).astype(np.uint8)
    return combined


# Translasi
def translate_image(image, x_shift, y_shift):
    translated_image = np.zeros_like(image)
    rows, cols = image.shape[:2]

    for i in range(rows):
        for j in range(cols):
            new_i = i + y_shift
            new_j = j + x_shift
            if 0 <= new_i < rows and 0 <= new_j < cols:
                translated_image[new_i, new_j] = image[i, j]
    return translated_image


# Rotasi
def rotate_image(image, angle):
    angle_rad = np.deg2rad(angle)
    rows, cols = image.shape[:2]
    rotated_image = np.zeros_like(image)
    center_i, center_j = rows // 2, cols // 2

    for i in range(rows):
        for j in range(cols):
            y = i - center_i
            x = j - center_j

            new_i = int(center_i + y * np.cos(angle_rad) - x * np.sin(angle_rad))
            new_j = int(center_j + y * np.sin(angle_rad) + x * np.cos(angle_rad))

            if 0 <= new_i < rows and 0 <= new_j < cols:
                rotated_image[new_i, new_j] = image[i, j]
    return rotated_image


# Flipping
def flip_image(image, flip_code):
    if flip_code == 0:  # Vertical flip
        flipped_image = image[::-1, :]
    elif flip_code == 1:  # Horizontal flip
        flipped_image = image[:, ::-1]
    elif flip_code == -1:  # Both horizontal and vertical
        flipped_image = image[::-1, ::-1]
    else:
        raise ValueError("Flip code must be 0, 1, or -1.")
    return flipped_image


# Zoom
def zoom_image(image, zoom_factor):
    rows, cols = image.shape[:2]
    center_i, center_j = rows // 2, cols // 2

    radius_i, radius_j = int(center_i / zoom_factor), int(center_j / zoom_factor)
    cropped_image = image[center_i - radius_i:center_i + radius_i, center_j - radius_j:center_j + radius_j]

    zoomed_image = np.zeros_like(image)
    zoomed_image[:cropped_image.shape[0], :cropped_image.shape[1]] = cropped_image

    return zoomed_image


def main():
    # Display original image and its pixel values
    show_image_with_pixels(image, 'Original Image')

    # User chooses an operation
    print("Choose an operation:")
    print("1. Negative Photo")
    print("2. Grayscale Photo")
    print("3. Image Brightening")
    print("4. Combine Two Photos (requires a second image)")
    print("5. Photo Translation")
    print("6. Photo Rotation")
    print("7. Photo Flipping")
    print("8. Photo Zooming")

    choice = int(input("Enter the number of the operation you want to perform (1-8): "))

    # Perform chosen operation
    if choice == 1:
        modified_image = negative_image(image)
        show_image_with_pixels(modified_image, 'Negative Image')

    elif choice == 2:
        modified_image = rgb_to_grayscale(image)
        show_image_with_pixels(modified_image, 'Grayscale Image')

    elif choice == 3:
        factor = float(input("Enter brightening factor (default 1.5): "))
        modified_image = brighten_image(image, factor)
        show_image_with_pixels(modified_image, 'Brightened Image')

    elif choice == 4:
        modified_image = combine_images(image11, image22)
        show_image_with_pixels(modified_image, 'Combined Image')

    elif choice == 5:
        x_shift = int(input("Enter x shift: "))
        y_shift = int(input("Enter y shift: "))
        modified_image = translate_image(image, x_shift, y_shift)
        show_image_with_pixels(modified_image, 'Translated Image')

    elif choice == 6:
        angle = float(input("Enter rotation angle: "))
        modified_image = rotate_image(image, angle)
        show_image_with_pixels(modified_image, 'Rotated Image')

    elif choice == 7:
        flip_code = int(input("Enter flip code (0 for vertical, 1 for horizontal, -1 for both): "))
        modified_image = flip_image(image, flip_code)
        show_image_with_pixels(modified_image, 'Flipped Image')

    elif choice == 8:
        zoom_factor = float(input("Enter zoom factor (e.g., 2 for 2x zoom): "))
        modified_image = zoom_image(image, zoom_factor)
        show_image_with_pixels(modified_image, 'Zoomed Image')

    else:
        print("Invalid choice. Please choose a number between 1 and 8.")


if __name__ == "__main__":
    main()
