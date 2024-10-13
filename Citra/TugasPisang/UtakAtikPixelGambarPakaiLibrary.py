import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import cv2

# Load the image (replace 'image.jpg' with your image file path)
image_path = 'pisang/pisang matang/3c680633c8138e2856ede6ce0722a17f.jpg'
image = imageio.imread(image_path)


# Show original image and its pixel values
def show_image_with_pixels(image, title):
    print(f"{title} Pixel Values:\n", image)
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()


# Convert image to grayscale
def rgb_to_grayscale(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return grayscale


# Make a negative of the image
def negative_image(image):
    negative = 255 - image
    return negative


# Brighten the image
def brighten_image(image, factor=1.5):
    brightened = np.clip(image * factor, 0, 255).astype(np.uint8)
    return brightened


# Arithmetic operation between two images (average here)
def combine_images(image1, image2):
    combined = cv2.addWeighted(image1, 0.5, image2, 0.5, 0)
    return combined


# Geometric operation: Translation
def translate_image(image, x_shift, y_shift):
    rows, cols = image.shape[:2]
    translation_matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    translated_image = cv2.warpAffine(image, translation_matrix, (cols, rows))
    return translated_image


# Geometric operation: Rotation
def rotate_image(image, angle):
    rows, cols = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    return rotated_image


# Geometric operation: Flipping
def flip_image(image, flip_code):
    flipped_image = cv2.flip(image, flip_code)
    return flipped_image


# Geometric operation: Zooming
def zoom_image(image, zoom_factor):
    center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
    radius_x, radius_y = center_x // zoom_factor, center_y // zoom_factor
    cropped_image = image[center_y - radius_y:center_y + radius_y, center_x - radius_x:center_x + radius_x]
    zoomed_image = cv2.resize(cropped_image, (image.shape[1], image.shape[0]))
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
        second_image_path = input("Enter the path for the second image: ")
        second_image = imageio.imread(second_image_path)
        modified_image = combine_images(image, second_image)
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
