import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# Load the image
image = imageio.imread('Citra\TugasPisang\pisang\pisang matang\coba.jpg')

# Show original image
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')
plt.show()

# Convert gambar to greyscale
def rgb_to_greyscale(image):
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    # rumus greyscale
    greyscale = 0.299 * r + 0.587 * g + 0.114 * b
    return greyscale

# Perform konvolusi
def konvolusi(image, kernel):
    # Mendapatkan dimensi gambar dan kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Mendefinisikan ukuran output gambar
    output = np.zeros((image_height, image_width))

    # hitung padding size
    pad_h = kernel_height // 2
    pad_w = kernel_width // 2

    # Menambahkan padding pada gambar asli untuk menangani kasus di tepi (padding nol)
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

    # Perform convolution
    for i in range(image_height):
        for j in range(image_width):
            # Menentukan bagian citra yang sesuai dengan kernel
            region = padded_image[i:i + kernel_height, j:j + kernel_width]
            # Melakukan perkalian elemen per elemen dan menjumlahkan hasilnya
            output[i, j] = np.sum(region * kernel)
    
    return output

# Convert gambar menjadi greyscale
greyscale_image = rgb_to_greyscale(image)

# Define a convolution kernel
# kernel deteksi tepi
kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])

# Apply konvolusi
convolved_image = konvolusi(greyscale_image, kernel)

# Show gambar greyscale
plt.subplot(1, 2, 1)
plt.imshow(greyscale_image, cmap='grey')
plt.title('Citra asli')
plt.axis('off')

# Show gambar hasil konvolusi
plt.subplot(1, 2, 2)
plt.imshow(convolved_image, cmap='grey')
plt.title('Citra Setelah Konvolusi ')
plt.axis('off')

plt.show()

# Menampilkan nilai piksel dari citra hasil konvolusi
print("Convolved Image Pixel Values:")
print(convolved_image)
