import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt

# Membaca citra
img = imageio.imread('Citra/TugasPisang/pisang/pisang matang/images (20).jpg')

# Mengonversi citra ke grayscale
gray_img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])  # Menggunakan rumus RGB ke grayscale

# Mengonversi citra grayscale ke citra biner dengan threshold 0.65
binary_img = gray_img > 0.65 * 255

# Melakukan komplemen citra biner (invert)
binary_img = np.logical_not(binary_img)

# Menampilkan citra biner hasil komplemen
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(binary_img, cmap='gray')
plt.title('Citra Biner Hasil Komplemen')
plt.axis('off')

# Membuat elemen struktural untuk erosi (kernel) 5x5
kernel = np.ones((5, 5), np.uint8)

# Melakukan operasi erosi
eroded_img = np.zeros_like(binary_img)
rows, cols = binary_img.shape

# Erosi manual dengan memindahkan kernel ke setiap pixel
for i in range(2, rows-2):
    for j in range(2, cols-2):
        if np.all(binary_img[i-2:i+3, j-2:j+3] == 1):
            eroded_img[i, j] = 1

# Menampilkan citra hasil erosi
plt.subplot(1, 3, 2)
plt.imshow(eroded_img, cmap='gray')
plt.title('Hasil Erosi')
plt.axis('off')

# Deteksi tepi dengan mengurangkan citra erosi dari citra biner
edge_img = np.logical_and(binary_img, np.logical_not(eroded_img))

# Menampilkan hasil deteksi tepi
plt.subplot(1, 3, 3)
plt.imshow(edge_img, cmap='gray')
plt.title('Hasil Deteksi Tepi dengan Erosi')
plt.axis('off')
plt.show()
