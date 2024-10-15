import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt

# Ambil Gambar greyscale
# img = imageio.imread('pisang/gambar/grey.jpeg')
img = imageio.imread('pisang/pisang matang/images (20).jpg')

# Cek apakah gambar sudah greyscale, kalau blm di convert
if img.ndim == 3:
    img = np.dot(img[..., :3], [0.299, 0.587, 0.114])  # Convert to grayscale

# convert ke uint8 jika perlu
img = img.astype(np.uint8)

# Get dimensi (lebar dan tinggi) dari citra
w, h = img.shape
total_pixels = w * h

# Get histogram (frekuensi) gambar
hist, bins = np.histogram(img.flatten(), bins=256, range=[0, 256])

# Hitung distribulasi kumulatif menggunakan fungsi
cdf = hist.cumsum()

# normalisasi distribulasi kumulatif
cdf_normalized = cdf * (2**8 - 1) / total_pixels

# Masukkan ke rumus Ko (Ko)
new_gray_values = np.round(cdf_normalized).astype(np.uint8)

# Petakan nilai abu-abu asli ke nilai baru menggunakan nilai penyetaraan yang dihitung
img_equalized = new_gray_values[img]

# Plot histogram asli dan histogram yang equalization secara berdampingan
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(img.flatten(), bins=256, range=[0, 256], color='gray')
plt.title('Histogram Citra Asli')

plt.subplot(1, 2, 2)
plt.hist(img_equalized.flatten(), bins=256, range=[0, 256], color='gray')
plt.title('Histogram Citra Equalization')

plt.show()

# Show the original and equalized images side by side
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Citra Asli')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_equalized, cmap='gray')
plt.title('Citra Setelah Equalizatin')
plt.axis('off')

plt.show()

# Save the equalized image
imageio.imwrite('equalized_image.png', img_equalized)
