import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt

# Path ke gambar input
image_path = 'pisang/pisang matang/images (20).jpg'
img = imageio.imread(image_path)

# Konversi ke grayscale jika gambar dalam format RGB
if img.ndim == 3:
    img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

# Konversi ke citra biner (0 dan 1)
threshold = 128
binary_image = (img >= threshold).astype(np.uint8)  # Nilai 0 dan 1

# Tampilkan hasil
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Gambar Grayscale')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(binary_image, cmap='gray')
plt.title('Citra Biner (0 dan 1)')
plt.axis('off')

plt.show()
