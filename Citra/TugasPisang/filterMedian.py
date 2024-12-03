import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt

def median_filter(img):
    # dimensi gambar
    m, n = img.shape

    # buat matriks kosong untuk menyimpan gambar yang sudah di mask
    filtered_img = np.zeros((m, n), dtype=np.uint8)

    # ulangi ke setiap pixel kecuali yang ada di tepi
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            # Ekstrak 3x3 pixel
            neighborhood = [
                img[i - 1, j - 1], img[i - 1, j], img[i - 1, j + 1],
                img[i, j - 1], img[i, j], img[i, j + 1],
                img[i + 1, j - 1], img[i + 1, j], img[i + 1, j + 1]
            ]

            # Cari nilai median
            median_value = np.median(neighborhood)

            # Buat nilai median menjadi nilai yang ada di tengah
            filtered_img[i, j] = median_value

    return filtered_img


img = imageio.imread('pisang/pisang matang/images (20).jpg')
# img = imageio.imread('pisang/gambar/saturnus.jpeg')

# Cek apakah gambar sudah greyscale, kalau blm di convert
if img.ndim == 3:
    img = np.dot(img[..., :3], [0.299, 0.587, 0.114])

img = img.astype(np.uint8)

# panggil fungsi median filter ke gambar
filtered_img = median_filter(img)

# Tampilkan gambar asli dan hasil
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Gambar Asli')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(filtered_img, cmap='gray')
plt.title('Gambar yang sudah di Filter')
plt.axis('off')

plt.show()

# Simpan gambar yang sudah di filter
imageio.imwrite('median_filtered_image.png', filtered_img)
