import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt

def hist_plot(img):

    # list kosong untuk mwnyimpan jumlah dari setiap nilai intensitas
    count = []

    # List untuk menyimpan nilai intensitas
    r = list(range(256))

    # loop untuk melintasi setiap nilai intensitas dan menghitung piksel
    for k in range(256):
        count_k = np.sum(img == k)
        count.append(count_k)

    return r, count


# Ambil gambar greyscale
# img = imageio.imread('pisang/gambar/saturnus.jpeg')
img = imageio.imread('Citra/TugasPisang/pisang/pisang matang/images (20).jpg')

# Cek apakah gambar sudah greyscale, kalau blm di convert
if img.ndim == 3:
    img = np.dot(img[..., :3], [0.299, 0.587, 0.114])  # Convert to grayscale

# convert ke uint8(unsigned 8-bit integer) jika perlu
img = img.astype(np.uint8)

# untuk memastikan jumlah kolom dan baris di citra
m, n = img.shape

# plot histogram original
r1, count1 = hist_plot(img)
plt.stem(r1, count1)
plt.xlabel('Nilai Intensitas')
plt.ylabel('Frekuensi')
plt.title('Histogram dari Gambar Asli')
plt.show()

# minimal pixel dan max pixel dari citra
min_pixel_value = img.min()
max_pixel_value = img.max()

# masukkan di rumus stretching
img_stretch = ((img - min_pixel_value) / (max_pixel_value - min_pixel_value) * 255).astype(np.uint8)

# plot histogram yang sudah normalisasi
r2, count2 = hist_plot(img_stretch)
plt.stem(r2, count2)
plt.xlabel('nilai intensitas')
plt.ylabel('Jumlah Pixel/frekuensi')
plt.title('Histogram dari Normalisasi Strech CItra')
plt.show()

# simpan gambar yang sudah normalisasi
imageio.imwrite('stretched_image.png', img_stretch)

# Tampilkan gambar asli dan gambar yang diregangkan secara berdampingan
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Citra Asli')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_stretch, cmap='gray')
plt.title('Citra setelah normalisasi strech')
plt.axis('off')

plt.show()
