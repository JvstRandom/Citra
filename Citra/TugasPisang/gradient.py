import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt

# Fungsi untuk mendeteksi tepi menggunakan operasi gradien
def gradient_edge_detection(image):
    # Konvolusi gambar dengan gradien sederhana
    grad_x = np.zeros_like(image, dtype=float)
    grad_y = np.zeros_like(image, dtype=float)

    # Padding untuk menghindari masalah boundary
    padded_image = np.pad(image, ((0, 1), (0, 1)), mode='constant', constant_values=0)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            grad_x[i, j] = padded_image[i, j + 1] - padded_image[i, j]  # Selisih horizontal
            grad_y[i, j] = padded_image[i + 1, j] - padded_image[i, j]  # Selisih vertikal

    # Menggabungkan gradien untuk mendapatkan magnitudo tepi
    edge_gradient = np.sqrt(grad_x**2 + grad_y**2)
    edge_gradient = (edge_gradient / edge_gradient.max() * 255).astype(np.uint8)  # Normalisasi

    return edge_gradient

# Membaca gambar
image_path = "Citra/TugasPisang/pisang/pisang matang/coba.jpg"  # Ganti dengan path gambar Anda
image = imageio.imread(image_path)

if image is None:
    print("Gambar tidak ditemukan! Periksa kembali path gambar.")
else:
    # Konversi ke grayscale jika gambar berwarna
    if image.ndim == 3:
        image = np.dot(image[..., :3], [0.299, 0.587, 0.114])  # Konversi ke grayscale

    # Pastikan gambar dalam format uint8
    image = image.astype(np.uint8)

    # Deteksi tepi menggunakan operasi gradien
    edges = gradient_edge_detection(image)

    # Menampilkan hasil
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Gradient Edge Detection")
    plt.imshow(edges, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
