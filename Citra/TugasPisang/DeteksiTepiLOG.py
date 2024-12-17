import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt

# Fungsi untuk mendeteksi tepi menggunakan operator Laplace
def laplace_edge_detection(image):
    # Kernel untuk operator Laplace
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])

    # Konvolusi gambar dengan kernel Laplace
    edge_laplace = np.zeros_like(image, dtype=float)

    # Padding untuk menghindari masalah boundary
    padded_image = np.pad(image, ((1, 1), (1, 1)), mode='constant', constant_values=0)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            edge_laplace[i, j] = np.sum(padded_image[i:i+3, j:j+3] * kernel)

    # Ambil nilai absolut dan normalisasi
    edge_laplace = np.abs(edge_laplace)
    edge_laplace = (edge_laplace / edge_laplace.max() * 255).astype(np.uint8)

    return edge_laplace

# Membaca gambar
image_path = "pisang/pisang matang/images (20).jpg"  # Ganti dengan path gambar Anda
image = imageio.imread(image_path)

if image is None:
    print("Gambar tidak ditemukan! Periksa kembali path gambar.")
else:
    # Konversi ke grayscale jika gambar berwarna
    if image.ndim == 3:
        image = np.dot(image[..., :3], [0.299, 0.587, 0.114])  # Konversi ke grayscale

    # Pastikan gambar dalam format uint8
    image = image.astype(np.uint8)

    # Deteksi tepi menggunakan operator Laplace
    edges = laplace_edge_detection(image)

    # Menampilkan hasil
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Laplace Edge Detection")
    plt.imshow(edges, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
