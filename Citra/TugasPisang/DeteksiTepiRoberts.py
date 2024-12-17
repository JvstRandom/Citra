import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt

# Fungsi untuk mendeteksi tepi menggunakan operator Roberts
def roberts_edge_detection(image):
    # Kernel untuk operator Roberts
    kernel_x = np.array([[1, 0], [0, -1]])
    kernel_y = np.array([[0, 1], [-1, 0]])

    # Konvolusi gambar dengan kernel Roberts
    grad_x = np.zeros_like(image, dtype=float)
    grad_y = np.zeros_like(image, dtype=float)

    # Padding untuk menghindari masalah boundary
    padded_image = np.pad(image, ((0, 1), (0, 1)), mode='constant', constant_values=0)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            grad_x[i, j] = np.sum(padded_image[i:i+2, j:j+2] * kernel_x)
            grad_y[i, j] = np.sum(padded_image[i:i+2, j:j+2] * kernel_y)

    # Menggabungkan gradien untuk mendapatkan magnitude tepi
    edge_roberts = np.sqrt(grad_x**2 + grad_y**2)
    edge_roberts = (edge_roberts / edge_roberts.max() * 255).astype(np.uint8)  # Normalisasi

    return edge_roberts

# Membaca gambar
image_path = "pisang/pisang matang/images (20).jpg"
image = imageio.imread(image_path)

if image is None:
    print("Gambar tidak ditemukan! Periksa kembali path gambar.")
else:
    # Konversi ke grayscale jika gambar berwarna
    if image.ndim == 3:
        image = np.dot(image[..., :3], [0.299, 0.587, 0.114])  # Konversi ke grayscale

    # Pastikan gambar dalam format uint8
    image = image.astype(np.uint8)

    # Deteksi tepi menggunakan operator Roberts
    edges = roberts_edge_detection(image)

    # Menampilkan hasil
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Roberts Edge Detection")
    plt.imshow(edges, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
