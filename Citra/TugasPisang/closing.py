import cv2
import numpy as np

# Membaca gambar
image = cv2.imread('Citra/TugasPisang/pisang/pisang matang/images (20).jpg', 0)  # 0 untuk membaca citra dalam grayscale

# Binarisasi citra
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Definisi kernel (struktur elemen)
kernel = np.ones((5, 5), np.uint8)  # Kernel ukuran 5x5

# Operasi closing
closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

# Menampilkan hasil
cv2.imshow('Original Image', image)
cv2.imshow('Binary Image', binary_image)
cv2.imshow('Closing', closing)
cv2.waitKey(0)
cv2.destroyAllWindows()
