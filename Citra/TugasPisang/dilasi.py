import cv2
import numpy as np

# Membaca gambar
image = cv2.imread("Citra/TugasPisang/pisang/pisang matang/images (20).jpg", cv2.IMREAD_GRAYSCALE)

# Cek apakah gambar berhasil dimuat
if image is None:
    print("Gambar tidak ditemukan!")
    exit()

# Membuat kernel untuk operasi dilasi (ukuran 5x5)
kernel = np.ones((5, 5), np.uint8)

# Operasi dilasi
dilated_image = cv2.dilate(image, kernel, iterations=1)

# Menampilkan gambar asli dan hasil dilasi
cv2.imshow("Original Image", image)
cv2.imshow("Dilated Image", dilated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
