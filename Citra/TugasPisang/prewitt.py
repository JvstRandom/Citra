import cv2
import numpy as np

# Fungsi untuk deteksi tepi menggunakan Prewitt
def prewitt_edge_detection(image):
    # Konversi gambar ke grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Kernel Prewitt untuk arah X dan Y
    kernel_x = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]])
    kernel_y = np.array([[-1, -1, -1],
                         [ 0,  0,  0],
                         [ 1,  1,  1]])
    
    # Filter dengan kernel Prewitt
    edge_x = cv2.filter2D(gray, -1, kernel_x)
    edge_y = cv2.filter2D(gray, -1, kernel_y)
    
    # Gabungkan hasil deteksi arah X dan Y
    edge = cv2.magnitude(edge_x.astype(float), edge_y.astype(float))
    edge = np.clip(edge, 0, 255).astype(np.uint8)
    
    return edge

# Load gambar input
image_path = 'pisang/pisang matang/images (20).jpg'
image = cv2.imread(image_path)

if image is None:
    print("Gambar tidak ditemukan. Pastikan path sudah benar.")
else:
    # Deteksi tepi dengan Prewitt
    edges = prewitt_edge_detection(image)
    
    # Tampilkan hasil
    cv2.imshow("Original Image", image)
    cv2.imshow("Prewitt Edge Detection", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
