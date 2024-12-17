import math
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

threshold_values = {}
h = [1]

# Fungsi untuk menghitung histogram gambar
def Hist(img):
    row, col = img.shape 
    y = np.zeros(256)
    for i in range(0, row):
        for j in range(0, col):
            y[img[i, j]] += 1
    return y

# Fungsi untuk menghasilkan gambar biner berdasarkan threshold
def regenerate_img(img, threshold):
    row, col = img.shape 
    y = np.zeros((row, col))
    for i in range(0, row):
        for j in range(0, col):
            if img[i, j] >= threshold:
                y[i, j] = 255
            else:
                y[i, j] = 0
    return y

# Fungsi untuk menghitung jumlah piksel
def countPixel(h):
    cnt = 0
    for i in range(0, len(h)):
        if h[i] > 0:
           cnt += h[i]
    return cnt

# Fungsi untuk menghitung berat (weight) piksel
def wieght(s, e):
    w = 0
    for i in range(s, e):
        w += h[i]
    return w

# Fungsi untuk menghitung rata-rata (mean)
def mean(s, e):
    m = 0
    w = wieght(s, e)
    for i in range(s, e):
        m += h[i] * i
    return m / float(w)

# Fungsi untuk menghitung variansi (variance)
def variance(s, e):
    v = 0
    m = mean(s, e)
    w = wieght(s, e)
    for i in range(s, e):
        v += ((i - m) ** 2) * h[i]
    v /= w
    return v

# Fungsi untuk menentukan threshold optimal
def threshold(h):
    cnt = countPixel(h)
    for i in range(1, len(h)):
        vb = variance(0, i)
        wb = wieght(0, i) / float(cnt)
        mb = mean(0, i)
        
        vf = variance(i, len(h))
        wf = wieght(i, len(h)) / float(cnt)
        mf = mean(i, len(h))
        
        V2w = wb * vb + wf * vf
        V2b = wb * wf * (mb - mf)**2

        # Menyimpan hasil analisis dalam dictionary
        if not math.isnan(V2w):
            threshold_values[i] = V2w

# Fungsi untuk mendapatkan threshold optimal
def get_optimal_threshold():
    min_V2w = min(threshold_values.values())
    optimal_threshold = [k for k, v in threshold_values.items() if v == min_V2w]
    print('Optimal Threshold:', optimal_threshold[0])
    return optimal_threshold[0]

# Memuat gambar dan mengubahnya menjadi grayscale
image = Image.open('pisang/pisang matang/coba.jpg').convert("L")
img = np.asarray(image)

# Menghitung histogram
h = Hist(img)

# Menentukan threshold menggunakan metode Otsu
threshold(h)
op_thres = get_optimal_threshold()

# Menerapkan threshold pada gambar untuk membuat citra biner
res = regenerate_img(img, op_thres)

# Menampilkan histogram dan hasil thresholding secara sejajar
plt.figure(figsize=(10, 4))

# Histogram
plt.subplot(1, 2, 1)
x = np.arange(256)
plt.bar(x, h, color='b', width=5, align='center', alpha=0.25)
plt.axvline(op_thres, color='red', linestyle='--', linewidth=2, label=f'Threshold: {op_thres}')
plt.title("Histogram Gambar")
plt.xlabel("Nilai Piksel")
plt.ylabel("Frekuensi")
plt.legend()

# Hasil Thresholding
plt.subplot(1, 2, 2)
plt.imshow(res, cmap='gray')
plt.title("Hasil Thresholding Otsu")
plt.axis('off')

# Tampilkan plot
plt.tight_layout()
plt.show()
