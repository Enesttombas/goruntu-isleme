"""Hafta 1 / Ödev 1: OpenCV CUDA destekli mi? Colab'de de, yerelde de çalışır."""
import cv2
import numpy as np

print("OpenCV sürümü:", cv2.__version__)
adet = cv2.cuda.getCudaEnabledDeviceCount()
print("CUDA'lı cihaz sayısı:", adet)
if adet == 0:
    raise SystemExit("CUDA yok: pip'teki opencv-python CPU'dur, CUDA için kaynaktan derlenmiş OpenCV gerekir.")

cv2.cuda.printShortCudaDeviceInfo(cv2.cuda.getDevice())

# Küçük uçtan uca deneme: GPU'ya yükle, griye çevir, geri indir, CPU sonucuyla karşılaştır.
resim = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
g = cv2.cuda_GpuMat()
g.upload(resim)
gpu_gri = cv2.cuda.cvtColor(g, cv2.COLOR_BGR2GRAY).download()
cpu_gri = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
print("GPU-CPU en büyük fark:", int(np.abs(gpu_gri.astype(int) - cpu_gri).max()))
