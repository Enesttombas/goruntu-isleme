"""Hafta 7 / 7.7: Resim alıp harmonik matrisini döndüren 2 boyutlu DFT.

H[u, v] = sum_x sum_y f[x, y] * e^(-j*2*pi*(u*x/M + v*y/N))

2B DFT ayrılabilir: önce her satırın 1B DFT'si, sonra çıkan matrisin her sütununun 1B DFT'si alınır.
1B dönüşüm için 7.4'teki kendi fft fonksiyonumuz kullanılıyor, bu yüzden resim boyutları 2'nin kuvveti olmalı.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

KLASOR = Path(__file__).resolve().parent
GIRDI = KLASOR.parent / "girdi" / "kamera.png"
CIKTI = KLASOR / "cikti"
for klasor in ["7.1-sinus-vektor", "7.2-gurultu-ekleme", "7.4-dft-harmonik"]:
    sys.path.insert(0, str(KLASOR.parent / klasor))

from dft import fft, ifft  # noqa: E402


def dft2(resim):
    satirlar = fft(np.asarray(resim, dtype=float))  # her satırın DFT'si
    return fft(satirlar.T).T                         # sonra her sütunun DFT'si


def idft2(harmonik):
    return ifft(ifft(harmonik).T).T


def merkeze_al(harmonik):
    """DC bileşenini (0, 0) köşesinden matrisin ortasına taşır, görüntülemek için."""
    m, n = harmonik.shape
    return np.roll(harmonik, (m // 2, n // 2), axis=(0, 1))


def spektrum_resmi(harmonik):
    """log(1 + |H|) ile ölçeklenmiş, ortalanmış genlik spektrumu (0-255)."""
    genlik = np.log1p(np.abs(merkeze_al(harmonik)))
    return (255 * genlik / genlik.max()).astype(np.uint8)


def resim_oku(yol=GIRDI):
    resim = cv2.imread(str(yol), cv2.IMREAD_GRAYSCALE)
    if resim is None:
        raise FileNotFoundError(f"Resim okunamadı: {yol}")
    return resim


def kaydet(ad, goruntu):
    if not cv2.imwrite(str(CIKTI / ad), goruntu):
        raise OSError(f"Kaydedilemedi: {ad}")


def calistir():
    CIKTI.mkdir(exist_ok=True)
    resim = resim_oku()

    bas = time.perf_counter()
    harmonik = dft2(resim)
    sure = time.perf_counter() - bas
    numpy_harmonik = np.fft.fft2(resim)
    geri = idft2(harmonik)

    print(f"Resim boyutu: {resim.shape}, harmonik matris boyutu: {harmonik.shape}, süre {sure * 1000:.0f} ms")
    print(f"max|dft2 - np.fft.fft2| / max|H| = {np.abs(harmonik - numpy_harmonik).max() / np.abs(numpy_harmonik).max():.2e}")
    print(f"max|idft2(dft2(resim)) - resim| = {np.abs(geri - resim).max():.2e}")
    print(f"H[0,0] = {harmonik[0, 0].real:.0f} = piksel toplamı {int(resim.sum())} (ortalama parlaklık x {resim.size})")

    spektrum = spektrum_resmi(harmonik)
    kaydet("harmonik_matris.png", spektrum)
    kaydet("resim_ve_harmonik_matris.png", np.hstack([resim, np.full((resim.shape[0], 8), 255, np.uint8), spektrum]))


if __name__ == "__main__":
    calistir()
