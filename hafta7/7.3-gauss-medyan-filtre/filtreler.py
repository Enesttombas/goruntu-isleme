"""Hafta 7 / 7.3: 7.2'deki gürültülü sinyali Gauss ve medyan filtreyle temizleme.

Gauss: her eleman, komşularının Gauss ağırlıklı ortalaması olur (çekirdek yarıçapı 3*sigma).
Medyan: her eleman, pencere içindeki değerlerin ortanca değeri olur.
Kenarlarda vektör aynalanarak (reflect) uzatılır, çıktı boyu girdiyle aynı kalır.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

KLASOR = Path(__file__).resolve().parent
CIKTI = KLASOR / "cikti"
sys.path.insert(0, str(KLASOR.parent / "7.1-sinus-vektor"))
sys.path.insert(0, str(KLASOR.parent / "7.2-gurultu-ekleme"))

from gurultu import TOHUM, gurultu_ekle, temiz_sinyal  # noqa: E402

ORAN = 0.3
SIGMA = 3
PENCERE = 9


def gauss_cekirdek(sigma):
    r = int(np.ceil(3 * sigma))
    x = np.arange(-r, r + 1)
    cekirdek = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return cekirdek / cekirdek.sum()


def gauss_filtre(x, sigma):
    cekirdek = gauss_cekirdek(sigma)
    r = len(cekirdek) // 2
    pencereler = sliding_window_view(np.pad(x, r, mode="reflect"), len(cekirdek))
    return pencereler @ cekirdek


def medyan_filtre(x, pencere):
    if pencere % 2 == 0:
        raise ValueError("Medyan penceresi tek sayı olmalı")
    r = pencere // 2
    pencereler = sliding_window_view(np.pad(x, r, mode="reflect"), pencere)
    return np.median(pencereler, axis=1)


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def calistir():
    CIKTI.mkdir(exist_ok=True)
    temiz = temiz_sinyal()
    gurultulu = gurultu_ekle(temiz, ORAN, np.random.default_rng(TOHUM))
    gauss = gauss_filtre(gurultulu, SIGMA)
    medyan = medyan_filtre(gurultulu, PENCERE)
    t = np.arange(len(temiz)) / len(temiz)

    sekil, eksenler = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    satirlar = [
        (gurultulu, "tab:orange", f"Gürültülü (oran {ORAN}), MSE = {mse(gurultulu, temiz):.4f}"),
        (gauss, "tab:green", f"Gauss filtre (sigma = {SIGMA}), MSE = {mse(gauss, temiz):.4f}"),
        (medyan, "tab:red", f"Medyan filtre (pencere = {PENCERE}), MSE = {mse(medyan, temiz):.4f}"),
    ]
    for eksen, (sinyal, renk, baslik) in zip(eksenler, satirlar):
        eksen.plot(t, temiz, color="tab:blue", lw=1.2, label="orijinal")
        eksen.plot(t, sinyal, color=renk, lw=1.0, label="sonuç")
        eksen.set_title(baslik)
        eksen.grid(alpha=0.3)
        eksen.legend(loc="upper right")
    eksenler[-1].set_xlabel("zaman (s)")
    sekil.tight_layout()
    sekil.savefig(CIKTI / "gauss_medyan.png", dpi=120)
    plt.close(sekil)

    print(f"Gürültülü MSE: {mse(gurultulu, temiz):.5f}")
    print("Gauss sigma -> MSE:", ", ".join(f"{s}: {mse(gauss_filtre(gurultulu, s), temiz):.5f}" for s in [1, 2, 3, 5, 8]))
    print("Medyan pencere -> MSE:", ", ".join(f"{k}: {mse(medyan_filtre(gurultulu, k), temiz):.5f}" for k in [3, 5, 9, 15, 25]))
    print(f"Tepe değeri (orijinal 1.0): gauss {gauss.max():.3f}, medyan {medyan.max():.3f}")


if __name__ == "__main__":
    calistir()
