"""Hafta 7 / 7.1: Verilen vektörü sinüs değerleriyle dolduran fonksiyon.

x[n] = A * sin(2*pi*f*n/fs + faz),  n = 0..N-1
Frekans yerine periyot verilirse f = 1 / periyot alınır.
Vektör boyu 2'nin kuvveti olmalı (sonraki ödevlerde FFT bunu istiyor).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

KLASOR = Path(__file__).resolve().parent
CIKTI = KLASOR / "cikti"


def ikinin_kuvveti_mi(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def sinus_doldur(vektor, genlik, faz, frekans=None, periyot=None, ornekleme_frekansi=None):
    """vektor'ü yerinde sinüs değerleriyle doldurur, aynı vektörü geri döndürür.

    ornekleme_frekansi verilmezse len(vektor) alınır, yani vektör 1 saniyelik sinyal olur.
    """
    n = len(vektor)
    if not ikinin_kuvveti_mi(n):
        raise ValueError(f"Vektör boyu 2'nin kuvveti olmalı, gelen: {n}")
    if (frekans is None) == (periyot is None):
        raise ValueError("Frekans ya da periyottan yalnız biri verilmeli")
    if frekans is None:
        frekans = 1.0 / periyot
    fs = n if ornekleme_frekansi is None else ornekleme_frekansi

    t = np.arange(n) / fs
    vektor[:] = genlik * np.sin(2 * np.pi * frekans * t + faz)
    return vektor


def ciz(eksen, vektor, genlik, faz, frekans, fs, baslik):
    """Vektör elemanları nokta olarak, sürekli sinüs dalgası çizgi olarak çizilir."""
    t_vektor = np.arange(len(vektor)) / fs
    t_surekli = np.linspace(0, len(vektor) / fs, 2000, endpoint=False)
    dalga = genlik * np.sin(2 * np.pi * frekans * t_surekli + faz)

    (cizgi,) = eksen.plot(t_surekli, dalga, color="tab:blue", lw=1.2)
    noktalar, _, _ = eksen.stem(t_vektor, vektor, linefmt="tab:orange", markerfmt="o", basefmt="k-")
    noktalar.set_color("tab:orange")
    eksen.set_title(baslik)
    eksen.set_xlabel("zaman (s)")
    eksen.set_ylabel("genlik")
    eksen.grid(alpha=0.3)
    eksen.legend([cizgi, noktalar], ["sinüs dalgası", "vektör elemanları"], loc="upper right")


def calistir():
    CIKTI.mkdir(exist_ok=True)

    # 1) Frekans ile: N = 64, A = 2, f = 3 Hz, faz = 45 derece
    v1 = np.zeros(64)
    sinus_doldur(v1, genlik=2, faz=np.pi / 4, frekans=3)

    # 2) Periyot ile: N = 128, A = 1, T = 0.25 s (f = 4 Hz), faz = 0
    v2 = np.zeros(128)
    sinus_doldur(v2, genlik=1, faz=0, periyot=0.25)

    sekil, eksenler = plt.subplots(2, 1, figsize=(11, 7))
    ciz(eksenler[0], v1, 2, np.pi / 4, 3, 64, "N=64, A=2, f=3 Hz, faz=45°")
    ciz(eksenler[1], v2, 1, 0, 4, 128, "N=128, A=1, periyot=0.25 s (f=4 Hz), faz=0")
    sekil.tight_layout()
    sekil.savefig(CIKTI / "sinus.png", dpi=120)
    plt.close(sekil)

    print(f"v1 ilk 4 eleman: {np.round(v1[:4], 4)}, en büyük |değer|: {np.abs(v1).max():.4f}")
    print(f"v2 ilk 4 eleman: {np.round(v2[:4], 4)}, en büyük |değer|: {np.abs(v2).max():.4f}")
    try:
        sinus_doldur(np.zeros(100), 1, 0, frekans=1)
    except ValueError as hata:
        print(f"N=100 denemesi: {hata}")


if __name__ == "__main__":
    calistir()
