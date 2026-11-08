"""Hafta 7 / 7.4: DFT ile frekans uzayına dönüşüm, harmonik vektörü döndüren fonksiyon.

Tanım: X[k] = sum_{n=0}^{N-1} x[n] * e^(-j*2*pi*k*n/N),  k = 0..N-1

dft(): tanımın birebir hali, N x N'lik matrisle çarpım, O(N^2).
fft(): aynı sonucu veren hızlı hali (Cooley-Tukey, radix-2). Vektör çift ve tek indisli
       elemanlara bölünüp iki yarım DFT birleştirilir, O(N log N). Bu yüzden boy 2'nin kuvveti olmalı.
ifft(): ters dönüşüm, eşlenik hilesiyle: x = conj(fft(conj(X))) / N
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

KLASOR = Path(__file__).resolve().parent
CIKTI = KLASOR / "cikti"
sys.path.insert(0, str(KLASOR.parent / "7.1-sinus-vektor"))
sys.path.insert(0, str(KLASOR.parent / "7.2-gurultu-ekleme"))

from gurultu import TOHUM, gurultu_ekle, temiz_sinyal  # noqa: E402
from sinus import ikinin_kuvveti_mi  # noqa: E402

ORAN = 0.3


def dft(x):
    x = np.asarray(x, dtype=complex)
    n = len(x)
    k = np.arange(n)
    w = np.exp(-2j * np.pi * np.outer(k, k) / n)
    return w @ x


def fft(x):
    """Son eksen boyunca FFT. 2B dizi verilirse her satırı ayrı dönüştürür (7.7'de kullanılıyor)."""
    x = np.asarray(x, dtype=complex)
    n = x.shape[-1]
    if not ikinin_kuvveti_mi(n):
        raise ValueError(f"Uzunluk 2'nin kuvveti olmalı, gelen: {n}")
    if n == 1:
        return x.copy()
    cift = fft(x[..., 0::2])
    tek = fft(x[..., 1::2]) * np.exp(-2j * np.pi * np.arange(n // 2) / n)
    return np.concatenate([cift + tek, cift - tek], axis=-1)


def ifft(harmonik):
    harmonik = np.asarray(harmonik, dtype=complex)
    return np.conj(fft(np.conj(harmonik))) / harmonik.shape[-1]


def harmonik_vektoru(x):
    """Ödevin istediği fonksiyon: 2'nin kuvveti boyundaki vektörün harmonik vektörünü döndürür."""
    return fft(x)


def genlik_spektrumu(harmonik, fs):
    """Tek taraflı genlik: |X[k]| * 2 / N. Böylece f Hz'teki A genlikli sinüs, f'de A yüksekliğinde tepe verir."""
    n = len(harmonik)
    genlik = np.abs(harmonik[: n // 2 + 1]) * 2 / n
    genlik[0] /= 2
    genlik[-1] /= 2
    frekans = np.arange(n // 2 + 1) * fs / n
    return frekans, genlik


def calistir():
    CIKTI.mkdir(exist_ok=True)
    temiz = temiz_sinyal()  # N=512, fs=512 Hz, A=1, f=5 Hz
    gurultulu = gurultu_ekle(temiz, ORAN, np.random.default_rng(TOHUM))
    fs = len(temiz)

    # Doğrulama: tanım hali, hızlı hal ve numpy aynı sonucu vermeli
    bas = time.perf_counter()
    x_dft = dft(gurultulu)
    sure_dft = time.perf_counter() - bas
    bas = time.perf_counter()
    x_fft = harmonik_vektoru(gurultulu)
    sure_fft = time.perf_counter() - bas
    print(f"max|dft - fft| = {np.abs(x_dft - x_fft).max():.2e}, max|fft - np.fft.fft| = {np.abs(x_fft - np.fft.fft(gurultulu)).max():.2e}")
    print(f"max|ifft(fft(x)) - x| = {np.abs(ifft(x_fft) - gurultulu).max():.2e}")
    print(f"Süre (N=512): dft {sure_dft * 1000:.1f} ms, fft {sure_fft * 1000:.1f} ms")

    sekil, eksenler = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for eksen, sinyal, ad, renk in [(eksenler[0], temiz, "Gürültüsüz", "tab:blue"), (eksenler[1], gurultulu, f"Gürültülü (oran {ORAN})", "tab:orange")]:
        frekans, genlik = genlik_spektrumu(harmonik_vektoru(sinyal), fs)
        tepe = int(np.argmax(genlik))
        digerleri = np.delete(genlik, tepe)
        eksen.stem(frekans, genlik, linefmt=renk, markerfmt=" ", basefmt="k-")
        eksen.set_title(f"{ad}: tepe {frekans[tepe]:.0f} Hz'te genlik {genlik[tepe]:.3f}, diğer harmoniklerin ortalaması {digerleri.mean():.4f}")
        eksen.set_ylabel("genlik")
        eksen.grid(alpha=0.3)
        print(f"{ad}: tepe {frekans[tepe]:.0f} Hz, genlik {genlik[tepe]:.4f}, diğerleri ort {digerleri.mean():.2e}, en büyüğü {digerleri.max():.4f}")
    eksenler[-1].set_xlabel("frekans (Hz)")
    sekil.tight_layout()
    sekil.savefig(CIKTI / "harmonik_vektor.png", dpi=120)
    plt.close(sekil)


if __name__ == "__main__":
    calistir()
