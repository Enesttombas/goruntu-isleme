"""Hafta 7 / 7.6: Eşik altındaki harmonikleri silip sinyali yeniden üreten frekans filtresi.

1. Harmonik vektör bulunur (7.4'teki fft).
2. Her harmoniğin genliği |X[k]| * 2 / N ile sinüs genliği birimine çevrilir.
3. Genliği eşikten küçük olan harmonikler 0 yapılır.
4. Ters DFT ile sinyal geri üretilir.
Gerçek sinyalde X[k] ve X[N-k] aynı genliktedir; ikisi birlikte silinir ya da kalır, sonuç gerçek sayı olur.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

KLASOR = Path(__file__).resolve().parent
CIKTI = KLASOR / "cikti"
for klasor in ["7.1-sinus-vektor", "7.2-gurultu-ekleme", "7.4-dft-harmonik", "7.5-iki-sinyal-toplam"]:
    sys.path.insert(0, str(KLASOR.parent / klasor))

from dft import fft, genlik_spektrumu, ifft  # noqa: E402
from iki_sinyal import FS, N, sinyalleri_olustur  # noqa: E402

ESIK = 1.0


def frekans_filtresi(x, esik):
    harmonik = fft(x)
    n = len(harmonik)
    genlik = np.abs(harmonik) * 2 / n
    genlik[0] /= 2       # DC bileşeni
    genlik[n // 2] /= 2  # Nyquist bileşeni
    tutulan = genlik >= esik
    yeni = np.where(tutulan, harmonik, 0)
    return np.real(ifft(yeni)), int(tutulan.sum())


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def calistir():
    CIKTI.mkdir(exist_ok=True)
    sinyaller = sinyalleri_olustur()
    t = np.arange(N) / FS
    goster = t < 0.1

    sekil, eksenler = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for eksen, (ad, (temiz, gurultulu)) in zip(eksenler, sinyaller.items()):
        filtreli, tutulan = frekans_filtresi(gurultulu, ESIK)
        eksen.plot(t[goster], gurultulu[goster], color="tab:orange", lw=0.8, label="gürültülü")
        eksen.plot(t[goster], temiz[goster], color="tab:blue", lw=2.5, alpha=0.6, label="orijinal (gürültüsüz)")
        eksen.plot(t[goster], filtreli[goster], "k--", lw=1.2, label="filtre sonrası")
        eksen.set_title(f"{ad}: RMSE gürültülü {rmse(gurultulu, temiz):.3f} -> filtreli {rmse(filtreli, temiz):.3f}, {N} harmonikten {tutulan} tanesi kaldı")
        eksen.set_ylabel("genlik")
        eksen.grid(alpha=0.3)
        eksen.legend(loc="upper right", fontsize=8)
        print(f"{ad}: eşik {ESIK}, kalan harmonik {tutulan}, RMSE {rmse(gurultulu, temiz):.4f} -> {rmse(filtreli, temiz):.4f}")
    eksenler[-1].set_xlabel("zaman (s)")
    sekil.tight_layout()
    sekil.savefig(CIKTI / "frekans_filtresi.png", dpi=120)
    plt.close(sekil)

    # Eşik seçimi: toplam sinyalde farklı eşikler
    temiz, gurultulu = sinyaller["toplam s1 + s2"]
    frekans, genlik = genlik_spektrumu(fft(gurultulu), FS)
    sekil, eksen = plt.subplots(figsize=(12, 4.5))
    eksen.stem(frekans[:150], genlik[:150], linefmt="tab:purple", markerfmt=" ", basefmt="k-")
    for esik, renk in [(ESIK, "tab:green"), (6.0, "tab:red")]:
        eksen.axhline(esik, color=renk, ls="--", label=f"eşik {esik}")
    eksen.set_xlabel("frekans (Hz)")
    eksen.set_ylabel("genlik")
    eksen.set_title("Toplam sinyalin harmonikleri ve eşikler")
    eksen.grid(alpha=0.3)
    eksen.legend()
    sekil.tight_layout()
    sekil.savefig(CIKTI / "esik_secimi.png", dpi=120)
    plt.close(sekil)

    print("Toplam sinyal, eşik -> kalan harmonik, RMSE:")
    for esik in [0.05, 0.1, 0.3, 1.0, 4.0, 6.0, 13.0]:
        filtreli, tutulan = frekans_filtresi(gurultulu, esik)
        print(f"  eşik {esik:5.2f}: {tutulan:4d} harmonik, RMSE {rmse(filtreli, temiz):.4f}")


if __name__ == "__main__":
    calistir()
