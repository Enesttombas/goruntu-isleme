"""Hafta 7 / 7.5: f=50 Hz A=12 ve f=30 Hz A=5 gürültülü iki sinyal, ayrı ayrı ve toplamın DFT analizi.

N = 1024 örnek, fs = 1024 Hz (1 saniye). Frekans çözünürlüğü fs/N = 1 Hz, yani 30 ve 50 Hz
tam bir harmoniğe denk gelir ve sızıntı (leakage) olmaz.
DFT doğrusal olduğu için DFT(s1 + s2) = DFT(s1) + DFT(s2) beklenir.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

KLASOR = Path(__file__).resolve().parent
CIKTI = KLASOR / "cikti"
for klasor in ["7.1-sinus-vektor", "7.2-gurultu-ekleme", "7.4-dft-harmonik"]:
    sys.path.insert(0, str(KLASOR.parent / klasor))

from dft import genlik_spektrumu, harmonik_vektoru  # noqa: E402
from gurultu import gurultu_ekle  # noqa: E402
from sinus import sinus_doldur  # noqa: E402

N = 1024
FS = 1024
ORAN = 0.2


def sinyalleri_olustur():
    """7.6'da da aynı sinyaller kullanılıyor. Temiz ve gürültülü halleri döner."""
    s1 = sinus_doldur(np.zeros(N), genlik=12, faz=0, frekans=50, ornekleme_frekansi=FS)
    s2 = sinus_doldur(np.zeros(N), genlik=5, faz=0, frekans=30, ornekleme_frekansi=FS)
    s1_g = gurultu_ekle(s1, ORAN, np.random.default_rng(1))
    s2_g = gurultu_ekle(s2, ORAN, np.random.default_rng(2))
    return {
        "s1 (50 Hz, A=12)": (s1, s1_g),
        "s2 (30 Hz, A=5)": (s2, s2_g),
        "toplam s1 + s2": (s1 + s2, s1_g + s2_g),
    }


def calistir():
    CIKTI.mkdir(exist_ok=True)
    sinyaller = sinyalleri_olustur()
    t = np.arange(N) / FS
    goster = t < 0.2  # zaman grafiğinde ilk 0.2 s

    sekil, eksenler = plt.subplots(3, 2, figsize=(14, 10), gridspec_kw={"width_ratios": [1, 1.2]})
    harmonikler = {}
    for satir, (ad, (_, gurultulu)) in enumerate(sinyaller.items()):
        harmonik = harmonik_vektoru(gurultulu)
        harmonikler[ad] = harmonik
        frekans, genlik = genlik_spektrumu(harmonik, FS)

        eksenler[satir, 0].plot(t[goster], gurultulu[goster], color="tab:orange", lw=1)
        eksenler[satir, 0].set_title(f"{ad}, gürültülü (ilk 0.2 s)")
        eksenler[satir, 0].set_ylabel("genlik")
        eksenler[satir, 1].stem(frekans[:150], genlik[:150], linefmt="tab:purple", markerfmt=" ", basefmt="k-")
        eksenler[satir, 1].set_title(f"{ad}, harmonik vektör (0-150 Hz)")
        for f in (30, 50):
            eksenler[satir, 1].annotate(f"{genlik[f]:.2f}", (f, genlik[f]), textcoords="offset points", xytext=(4, 2), fontsize=9)
        print(f"{ad}: 30 Hz genlik {genlik[30]:.3f}, 50 Hz genlik {genlik[50]:.3f}, "
              f"diğer harmonikler ort {np.delete(genlik, [30, 50]).mean():.4f}, en büyüğü {np.delete(genlik, [30, 50]).max():.4f}")
    for eksen in eksenler.ravel():
        eksen.grid(alpha=0.3)
    eksenler[2, 0].set_xlabel("zaman (s)")
    eksenler[2, 1].set_xlabel("frekans (Hz)")
    sekil.tight_layout()
    sekil.savefig(CIKTI / "iki_sinyal_ve_toplam.png", dpi=120)
    plt.close(sekil)

    adlar = list(harmonikler)
    fark = np.abs(harmonikler[adlar[2]] - (harmonikler[adlar[0]] + harmonikler[adlar[1]])).max()
    print(f"Doğrusallık: max|DFT(s1+s2) - (DFT(s1) + DFT(s2))| = {fark:.2e}")


if __name__ == "__main__":
    calistir()
