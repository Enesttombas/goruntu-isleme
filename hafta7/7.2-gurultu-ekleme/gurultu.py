"""Hafta 7 / 7.2: Vektöre, genliğine ve gürültü oranına göre rastgele gürültü ekleme.

genlik = vektördeki en büyük mutlak değer
Her elemana [-oran*genlik, +oran*genlik] aralığından düzgün dağılımlı rastgele bir sayı eklenir.
oran = 0 ise vektör aynen kalır, oran = 1 ise gürültü sinyalin kendisi kadar büyük olur.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

KLASOR = Path(__file__).resolve().parent
CIKTI = KLASOR / "cikti"
sys.path.insert(0, str(KLASOR.parent / "7.1-sinus-vektor"))

from sinus import sinus_doldur  # noqa: E402

TOHUM = 7  # aynı gürültü her çalıştırmada tekrar çıksın diye


def gurultu_ekle(vektor, oran, rng=None):
    """Girdi vektörü değişmez, gürültülü yeni bir vektör döner."""
    if not 0 <= oran <= 1:
        raise ValueError(f"Gürültü oranı [0, 1] aralığında olmalı, gelen: {oran}")
    rng = np.random.default_rng() if rng is None else rng
    vektor = np.asarray(vektor, dtype=float)
    genlik = np.max(np.abs(vektor))
    gurultu = rng.uniform(-oran * genlik, oran * genlik, size=vektor.shape)
    return vektor + gurultu


def temiz_sinyal():
    """7.3 ve 7.4'te de kullanılan sinyal: N=512, A=1, f=5 Hz, faz=0."""
    return sinus_doldur(np.zeros(512), genlik=1, faz=0, frekans=5)


def calistir():
    CIKTI.mkdir(exist_ok=True)
    temiz = temiz_sinyal()
    t = np.arange(len(temiz)) / len(temiz)
    oranlar = [0.1, 0.3, 0.6]

    sekil, eksenler = plt.subplots(len(oranlar), 1, figsize=(11, 9), sharex=True)
    for eksen, oran in zip(eksenler, oranlar):
        gurultulu = gurultu_ekle(temiz, oran, np.random.default_rng(TOHUM))
        fark = gurultulu - temiz
        eksen.plot(t, gurultulu, color="tab:orange", lw=0.8, label="gürültülü")
        eksen.plot(t, temiz, color="tab:blue", lw=1.5, label="orijinal")
        eksen.set_title(f"oran = {oran}  (gürültü aralığı ±{oran:.1f}, ölçülen en büyük |gürültü| = {np.abs(fark).max():.3f})")
        eksen.set_ylabel("genlik")
        eksen.grid(alpha=0.3)
        eksen.legend(loc="upper right")
        print(f"oran={oran}: en büyük |gürültü|={np.abs(fark).max():.4f}, gürültü RMS={np.sqrt(np.mean(fark ** 2)):.4f}")
    eksenler[-1].set_xlabel("zaman (s)")
    sekil.tight_layout()
    sekil.savefig(CIKTI / "gurultulu_sinyal.png", dpi=120)
    plt.close(sekil)


if __name__ == "__main__":
    calistir()
