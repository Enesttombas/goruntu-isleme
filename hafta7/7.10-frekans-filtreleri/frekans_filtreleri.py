"""Hafta 7 / 7.10: Frekans uzayında alçak, yüksek, bant geçiren ve bant durduran filtreler.

Hepsi Gauss biçimli maske (ideal filtredeki keskin kesimin yarattığı halka/ringing olmasın diye).
D(u, v) = harmoniğin DC'ye uzaklığı
    alçak geçiren   AG = exp(-D^2 / (2 * D0^2))
    yüksek geçiren  YG = 1 - AG
    bant geçiren    BG = exp(-((D^2 - C0^2) / (D * W))^2)     C0: bant merkezi, W: bant genişliği
    bant durduran   BD = 1 - BG
Uygulama: harmonik matris maskeyle çarpılır, ters DFT alınır. Görüntülerken ölçekleme yapılmaz, 0-255'e kırpılır.
Yüksek ve bant geçiren DC'yi (ortalama parlaklığı) siler, sonuç 0 etrafında artı/eksi olur; bu ikisi
kenarlar ve gürültü görünsün diye +128 kaydırılarak gösterilir. Gerçek ortalama başlıkta yazar.
"""

import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

KLASOR = Path(__file__).resolve().parent
CIKTI = KLASOR / "cikti"
for klasor in ["7.1-sinus-vektor", "7.2-gurultu-ekleme", "7.4-dft-harmonik", "7.7-2b-dft", "7.8-resim-gurultu", "7.9-ters-dft-zncc"]:
    sys.path.insert(0, str(KLASOR.parent / klasor))

from dft2 import dft2, idft2, merkeze_al  # noqa: E402
from resim_gurultu import gurultulu_resimler  # noqa: E402
from ters_dft import uzaklik_matrisi  # noqa: E402

D0 = 40
C0, W = 60, 40


def alcak_geciren(boyut, d0):
    return np.exp(-uzaklik_matrisi(*boyut) ** 2 / (2 * d0 ** 2))


def yuksek_geciren(boyut, d0):
    return 1 - alcak_geciren(boyut, d0)


def bant_geciren(boyut, c0, w):
    d = uzaklik_matrisi(*boyut)
    d[d == 0] = 1e-9  # DC'de bölme hatası olmasın, DC zaten bandın dışında
    return np.exp(-(((d ** 2 - c0 ** 2) / (d * w)) ** 2))


def bant_durduran(boyut, c0, w):
    return 1 - bant_geciren(boyut, c0, w)


def filtre_uygula(resim, maske):
    """Kırpılmamış (float) sonuç döner, ortalaması parlaklığı gösterir."""
    return np.real(idft2(dft2(resim) * maske))


def kenar_gucu(resim):
    gx = cv2.Sobel(resim.astype(float), cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(resim.astype(float), cv2.CV_64F, 0, 1, ksize=3)
    return float(np.hypot(gx, gy).mean())


def calistir():
    CIKTI.mkdir(exist_ok=True)
    resimler = gurultulu_resimler()
    boyut = resimler["orijinal"].shape
    maskeler = {
        f"alçak geçiren D0={D0}": alcak_geciren(boyut, D0),
        f"yüksek geçiren D0={D0}": yuksek_geciren(boyut, D0),
        f"bant geçiren C0={C0} W={W}": bant_geciren(boyut, C0, W),
        f"bant durduran C0={C0} W={W}": bant_durduran(boyut, C0, W),
    }

    sekil, eksenler = plt.subplots(1, 4, figsize=(16, 4.4))
    for eksen, (ad, maske) in zip(eksenler, maskeler.items()):
        eksen.imshow(merkeze_al(maske), cmap="gray", vmin=0, vmax=1)
        eksen.set_title(ad, fontsize=10)
        eksen.axis("off")
    sekil.tight_layout()
    sekil.savefig(CIKTI / "maskeler.png", dpi=110)
    plt.close(sekil)

    # Her filtrenin temiz resme uygulanmış hali, gürültü artığını ölçmek için referans
    temiz_sonuclar = {ad: filtre_uygula(resimler["orijinal"], m) for ad, m in maskeler.items()}
    kenar_orijinal = kenar_gucu(resimler["orijinal"])

    sekil, eksenler = plt.subplots(3, 5, figsize=(20, 13.2), layout="constrained")
    print(f"{'girdi':10s} {'filtre':28s} {'ort. parlaklık':>14s} {'kenar gücü %':>12s} {'gürültü RMSE':>12s}")
    for satir, (girdi_ad, resim) in enumerate(resimler.items()):
        eksenler[satir, 0].imshow(resim, cmap="gray", vmin=0, vmax=255)
        eksenler[satir, 0].set_title(f"{girdi_ad} (ort. {resim.mean():.0f})", fontsize=11)
        print(f"{girdi_ad:10s} {'filtresiz':28s} {round(resim.mean(), 1) + 0.0:14.1f} {100 * kenar_gucu(resim) / kenar_orijinal:12.0f} "
              f"{np.sqrt(np.mean((resim.astype(float) - resimler['orijinal']) ** 2)):12.1f}")
        for sutun, (filtre_ad, maske) in enumerate(maskeler.items(), start=1):
            sonuc = filtre_uygula(resim, maske)
            gurultu_artigi = np.sqrt(np.mean((sonuc - temiz_sonuclar[filtre_ad]) ** 2))
            kaydir = 128 if filtre_ad.startswith(("yüksek", "bant geçiren")) else 0
            eksenler[satir, sutun].imshow(np.clip(sonuc + kaydir, 0, 255), cmap="gray", vmin=0, vmax=255)
            ek = " (gösterim +128)" if kaydir else ""
            eksenler[satir, sutun].set_title(f"{filtre_ad}\nort. {round(sonuc.mean()) + 0:.0f}{ek}, gürültü RMSE {gurultu_artigi:.1f}", fontsize=10)
            print(f"{girdi_ad:10s} {filtre_ad:28s} {round(sonuc.mean(), 1) + 0.0:14.1f} {100 * kenar_gucu(sonuc) / kenar_orijinal:12.0f} {gurultu_artigi:12.1f}")
    for eksen in eksenler.ravel():
        eksen.axis("off")
    sekil.savefig(CIKTI / "filtre_sonuclari.png", dpi=100)
    plt.close(sekil)


if __name__ == "__main__":
    calistir()
