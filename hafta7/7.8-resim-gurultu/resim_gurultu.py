"""Hafta 7 / 7.8: Resme Gauss ve tuz-biber (salt and pepper) gürültü ekleme, harmonik matris karşılaştırması.

gauss_gurultu_ekle: her piksele ortalaması 0, standart sapması sigma olan normal dağılımlı sayı eklenir.
tuz_biber_ekle: piksellerin oran kadarı rastgele seçilir, yarısı 0 (biber), yarısı 255 (tuz) yapılır.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

KLASOR = Path(__file__).resolve().parent
CIKTI = KLASOR / "cikti"
for klasor in ["7.1-sinus-vektor", "7.2-gurultu-ekleme", "7.4-dft-harmonik", "7.7-2b-dft"]:
    sys.path.insert(0, str(KLASOR.parent / klasor))

from dft2 import dft2, merkeze_al, resim_oku, spektrum_resmi  # noqa: E402

SIGMA = 25
ORAN = 0.05


def gauss_gurultu_ekle(resim, sigma, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    gurultulu = resim.astype(float) + rng.normal(0, sigma, resim.shape)
    return np.clip(gurultulu, 0, 255).astype(np.uint8)


def tuz_biber_ekle(resim, oran, rng=None):
    if not 0 <= oran <= 1:
        raise ValueError(f"Oran [0, 1] aralığında olmalı, gelen: {oran}")
    rng = np.random.default_rng() if rng is None else rng
    sonuc = resim.copy()
    r = rng.random(resim.shape)
    sonuc[r < oran / 2] = 0
    sonuc[(r >= oran / 2) & (r < oran)] = 255
    return sonuc


def gurultulu_resimler():
    """7.9 ve 7.10 aynı gürültülü resimleri kullansın diye sabit tohumla üretilir."""
    resim = resim_oku()
    return {
        "orijinal": resim,
        "gauss": gauss_gurultu_ekle(resim, SIGMA, np.random.default_rng(11)),
        "tuz_biber": tuz_biber_ekle(resim, ORAN, np.random.default_rng(12)),
    }


def yuksek_frekans_payi(harmonik, yaricap_orani=0.25):
    """DC hariç toplam enerjinin, merkezden yarıçap_oranı * N'den uzak (yüksek frekans) kısmındaki yüzdesi."""
    m, n = harmonik.shape
    enerji = np.abs(merkeze_al(harmonik)) ** 2
    enerji[m // 2, n // 2] = 0
    v, u = np.ogrid[:m, :n]
    uzaklik = np.hypot(v - m // 2, u - n // 2)
    return 100 * enerji[uzaklik > yaricap_orani * min(m, n)].sum() / enerji.sum()


def kaydet(ad, goruntu):
    if not cv2.imwrite(str(CIKTI / ad), goruntu):
        raise OSError(f"Kaydedilemedi: {ad}")


def calistir():
    CIKTI.mkdir(exist_ok=True)
    resimler = gurultulu_resimler()
    harmonikler = {ad: dft2(r) for ad, r in resimler.items()}
    bosluk = np.full((resimler["orijinal"].shape[0], 8), 255, np.uint8)

    ust, alt = [], []
    for ad, resim in resimler.items():
        spektrum = spektrum_resmi(harmonikler[ad])
        kaydet(f"{ad}.png", resim)
        kaydet(f"{ad}_harmonik.png", spektrum)
        ust += [resim, bosluk]
        alt += [spektrum, bosluk]
        print(f"{ad:9s}: yüksek frekans enerji payı %{yuksek_frekans_payi(harmonikler[ad]):.2f}, "
              f"ortalama log(1+|H|) {np.log1p(np.abs(harmonikler[ad])).mean():.3f}")
    yatay_bosluk = np.full((8, sum(p.shape[1] for p in ust[:-1])), 255, np.uint8)
    kaydet("karsilastirma.png", np.vstack([np.hstack(ust[:-1]), yatay_bosluk, np.hstack(alt[:-1])]))

    # Gürültünün spektrumu: gürültülü harmonik matris - orijinal harmonik matris
    for ad in ["gauss", "tuz_biber"]:
        fark = np.abs(harmonikler[ad] - harmonikler["orijinal"])
        ic = fark[1:, 1:]
        print(f"{ad:9s} gürültü spektrumu |H_g - H_o|: ortalama {ic.mean():.0f}, std/ortalama {ic.std() / ic.mean():.2f}")


if __name__ == "__main__":
    calistir()
