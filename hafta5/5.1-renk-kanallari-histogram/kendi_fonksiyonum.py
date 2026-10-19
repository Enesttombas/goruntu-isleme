"""Hafta 5 / 5.1: kanal histogramları, KENDİ yazdığımız calchist ile (OpenCV yok)."""

import sys
from pathlib import Path

import cv2
import numpy as np

KLASOR = Path(__file__).resolve().parent
sys.path.insert(0, str(KLASOR.parent))
from histogram_grafik import histogram_ciz  # noqa: E402

GIRDI = KLASOR.parent / "girdi" / "astronot.png"
CIKTI = KLASOR / "cikti"


def kendi_calchist(kanal: np.ndarray) -> np.ndarray:
    """Tek kanallı 8-bit görüntüde her yoğunluğun kaç piksel olduğunu say."""
    histogram = np.zeros(256, dtype=np.int64)
    for satir in kanal:
        for deger in satir:
            histogram[deger] += 1
    return histogram


def calistir() -> None:
    renkli = cv2.imread(str(GIRDI), cv2.IMREAD_COLOR)
    if renkli is None:
        raise FileNotFoundError(f"Renkli fotoğraf okunamadı: {GIRDI}")
    CIKTI.mkdir(exist_ok=True)

    # OpenCV BGR sırasıyla okur: mavi 0, yeşil 1, kırmızı 2.
    kanallar = (
        ("kirmizi", 2, (0, 0, 210)),
        ("yesil", 1, (0, 160, 0)),
        ("mavi", 0, (210, 0, 0)),
    )
    for ad, indeks, renk in kanallar:
        kanal = renkli[:, :, indeks]
        histogram = kendi_calchist(kanal)
        print(f"{ad}: toplam piksel = {int(histogram.sum())}")
        grafik = histogram_ciz(histogram, f"{ad.capitalize()} kanal histogrami (kendi calcHist)", renk)
        if not cv2.imwrite(str(CIKTI / f"histogram_{ad}_kendi.png"), grafik):
            raise OSError(f"{ad} histogramı kaydedilemedi")


if __name__ == "__main__":
    calistir()
