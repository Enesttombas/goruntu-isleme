"""Hafta 5 / 5.1: RGB kanal histogramları ve gri tonlamaya dönüşüm."""

from pathlib import Path

import cv2
import numpy as np


KLASOR = Path(__file__).resolve().parent
GIRDI = KLASOR / "girdi" / "astronot.png"
CIKTI = KLASOR / "cikti"


def histogram_ciz(histogram: np.ndarray, baslik: str, renk: tuple[int, int, int]) -> np.ndarray:
    """256 yoğunluk değerinin piksel sayılarını etiketli bir grafikte göster."""
    genislik, yukseklik = 760, 430
    sol, ust, sag, alt = 65, 55, 25, 55
    grafik = np.full((yukseklik, genislik, 3), 255, dtype=np.uint8)
    x0, y0 = sol, yukseklik - alt
    x1, y1 = genislik - sag, ust
    cv2.line(grafik, (x0, y0), (x1, y0), (40, 40, 40), 1)
    cv2.line(grafik, (x0, y0), (x0, y1), (40, 40, 40), 1)

    en_yuksek = float(histogram.max())
    for yogunluk in range(256):
        x = x0 + round(yogunluk * (x1 - x0) / 255)
        y = y0 - round(float(histogram[yogunluk]) * (y0 - y1) / en_yuksek)
        cv2.line(grafik, (x, y0), (x, y), renk, 2)

    cv2.putText(grafik, baslik, (sol, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
    cv2.putText(grafik, "Piksel sayisi", (sol + 5, ust + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (70, 70, 70), 1)
    cv2.putText(grafik, "Yogunluk", (genislik // 2 - 40, yukseklik - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (70, 70, 70), 1)
    for deger in (0, 64, 128, 192, 255):
        x = x0 + round(deger * (x1 - x0) / 255)
        cv2.putText(grafik, str(deger), (x - 12, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (70, 70, 70), 1)
    cv2.putText(grafik, str(round(en_yuksek)), (5, y1 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (70, 70, 70), 1)
    return grafik


def calistir() -> None:
    renkli = cv2.imread(str(GIRDI), cv2.IMREAD_COLOR)
    if renkli is None:
        raise FileNotFoundError(f"Renkli fotoğraf okunamadı: {GIRDI}")
    CIKTI.mkdir(exist_ok=True)

    # OpenCV görüntüyü BGR sırasıyla okur. Dosya adları ve grafik renkleri
    # doğru eşleşsin diye kanalları açıkça B, G, R indisleriyle işleriz.
    kanallar = (
        ("kirmizi", 2, (0, 0, 210)),
        ("yesil", 1, (0, 160, 0)),
        ("mavi", 0, (210, 0, 0)),
    )
    for ad, indeks, renk in kanallar:
        histogram = cv2.calcHist([renkli], [indeks], None, [256], [0, 256]).ravel()
        grafik = histogram_ciz(histogram, f"{ad.capitalize()} kanal histogrami", renk)
        if not cv2.imwrite(str(CIKTI / f"histogram_{ad}.png"), grafik):
            raise OSError(f"{ad} histogramı kaydedilemedi")

    gri = cv2.cvtColor(renkli, cv2.COLOR_BGR2GRAY)
    if not cv2.imwrite(str(CIKTI / "astronot_gri.png"), gri):
        raise OSError("Gri tonlamalı görüntü kaydedilemedi")
    print(f"3 kanal histogramı ve gri görüntü kaydedildi: {CIKTI}")


if __name__ == "__main__":
    calistir()
