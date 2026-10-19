"""5.1 ve 5.2 ortak yardımcı: 256 değerlik histogramı etiketli grafiğe çizer."""

import cv2
import numpy as np


def histogram_ciz(histogram: np.ndarray, baslik: str, renk: tuple[int, int, int]) -> np.ndarray:
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
