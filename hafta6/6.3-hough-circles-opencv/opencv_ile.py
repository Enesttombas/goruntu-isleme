"""Hafta 6 / 6.3: Canny -> median filtre -> OpenCV HoughCircles."""

from pathlib import Path

import cv2
import numpy as np

KLASOR = Path(__file__).resolve().parent
GIRDI = KLASOR.parent / "girdi" / "smarties.png"
CIKTI = KLASOR / "cikti"

CEMBER_RENGI = (255, 255, 0)  # camgöbeği
MERKEZ_RENGI = (0, 0, 255)


def kaydet(ad: str, goruntu: np.ndarray) -> None:
    if not cv2.imwrite(str(CIKTI / ad), goruntu):
        raise OSError(f"Kaydedilemedi: {ad}")


def calistir() -> None:
    renkli = cv2.imread(str(GIRDI), cv2.IMREAD_COLOR)
    if renkli is None:
        raise FileNotFoundError(f"Resim okunamadı: {GIRDI}")
    CIKTI.mkdir(exist_ok=True)

    gri = cv2.cvtColor(renkli, cv2.COLOR_BGR2GRAY)
    kenar = cv2.Canny(gri, 50, 150)
    kenar_median = cv2.medianBlur(kenar, 3)
    kaydet("kenar_canny.png", kenar)
    kaydet("kenar_median.png", kenar_median)

    # HoughCircles girdiyi kendi içinde yine Canny'den geçirir; ikili kenar resminde
    # param1 eşiği her kenarı korur. param2 = merkez oylamasında gereken en az oy.
    # dp=2: oylama matrisi yarı çözünürlükte; median kenarları seyrelttiği için oylar toplanır.
    cemberler = cv2.HoughCircles(kenar_median, cv2.HOUGH_GRADIENT, dp=2, minDist=30,
                                 param1=100, param2=20, minRadius=20, maxRadius=40)
    cizim = renkli.copy()
    sayi = 0
    if cemberler is not None:
        for x, y, r in np.round(cemberler[0]).astype(int):
            cv2.circle(cizim, (x, y), r, CEMBER_RENGI, 2)
            cv2.circle(cizim, (x, y), 2, MERKEZ_RENGI, 3)
        sayi = cemberler.shape[1]
    kaydet("cemberler_houghcircles.png", cizim)

    print(f"Canny kenar pikseli: {(kenar > 0).sum()}, median sonrası: {(kenar_median > 0).sum()}")
    print(f"cv2.HoughCircles: {sayi} çember")


if __name__ == "__main__":
    calistir()
