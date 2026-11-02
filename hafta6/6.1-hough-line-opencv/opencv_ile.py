"""Hafta 6 / 6.1: Canny -> median filtre -> iki farklı OpenCV Hough Line fonksiyonu."""

from pathlib import Path

import cv2
import numpy as np

KLASOR = Path(__file__).resolve().parent
GIRDI = KLASOR.parent / "girdi" / "sudoku.png"
CIKTI = KLASOR / "cikti"

KIRMIZI = (0, 0, 255)  # cv2.HoughLines
YESIL = (0, 200, 0)    # cv2.HoughLinesP


def kaydet(ad: str, goruntu: np.ndarray) -> None:
    if not cv2.imwrite(str(CIKTI / ad), goruntu):
        raise OSError(f"Kaydedilemedi: {ad}")


def hough_lines_ciz(resim: np.ndarray, kenar: np.ndarray) -> int:
    """Standart Hough: her doğru (rho, theta) olarak döner, resmin bir ucundan öbürüne çizilir."""
    dogrular = cv2.HoughLines(kenar, rho=1, theta=np.pi / 180, threshold=100)
    if dogrular is None:
        return 0
    uzunluk = 2 * max(resim.shape)
    for rho, theta in dogrular[:, 0]:
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x0, y0 = rho * cos_t, rho * sin_t
        p1 = (int(x0 - uzunluk * sin_t), int(y0 + uzunluk * cos_t))
        p2 = (int(x0 + uzunluk * sin_t), int(y0 - uzunluk * cos_t))
        cv2.line(resim, p1, p2, KIRMIZI, 2)
    return len(dogrular)


def hough_lines_p_ciz(resim: np.ndarray, kenar: np.ndarray) -> int:
    """Olasılıksal Hough: doğru parçasının iki uç noktasını döner."""
    parcalar = cv2.HoughLinesP(kenar, rho=1, theta=np.pi / 180, threshold=50,
                               minLineLength=60, maxLineGap=15)
    if parcalar is None:
        return 0
    for x1, y1, x2, y2 in parcalar[:, 0]:
        cv2.line(resim, (int(x1), int(y1)), (int(x2), int(y2)), YESIL, 2)
    return len(parcalar)


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

    cizim_lines = renkli.copy()
    cizim_lines_p = renkli.copy()
    n_lines = hough_lines_ciz(cizim_lines, kenar_median)
    n_lines_p = hough_lines_p_ciz(cizim_lines_p, kenar_median)
    kaydet("dogrular_houghlines.png", cizim_lines)
    kaydet("dogrular_houghlinesp.png", cizim_lines_p)

    print(f"Canny kenar pikseli: {(kenar > 0).sum()}, median sonrası: {(kenar_median > 0).sum()}")
    print(f"cv2.HoughLines (kırmızı): {n_lines} doğru")
    print(f"cv2.HoughLinesP (yeşil): {n_lines_p} doğru parçası")


if __name__ == "__main__":
    calistir()
