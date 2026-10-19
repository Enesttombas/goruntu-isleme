"""Hafta 5 / 5.3: RGB -> gri dönüşümü, OpenCV (cv2.cvtColor) ile."""

from pathlib import Path

import cv2

KLASOR = Path(__file__).resolve().parent
GIRDI = KLASOR.parent / "girdi" / "astronot.png"
CIKTI = KLASOR / "cikti"


def calistir() -> None:
    renkli = cv2.imread(str(GIRDI), cv2.IMREAD_COLOR)
    if renkli is None:
        raise FileNotFoundError(f"Renkli fotoğraf okunamadı: {GIRDI}")
    CIKTI.mkdir(exist_ok=True)
    gri = cv2.cvtColor(renkli, cv2.COLOR_BGR2GRAY)
    if not cv2.imwrite(str(CIKTI / "gri_opencv.png"), gri):
        raise OSError("Gri görüntü kaydedilemedi")
    print(f"Gri görüntü kaydedildi: {gri.shape}, {gri.dtype}")


if __name__ == "__main__":
    calistir()
