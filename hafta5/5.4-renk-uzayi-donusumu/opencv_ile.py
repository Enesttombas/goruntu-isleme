"""Hafta 5 / 5.4: renk uzayı dönüşümleri, OpenCV (cv2.cvtColor) ile."""

from pathlib import Path

import cv2

KLASOR = Path(__file__).resolve().parent
GIRDI = KLASOR.parent / "girdi" / "astronot.png"
CIKTI = KLASOR / "cikti"

DONUSUMLER = {
    "gri": cv2.COLOR_BGR2GRAY,
    "ycrcb": cv2.COLOR_BGR2YCrCb,
    "yuv": cv2.COLOR_BGR2YUV,
    "xyz": cv2.COLOR_BGR2XYZ,
    "hsv": cv2.COLOR_BGR2HSV,
    "hls": cv2.COLOR_BGR2HLS,
}


def calistir() -> None:
    renkli = cv2.imread(str(GIRDI), cv2.IMREAD_COLOR)
    if renkli is None:
        raise FileNotFoundError(f"Renkli fotoğraf okunamadı: {GIRDI}")
    CIKTI.mkdir(exist_ok=True)
    for ad, kod in DONUSUMLER.items():
        if not cv2.imwrite(str(CIKTI / f"{ad}_opencv.png"), cv2.cvtColor(renkli, kod)):
            raise OSError(f"{ad} kaydedilemedi")
        print(f"{ad}: kaydedildi")


if __name__ == "__main__":
    calistir()
