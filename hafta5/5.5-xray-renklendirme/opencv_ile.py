"""Hafta 5 / 5.5: X-ray görüntülerini gri yapıp renklendirme, OpenCV (cvtColor + applyColorMap) ile."""

import sys
from pathlib import Path

import cv2

KLASOR = Path(__file__).resolve().parent
sys.path.insert(0, str(KLASOR.parent))
from xray_palet import palet_olustur  # noqa: E402

GIRDI = KLASOR.parent / "girdi" / "ThreatsRGB"
CIKTI = KLASOR / "cikti"


def calistir() -> None:
    CIKTI.mkdir(exist_ok=True)
    palet = palet_olustur().reshape(256, 1, 3)  # applyColorMap için 256x1 üç kanallı tablo
    dosyalar = sorted(GIRDI.glob("*.jpg"))
    if not dosyalar:
        raise FileNotFoundError(f"Görüntü yok: {GIRDI}")
    for dosya in dosyalar:
        renkli = cv2.imread(str(dosya), cv2.IMREAD_COLOR)
        if renkli is None:
            raise FileNotFoundError(f"Okunamadı: {dosya}")
        gri = cv2.cvtColor(renkli, cv2.COLOR_BGR2GRAY)
        boyali = cv2.applyColorMap(gri, palet)
        cv2.imwrite(str(CIKTI / f"{dosya.stem}_gri_opencv.png"), gri)
        cv2.imwrite(str(CIKTI / f"{dosya.stem}_renkli_opencv.png"), boyali)
        print(f"{dosya.name}: gri ve renkli kaydedildi")


if __name__ == "__main__":
    calistir()
