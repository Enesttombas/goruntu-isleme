"""Ödev 2 (hazır fonksiyonlu): cv2.calcHist, np.cumsum ve cv2.equalizeHist.

Kullanım: python histogram_esitleme_hazir.py [girdi ...]
Varsayılan: ../girdi/dag.jpg ve sis.jpg; çıktılar cikti/ klasörüne yazılır.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

KLASOR = Path(__file__).resolve().parent
GIRDI = KLASOR.parent / "girdi"


def main():
    ap = argparse.ArgumentParser(description="Histogram eşitleme, OpenCV ile.")
    ap.add_argument("girdiler", nargs="*", type=Path, default=[GIRDI / "dag.jpg", GIRDI / "sis.jpg"])
    args = ap.parse_args()
    cikti = KLASOR / "cikti"
    cikti.mkdir(exist_ok=True)
    for yol in args.girdiler:
        gri = cv2.imread(str(yol), cv2.IMREAD_GRAYSCALE)
        if gri is None:
            raise SystemExit(f"Görüntü okunamadı: {yol}")
        hist = cv2.calcHist([gri], [0], None, [256], [0, 256]).ravel()
        cdf = np.cumsum(hist)
        sonuc = cv2.equalizeHist(gri)
        hedef = cikti / f"{yol.stem}_esitlenmis.png"
        if not cv2.imwrite(str(hedef), sonuc):
            raise SystemExit(f"Görüntü yazılamadı: {hedef}")
        print(f"{yol.name}: hist toplamı={int(hist.sum())} CDF son={int(cdf[-1])}, "
              f"std {gri.std():.2f} -> {sonuc.std():.2f}")


if __name__ == "__main__":
    main()
