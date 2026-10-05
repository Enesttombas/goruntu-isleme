"""Ödev 1 (hazır fonksiyonlu): cv2.minMaxLoc + cv2.normalize(NORM_MINMAX) ile [0, 255] ölçekleme.

Kullanım: python kontrast_germe_hazir.py [girdi ...]
Varsayılan: ../girdi/dag.jpg ve sis.jpg; çıktılar cikti/ klasörüne yazılır.
"""
import argparse
from pathlib import Path

import cv2

KLASOR = Path(__file__).resolve().parent
GIRDI = KLASOR.parent / "girdi"


def main():
    ap = argparse.ArgumentParser(description="Doğrusal kontrast germe, OpenCV ile.")
    ap.add_argument("girdiler", nargs="*", type=Path, default=[GIRDI / "dag.jpg", GIRDI / "sis.jpg"])
    args = ap.parse_args()
    cikti = KLASOR / "cikti"
    cikti.mkdir(exist_ok=True)
    for yol in args.girdiler:
        gri = cv2.imread(str(yol), cv2.IMREAD_GRAYSCALE)
        if gri is None:
            raise SystemExit(f"Görüntü okunamadı: {yol}")
        en_kucuk, en_buyuk, _, _ = cv2.minMaxLoc(gri)
        sonuc = cv2.normalize(gri, None, 0, 255, cv2.NORM_MINMAX)
        hedef = cikti / f"{yol.stem}_sonra.png"
        if not cv2.imwrite(str(hedef), sonuc):
            raise SystemExit(f"Görüntü yazılamadı: {hedef}")
        print(f"{yol.name}: min={int(en_kucuk)} max={int(en_buyuk)} -> "
              f"min={int(sonuc.min())} max={int(sonuc.max())}")


if __name__ == "__main__":
    main()
