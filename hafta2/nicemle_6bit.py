"""Ödev 1: Gri görüntüyü 6-bit parlaklık seviyesine nicemle.

Kullanım: python nicemle_6bit.py [girdi] [cikti]
Varsayılan: dama.jpg ve cikti klasörü.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

KLASOR = Path(__file__).resolve().parent


def nicemle_6bit(gri):
    """Alt iki biti atıp sonucu 0-255 aralığında göster."""
    return ((gri >> 2) << 2).astype(np.uint8)


def fark_gorseli(gri, nicemli):
    fark = cv2.absdiff(gri, nicemli)
    en_buyuk = int(fark.max())
    return fark if en_buyuk == 0 else (fark.astype(np.float32) * (255 / en_buyuk)).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description="8-bit gri görüntüyü 6-bit nicemle.")
    ap.add_argument("girdi", nargs="?", type=Path, default=KLASOR / "dama.jpg")
    ap.add_argument("cikti", nargs="?", type=Path, default=KLASOR / "cikti")
    args = ap.parse_args()
    gri = cv2.imread(str(args.girdi), cv2.IMREAD_GRAYSCALE)
    if gri is None:
        raise SystemExit(f"Görüntü okunamadı: {args.girdi}")
    nicemli = nicemle_6bit(gri)
    args.cikti.mkdir(parents=True, exist_ok=True)
    for ad, goruntu in (("gri.png", gri), ("6bit.png", nicemli), ("fark.png", fark_gorseli(gri, nicemli))):
        if not cv2.imwrite(str(args.cikti / ad), goruntu):
            raise SystemExit(f"Görüntü yazılamadı: {args.cikti / ad}")
    print(f"Benzersiz seviye: önce {np.unique(gri).size}, sonra {np.unique(nicemli).size}")


if __name__ == "__main__":
    main()
