"""Ödev 1: düşük kontrastlı tek kanallı görüntüyü [0, 255] aralığına doğrusal ölçekle.

y = (x - min) * 255 / (max - min). cv2.normalize / equalizeHist gibi hazır ölçekleme yok.
Kullanım: python kontrast_germe.py [girdi ...]
Varsayılan: dag.jpg ve sis.jpg; çıktılar cikti/ klasörüne yazılır.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

KLASOR = Path(__file__).resolve().parent


def dogrusal_olcekle(gri):
    en_kucuk = int(gri.min())
    en_buyuk = int(gri.max())
    if en_buyuk == en_kucuk:
        return np.zeros_like(gri), en_kucuk, en_buyuk
    x = gri.astype(np.float32)
    y = (x - en_kucuk) * 255.0 / (en_buyuk - en_kucuk)
    return np.round(y).astype(np.uint8), en_kucuk, en_buyuk


def yan_yana(once, sonra):
    hucreler = []
    for goruntu, etiket in ((once, "Once"), (sonra, "Sonra")):
        hucre = cv2.copyMakeBorder(goruntu, 32, 0, 0, 0, cv2.BORDER_CONSTANT, value=0)
        cv2.putText(hucre, etiket, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1, cv2.LINE_AA)
        hucreler.append(hucre)
    return cv2.hconcat(hucreler)


def main():
    ap = argparse.ArgumentParser(description="Doğrusal kontrast germe (min-max).")
    ap.add_argument("girdiler", nargs="*", type=Path, default=[KLASOR / "dag.jpg", KLASOR / "sis.jpg"])
    args = ap.parse_args()
    cikti = KLASOR / "cikti"
    cikti.mkdir(exist_ok=True)
    for yol in args.girdiler:
        gri = cv2.imread(str(yol), cv2.IMREAD_GRAYSCALE)
        if gri is None:
            raise SystemExit(f"Görüntü okunamadı: {yol}")
        sonuc, en_kucuk, en_buyuk = dogrusal_olcekle(gri)
        ad = yol.stem
        for dosya, goruntu in ((f"{ad}_once.png", gri), (f"{ad}_sonra.png", sonuc),
                               (f"{ad}_karsilastirma.png", yan_yana(gri, sonuc))):
            if not cv2.imwrite(str(cikti / dosya), goruntu):
                raise SystemExit(f"Görüntü yazılamadı: {cikti / dosya}")
        print(f"{yol.name}: min={en_kucuk} max={en_buyuk} -> min={int(sonuc.min())} max={int(sonuc.max())}")
        print(f"  ortalama {gri.mean():.2f} -> {sonuc.mean():.2f}, std {gri.std():.2f} -> {sonuc.std():.2f}")


if __name__ == "__main__":
    main()
