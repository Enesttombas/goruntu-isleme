"""Ödev 5: y = 0.75*x + 20 parlaklık dönüşümünü ayrı etkilerle karşılaştır.

Kullanım: python parlaklik_donusum.py [girdi] [cikti]
Varsayılan: dama.jpg ve cikti klasörü.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

KLASOR = Path(__file__).resolve().parent


def parlaklik_donusumleri(gri):
    x = gri.astype(np.float32)
    carp = np.clip(0.75 * x, 0, 255).astype(np.uint8)
    topla = np.clip(x + 20, 0, 255).astype(np.uint8)
    birlikte = np.clip(0.75 * x + 20, 0, 255).astype(np.uint8)
    return carp, topla, birlikte


def karsilastirma_gorseli(goruntuler):
    etiketler = ("Orijinal", "0.75 * x", "x + 20", "0.75 * x + 20")
    hucreler = []
    for goruntu, etiket in zip(goruntuler, etiketler):
        genislik = min(600, goruntu.shape[1])
        yukseklik = max(1, round(goruntu.shape[0] * genislik / goruntu.shape[1]))
        kucuk = cv2.resize(goruntu, (genislik, yukseklik), interpolation=cv2.INTER_AREA)
        hucre = cv2.copyMakeBorder(kucuk, 32, 0, 0, 0, cv2.BORDER_CONSTANT, value=0)
        cv2.putText(hucre, etiket, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1, cv2.LINE_AA)
        hucreler.append(hucre)
    return cv2.vconcat([cv2.hconcat(hucreler[:2]), cv2.hconcat(hucreler[2:])])


def main():
    ap = argparse.ArgumentParser(description="Parlaklık ve kontrast dönüşümlerini karşılaştır.")
    ap.add_argument("girdi", nargs="?", type=Path, default=KLASOR / "dama.jpg")
    ap.add_argument("cikti", nargs="?", type=Path, default=KLASOR / "cikti")
    args = ap.parse_args()
    gri = cv2.imread(str(args.girdi), cv2.IMREAD_GRAYSCALE)
    if gri is None:
        raise SystemExit(f"Görüntü okunamadı: {args.girdi}")
    carp, topla, birlikte = parlaklik_donusumleri(gri)
    args.cikti.mkdir(parents=True, exist_ok=True)
    dosyalar = (("carp_075.png", carp), ("topla_20.png", topla), ("donusum.png", birlikte),
                ("karsilastirma.png", karsilastirma_gorseli((gri, carp, topla, birlikte))))
    for ad, goruntu in dosyalar:
        if not cv2.imwrite(str(args.cikti / ad), goruntu):
            raise SystemExit(f"Görüntü yazılamadı: {args.cikti / ad}")
    print(f"Ortalama parlaklık: önce {gri.mean():.2f}, sonra {birlikte.mean():.2f}")
    print(f"Min/max: önce {int(gri.min())}/{int(gri.max())}, sonra {int(birlikte.min())}/{int(birlikte.max())}")


if __name__ == "__main__":
    main()
