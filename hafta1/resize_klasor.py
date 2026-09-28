"""Hafta 1 / Ödev: Bir klasördeki tüm resimleri 1024x768 boyutuna ölçekle.

Kullanım:
    python resize_klasor.py girdi cikti
"""
import argparse
import time
from pathlib import Path

import cv2

HEDEF_GENISLIK = 1024
HEDEF_YUKSEKLIK = 768
RESIM_UZANTILARI = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def olcekle(resim):
    # cv2.resize boyutu (GENİŞLİK, YÜKSEKLİK) sırasıyla ister; resim.shape ise (yükseklik, genişlik, kanal) verir.
    return cv2.resize(resim, (HEDEF_GENISLIK, HEDEF_YUKSEKLIK), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("girdi", type=Path)
    ap.add_argument("cikti", type=Path)
    args = ap.parse_args()

    args.cikti.mkdir(parents=True, exist_ok=True)
    dosyalar = sorted(p for p in args.girdi.iterdir() if p.suffix.lower() in RESIM_UZANTILARI)
    if not dosyalar:
        raise SystemExit(f"{args.girdi} içinde resim yok.")

    baslangic = time.perf_counter()
    for yol in dosyalar:
        resim = cv2.imread(str(yol))
        if resim is None:                        # bozuk/okunamayan dosya: atla, çökme
            print(f"ATLANDI (okunamadı): {yol.name}")
            continue
        yeni = olcekle(resim)
        cv2.imwrite(str(args.cikti / yol.name), yeni)
        print(f"{yol.name}: {resim.shape[1]}x{resim.shape[0]} -> {yeni.shape[1]}x{yeni.shape[0]}")
    print(f"{len(dosyalar)} dosya, {time.perf_counter() - baslangic:.3f} sn")


if __name__ == "__main__":
    main()
