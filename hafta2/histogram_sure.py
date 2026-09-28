"""Ödev 4: Tek ve dört thread histogram sürelerini karşılaştır.

Kullanım: python histogram_sure.py [girdi] [cikti] [--tekrar 20]
Varsayılan: dama.jpg ve cikti klasörü.
"""
import argparse
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

try:
    from .histogram_thread import histogramlari_birlestir, parca_histogramlari, tum_histogram
except ImportError:
    from histogram_thread import histogramlari_birlestir, parca_histogramlari, tum_histogram

KLASOR = Path(__file__).resolve().parent


def histogram_4_thread(gri):
    return histogramlari_birlestir(parca_histogramlari(gri, paralel=True))


def _sure_olc(gri, hesapla, tekrar):
    baslangic = perf_counter()
    for _ in range(tekrar):
        sonuc = hesapla(gri)
    return (perf_counter() - baslangic) / tekrar, sonuc


def main():
    ap = argparse.ArgumentParser(description="Histogramı tek ve dört thread ile ölç.")
    ap.add_argument("girdi", nargs="?", type=Path, default=KLASOR / "dama.jpg")
    ap.add_argument("cikti", nargs="?", type=Path, default=KLASOR / "cikti")
    ap.add_argument("--tekrar", type=int, default=20)
    args = ap.parse_args()
    if args.tekrar < 1:
        raise SystemExit("--tekrar en az 1 olmalı.")
    gri = cv2.imread(str(args.girdi), cv2.IMREAD_GRAYSCALE)
    if gri is None:
        raise SystemExit(f"Görüntü okunamadı: {args.girdi}")
    tek, hist_tek = _sure_olc(gri, tum_histogram, args.tekrar)
    dort, hist_dort = _sure_olc(gri, histogram_4_thread, args.tekrar)
    assert np.array_equal(hist_tek, hist_dort), "Tek ve dört thread histogramları eşit değil."
    print(f"Histogram süre tablosu ({args.tekrar} tekrar, ortalama):")
    print(f"1 thread | {tek:.6f} sn")
    print(f"4 thread | {dort:.6f} sn")
    print(f"Fark (4 - 1) | {dort - tek:+.6f} sn")


if __name__ == "__main__":
    main()
