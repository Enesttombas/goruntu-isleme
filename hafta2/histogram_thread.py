"""Ödev 3: Dört parçanın histogramlarını thread ile hesaplayıp birleştir.

Kullanım: python histogram_thread.py [girdi] [cikti]
Varsayılan: dama.jpg ve cikti klasörü.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

KLASOR = Path(__file__).resolve().parent


def tum_histogram(gri):
    return np.bincount(gri.ravel(), minlength=256)


def _bol(gri):
    h, w = gri.shape
    y, x = h // 2, w // 2
    return [gri[:y, :x], gri[:y, x:], gri[y:, :x], gri[y:, x:]]


def parca_histogramlari(gri, paralel=True):
    parcalar = _bol(gri)
    if paralel:
        with ThreadPoolExecutor(max_workers=4) as havuz:
            return list(havuz.map(tum_histogram, parcalar))
    return [tum_histogram(parca) for parca in parcalar]


def histogramlari_birlestir(histogramlar):
    return np.sum(histogramlar, axis=0)


def histogram_grafigi(histogram):
    yukseklik, genislik = 400, 512
    grafik = np.zeros((yukseklik, genislik), dtype=np.uint8)
    olcekli = cv2.normalize(histogram.astype(np.float32), None, 0, yukseklik - 30, cv2.NORM_MINMAX).ravel()
    for i, deger in enumerate(olcekli):
        cv2.line(grafik, (i * 2, yukseklik - 1), (i * 2, yukseklik - 1 - int(deger)), 255, 1)
    cv2.putText(grafik, "Gri seviye histogrami", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 255, 1)
    cv2.putText(grafik, "0", (4, yukseklik - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 180, 1)
    cv2.putText(grafik, "255", (genislik - 42, yukseklik - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 180, 1)
    return grafik


def main():
    ap = argparse.ArgumentParser(description="Parça histogramlarını paralel hesapla ve çiz.")
    ap.add_argument("girdi", nargs="?", type=Path, default=KLASOR / "dama.jpg")
    ap.add_argument("cikti", nargs="?", type=Path, default=KLASOR / "cikti")
    args = ap.parse_args()
    gri = cv2.imread(str(args.girdi), cv2.IMREAD_GRAYSCALE)
    if gri is None:
        raise SystemExit(f"Görüntü okunamadı: {args.girdi}")
    ana = histogramlari_birlestir(parca_histogramlari(gri, paralel=True))
    assert np.array_equal(ana, tum_histogram(gri)), "Parça ve tam görüntü histogramları eşit değil."
    args.cikti.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.cikti / "histogram.png"), histogram_grafigi(ana)):
        raise SystemExit("histogram.png yazılamadı.")
    print(f"4 parça histogramı birleşti; toplam piksel: {int(ana.sum())}; tam histogramla eşit.")


if __name__ == "__main__":
    main()
