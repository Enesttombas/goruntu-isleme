"""Ödev 2: histogram, CDF ve histogram eşitleme (hazır fonksiyon yok).

Adımlar: 256'lık histogram -> CDF -> eşitleme tablosu -> görüntüye uygula.
Kullanım: python histogram_esitleme.py [girdi ...]
Varsayılan: dag.jpg ve sis.jpg; çıktılar cikti/ klasörüne yazılır.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

KLASOR = Path(__file__).resolve().parent


def histogram_cikar(gri):
    hist = [0] * 256
    for deger in gri.ravel().tolist():
        hist[deger] += 1
    return hist


def cdf_hesapla(hist):
    cdf = []
    toplam = 0
    for adet in hist:
        toplam += adet
        cdf.append(toplam)
    return cdf


def esitle(gri, hist, cdf):
    n = gri.size
    cdf_min = next(c for c in cdf if c > 0)
    if n == cdf_min:
        return gri.copy()
    tablo = np.zeros(256, dtype=np.uint8)
    for deger in range(256):
        tablo[deger] = min(255, max(0, round((cdf[deger] - cdf_min) / (n - cdf_min) * 255)))
    return tablo[gri]


def cizim(veri, baslik, cdf_mi):
    """256 değeri 512x300 beyaz tuvale çubuk (histogram) veya çizgi (CDF) olarak çizer."""
    g, y = 512, 300
    tuval = np.full((y + 30, g, 3), 255, dtype=np.uint8)
    tepe = max(max(veri), 1)
    noktalar = [(2 * i, y + 30 - 1 - int(v / tepe * (y - 1))) for i, v in enumerate(veri)]
    for (x, ust) in noktalar if not cdf_mi else []:
        cv2.rectangle(tuval, (x, ust), (x + 1, y + 29), (80, 80, 80), -1)
    if cdf_mi:
        cv2.polylines(tuval, [np.array(noktalar, dtype=np.int32)], False, (200, 60, 0), 2, cv2.LINE_AA)
    cv2.putText(tuval, baslik, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return tuval


def grafik_ciz(yol, hist_once, hist_sonra, cdf_once, cdf_sonra):
    ust = cv2.hconcat([cizim(hist_once, "Histogram (once)", False), cizim(hist_sonra, "Histogram (sonra)", False)])
    alt = cv2.hconcat([cizim(cdf_once, "CDF (once)", True), cizim(cdf_sonra, "CDF (sonra)", True)])
    if not cv2.imwrite(str(yol), cv2.vconcat([ust, alt])):
        raise SystemExit(f"Görüntü yazılamadı: {yol}")


def yan_yana(once, sonra):
    hucreler = []
    for goruntu, etiket in ((once, "Once"), (sonra, "Esitlenmis")):
        hucre = cv2.copyMakeBorder(goruntu, 32, 0, 0, 0, cv2.BORDER_CONSTANT, value=0)
        cv2.putText(hucre, etiket, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1, cv2.LINE_AA)
        hucreler.append(hucre)
    return cv2.hconcat(hucreler)


def main():
    ap = argparse.ArgumentParser(description="Histogram eşitleme.")
    ap.add_argument("girdiler", nargs="*", type=Path, default=[KLASOR / "dag.jpg", KLASOR / "sis.jpg"])
    args = ap.parse_args()
    cikti = KLASOR / "cikti"
    cikti.mkdir(exist_ok=True)
    for yol in args.girdiler:
        gri = cv2.imread(str(yol), cv2.IMREAD_GRAYSCALE)
        if gri is None:
            raise SystemExit(f"Görüntü okunamadı: {yol}")
        hist = histogram_cikar(gri)
        cdf = cdf_hesapla(hist)
        sonuc = esitle(gri, hist, cdf)
        hist_sonra = histogram_cikar(sonuc)
        ad = yol.stem
        for dosya, goruntu in ((f"{ad}_esitlenmis.png", sonuc),
                               (f"{ad}_esitleme_karsilastirma.png", yan_yana(gri, sonuc))):
            if not cv2.imwrite(str(cikti / dosya), goruntu):
                raise SystemExit(f"Görüntü yazılamadı: {cikti / dosya}")
        grafik_ciz(cikti / f"{ad}_histogram_cdf.png", hist, hist_sonra, cdf, cdf_hesapla(hist_sonra))
        print(f"{yol.name}: piksel={gri.size} hist toplamı={sum(hist)} CDF son={cdf[-1]}")
        print(f"  min/max {int(gri.min())}/{int(gri.max())} -> {int(sonuc.min())}/{int(sonuc.max())}, "
              f"std {gri.std():.2f} -> {sonuc.std():.2f}")


if __name__ == "__main__":
    main()
