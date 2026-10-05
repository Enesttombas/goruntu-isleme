from pathlib import Path

import cv2
import numpy as np

klasor = Path(__file__).parent

for ad in ["dag", "sis"]:
    gri = cv2.imread(str(klasor.parent / "girdi" / f"{ad}.jpg"), cv2.IMREAD_GRAYSCALE)

    # 1) 256 değerli histogram
    hist = [0] * 256
    for deger in gri.ravel().tolist():
        hist[deger] += 1

    # 2) Kümülatif dağılım (CDF)
    cdf = [0] * 256
    toplam = 0
    for i in range(256):
        toplam += hist[i]
        cdf[i] = toplam

    # 3) Eşitleme tablosu: yeni = (cdf[x] - cdf_min) / (N - cdf_min) * 255
    n = gri.size
    cdf_min = next(c for c in cdf if c > 0)
    tablo = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        tablo[i] = max(0, round((cdf[i] - cdf_min) / (n - cdf_min) * 255))

    # 4) Her pikseli tablodan geçir
    sonuc = tablo[gri]

    cv2.imwrite(str(klasor / "cikti" / f"{ad}_hazir_fonksiyonsuz.png"), sonuc)
    print(f"{ad}: std {gri.std():.2f} -> {sonuc.std():.2f}")
