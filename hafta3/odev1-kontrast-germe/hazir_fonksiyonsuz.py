from pathlib import Path

import cv2
import numpy as np

klasor = Path(__file__).parent

for ad in ["dag", "sis"]:
    gri = cv2.imread(str(klasor.parent / "girdi" / f"{ad}.jpg"), cv2.IMREAD_GRAYSCALE)

    en_kucuk = int(gri.min())
    en_buyuk = int(gri.max())

    # y = (x - min) * 255 / (max - min)
    sonuc = (gri.astype(np.float32) - en_kucuk) * 255 / (en_buyuk - en_kucuk)
    sonuc = np.round(sonuc).astype(np.uint8)

    cv2.imwrite(str(klasor / "cikti" / f"{ad}_hazir_fonksiyonsuz.png"), sonuc)
    print(f"{ad}: min={en_kucuk} max={en_buyuk} -> min={sonuc.min()} max={sonuc.max()}")
