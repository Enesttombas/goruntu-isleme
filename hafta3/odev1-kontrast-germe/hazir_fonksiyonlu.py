from pathlib import Path

import cv2

klasor = Path(__file__).parent

for ad in ["dag", "sis"]:
    gri = cv2.imread(str(klasor.parent / "girdi" / f"{ad}.jpg"), cv2.IMREAD_GRAYSCALE)

    en_kucuk, en_buyuk, _, _ = cv2.minMaxLoc(gri)
    sonuc = cv2.normalize(gri, None, 0, 255, cv2.NORM_MINMAX)

    cv2.imwrite(str(klasor / "cikti" / f"{ad}_hazir_fonksiyonlu.png"), sonuc)
    print(f"{ad}: min={int(en_kucuk)} max={int(en_buyuk)} -> min={sonuc.min()} max={sonuc.max()}")
