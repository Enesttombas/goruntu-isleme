from pathlib import Path

import cv2

klasor = Path(__file__).parent

for ad in ["dag", "sis"]:
    gri = cv2.imread(str(klasor.parent / "girdi" / f"{ad}.jpg"), cv2.IMREAD_GRAYSCALE)

    sonuc = cv2.equalizeHist(gri)

    cv2.imwrite(str(klasor / "cikti" / f"{ad}_hazir_fonksiyonlu.png"), sonuc)
    print(f"{ad}: std {gri.std():.2f} -> {sonuc.std():.2f}")
