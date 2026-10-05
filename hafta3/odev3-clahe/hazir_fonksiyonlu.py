from pathlib import Path

import cv2

klasor = Path(__file__).parent
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

for ad in ["dag", "sis"]:
    gri = cv2.imread(str(klasor.parent / "girdi" / f"{ad}.jpg"), cv2.IMREAD_GRAYSCALE)

    sonuc = clahe.apply(gri)

    cv2.imwrite(str(klasor / "cikti" / f"{ad}_hazir_fonksiyonlu.png"), sonuc)
    print(f"{ad}: std {gri.std():.2f} -> {sonuc.std():.2f}")
