from pathlib import Path

import cv2

klasor = Path(__file__).parent
girdi = klasor.parent / "girdi"

for ad in ["dag", "sis"]:
    temiz = cv2.imread(str(girdi / f"{ad}.jpg"), cv2.IMREAD_GRAYSCALE)
    gurultulu = cv2.imread(str(girdi / f"{ad}_gurultulu.png"), cv2.IMREAD_GRAYSCALE)

    sonuc = cv2.blur(gurultulu, (3, 3))

    cv2.imwrite(str(klasor / "cikti" / f"{ad}_hazir_fonksiyonlu.png"), sonuc)
    print(f"{ad}: PSNR gürültülü {cv2.PSNR(temiz, gurultulu):.2f} dB -> filtreli {cv2.PSNR(temiz, sonuc):.2f} dB")
