from pathlib import Path

import cv2
import numpy as np

klasor = Path(__file__).parent
girdi = klasor.parent / "girdi"
rng = np.random.default_rng(0)

for ad in ["dag", "sis"]:
    gri = cv2.imread(str(girdi / f"{ad}.jpg"), cv2.IMREAD_GRAYSCALE)

    # Piksellerin %5'i siyah (biber), %5'i beyaz (tuz)
    zar = rng.random(gri.shape)
    gurultulu = gri.copy()
    gurultulu[zar < 0.05] = 0
    gurultulu[zar > 0.95] = 255

    cv2.imwrite(str(girdi / f"{ad}_tuz_biber.png"), gurultulu)
