from pathlib import Path

import cv2
import numpy as np

klasor = Path(__file__).parent
girdi = klasor.parent / "girdi"
rng = np.random.default_rng(0)

for ad in ["dag", "sis"]:
    gri = cv2.imread(str(girdi / f"{ad}.jpg"), cv2.IMREAD_GRAYSCALE)

    # Ortalaması 0, standart sapması 20 olan Gauss gürültüsü
    gurultu = rng.normal(0, 20, gri.shape)
    gurultulu = np.clip(gri + gurultu, 0, 255).astype(np.uint8)

    cv2.imwrite(str(girdi / f"{ad}_gurultulu.png"), gurultulu)
