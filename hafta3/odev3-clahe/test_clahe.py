from pathlib import Path

import cv2
import numpy as np

from hazir_fonksiyonsuz import clahe

klasor = Path(__file__).parent

ornekler = {
    "dag.jpg": cv2.imread(str(klasor.parent / "girdi" / "dag.jpg"), cv2.IMREAD_GRAYSCALE),
    "sis.jpg": cv2.imread(str(klasor.parent / "girdi" / "sis.jpg"), cv2.IMREAD_GRAYSCALE),
    "rastgele 101x77": np.random.default_rng(0).integers(60, 180, (101, 77), dtype=np.uint8),
}

for ad, gri in ornekler.items():
    for clip_limit, karo in [(2.0, 8), (4.0, 4)]:
        bizim = clahe(gri, clip_limit, karo)
        opencv = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(karo, karo)).apply(gri)
        fark = np.abs(bizim.astype(int) - opencv.astype(int))
        print(f"{ad}, clip={clip_limit}, karo={karo}: en büyük fark {fark.max()}, "
              f"farklı piksel %{100 * (fark > 0).mean():.3f}")
        assert fark.max() <= 1
