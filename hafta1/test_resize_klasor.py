import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

BURASI = Path(__file__).parent


def test_klasordeki_resimler_1024x768_olur(tmp_path):
    girdi = tmp_path / "girdi"
    girdi.mkdir()
    cv2.imwrite(str(girdi / "yatay.png"), np.zeros((300, 500, 3), np.uint8))
    cv2.imwrite(str(girdi / "dikey.jpg"), np.zeros((900, 400, 3), np.uint8))
    (girdi / "not.txt").write_text("resim değil")

    cikti = tmp_path / "cikti"
    subprocess.run([sys.executable, str(BURASI / "resize_klasor.py"), str(girdi), str(cikti)], check=True)

    sonuclar = sorted(p.name for p in cikti.iterdir())
    assert sonuclar == ["dikey.jpg", "yatay.png"]
    for ad in sonuclar:
        assert cv2.imread(str(cikti / ad)).shape == (768, 1024, 3)
