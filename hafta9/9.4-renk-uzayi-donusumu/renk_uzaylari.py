"""9.4: BGR görüntüyü HSV, HSI, YUV ve CIE Lab uzaylarına dönüştürme."""

from pathlib import Path

import cv2
import numpy as np

KLASOR = Path(__file__).resolve().parent
GIRDI = KLASOR.parent / "girdi" / "renkli.bmp"
CIKTI = KLASOR / "cikti"


def hsi_donustur(resim):
    """H/2, S*255, I sıralı uint8; en yakına yuvarla, aralığa kırp."""
    b, g, r = np.moveaxis(resim.astype(np.float64), -1, 0)
    i = (r + g + b) / 3
    h = np.degrees(np.arctan2(np.sqrt(3) * (g - b), 2 * r - g - b)) % 360
    oran = np.zeros_like(i)
    np.divide(np.minimum(np.minimum(r, g), b), i, out=oran, where=i > 0)
    s = np.where(i > 0, 1 - oran, 0)
    renksiz = (r == g) & (g == b)
    h[renksiz] = 0
    s[renksiz] = 0
    return np.stack([np.clip(np.rint(h / 2), 0, 179),
                     np.clip(np.rint(s * 255), 0, 255),
                     np.clip(np.rint(i), 0, 255)], axis=-1).astype(np.uint8)


def donustur(resim):
    if (not isinstance(resim, np.ndarray) or resim.dtype != np.uint8
            or resim.ndim != 3 or resim.shape[2] != 3 or resim.size == 0):
        raise ValueError("Görüntü boş olmayan, üç kanallı uint8 BGR olmalı.")
    return {"hsv": cv2.cvtColor(resim, cv2.COLOR_BGR2HSV),
            "hsi": hsi_donustur(resim),
            "yuv": cv2.cvtColor(resim, cv2.COLOR_BGR2YUV),
            "lab": cv2.cvtColor(resim, cv2.COLOR_BGR2Lab)}


def kaydet(yol, resim):
    if not cv2.imwrite(str(yol), resim):
        raise OSError(f"Kaydedilemedi: {yol}")


def karsilastirma(resim, donusumler):
    """Her satırda orijinal ve üç gri kanal; ham kanalları renklendirmez."""
    h, w = resim.shape[:2]
    genislik = max(w, 240)
    satirlar = []
    kanallar = {"hsv": ["H (0..179)", "S (0..255)", "V (0..255)"],
                "hsi": ["H/2 (0..179)", "S*255", "I (0..255)"],
                "yuv": ["Y", "U", "V"], "lab": ["L*255/100", "a+128", "b+128"]}
    for ad, ham in donusumler.items():
        paneller = []
        goruntuler = [resim] + [cv2.cvtColor(ham[:, :, k], cv2.COLOR_GRAY2BGR) for k in range(3)]
        for baslik, goruntu in zip(["Orijinal BGR"] + [ad.upper() + " / " + k for k in kanallar[ad]], goruntuler):
            panel = np.full((h + 38, genislik, 3), 245, np.uint8)
            panel[38:, :w] = goruntu
            cv2.putText(panel, baslik, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
            paneller.append(panel)
        satirlar.append(np.hstack(paneller))
    return np.vstack(satirlar)


def calistir(girdi=GIRDI, cikti=CIKTI):
    girdi, cikti = Path(girdi), Path(cikti)
    if not girdi.is_file():
        raise FileNotFoundError(f"Girdi bulunamadı: {girdi}")
    resim = cv2.imread(str(girdi), cv2.IMREAD_COLOR)
    if resim is None:
        raise ValueError(f"Görüntü okunamadı: {girdi}")
    donusumler = donustur(resim)
    cikti.mkdir(parents=True, exist_ok=True)
    for ad, ham in donusumler.items():
        kaydet(cikti / f"{ad}.bmp", ham)
    kaydet(cikti / "karsilastirma.png", karsilastirma(resim, donusumler))
    print("HSV, HSI, YUV, Lab BMP ve kanal karşılaştırması kaydedildi.")


if __name__ == "__main__":
    calistir()
