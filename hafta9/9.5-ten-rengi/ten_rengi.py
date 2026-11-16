"""9.5: RGB, YCbCr ve HSV eşikleriyle örnek ten maskeleri."""

from pathlib import Path

import cv2
import numpy as np

KLASOR = Path(__file__).resolve().parent
GIRDI = KLASOR.parent / "girdi" / "renkli.bmp"
CIKTI = KLASOR / "cikti"


def rgb_maskesi(resim):
    # İşaretli tür, R-G gibi farkların uint8 içinde taşmasını önler.
    b, g, r = np.moveaxis(resim.astype(np.int16), -1, 0)
    return ((r > 95) & (g > 40) & (b > 20)
            & (np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b) > 15)
            & (np.abs(r - g) > 15) & (r > g) & (r > b))


def ycbcr_maskesi(resim):
    # OpenCV'nin dizilimi Y, Cr, Cb; yöntem adı literatürdeki YCbCr'dir.
    ycrcb = cv2.cvtColor(resim, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    return (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)


def hsv_maskesi(resim):
    h, s, v = np.moveaxis(cv2.cvtColor(resim, cv2.COLOR_BGR2HSV), -1, 0)
    return ((h <= 20) | (h >= 170)) & (s >= 40) & (v >= 60)


def kirmizi_yap(resim, maske):
    sonuc = resim.copy()
    sonuc[~maske] = (0, 0, 255)
    return sonuc


def kaydet(yol, resim):
    if not cv2.imwrite(str(yol), resim):
        raise OSError(f"Kaydedilemedi: {yol}")


def calistir(girdi=GIRDI, cikti=CIKTI):
    girdi, cikti = Path(girdi), Path(cikti)
    if not girdi.is_file():
        raise FileNotFoundError(f"Girdi bulunamadı: {girdi}")
    resim = cv2.imread(str(girdi), cv2.IMREAD_COLOR)
    if resim is None:
        raise ValueError(f"Görüntü okunamadı: {girdi}")
    cikti.mkdir(parents=True, exist_ok=True)
    h, w = resim.shape[:2]
    paneller, satirlar = [], []
    goruntuler = [("Orijinal", resim)]
    for ad, fonksiyon in [("rgb", rgb_maskesi), ("ycbcr", ycbcr_maskesi), ("hsv", hsv_maskesi)]:
        maske = fonksiyon(resim)
        sonuc = kirmizi_yap(resim, maske)
        kaydet(cikti / f"{ad}_maske.bmp", maske.astype(np.uint8) * 255)
        kaydet(cikti / f"{ad}.bmp", sonuc)
        sayi = int(np.count_nonzero(maske))
        etiket = "YCbCr" if ad == "ycbcr" else ad.upper()
        satir = f"{etiket}: {sayi}/{maske.size} piksel (%{100 * sayi / maske.size:.2f})"
        satirlar.append(satir)
        goruntuler.append((satir, sonuc))
    for baslik, goruntu in goruntuler:
        panel = np.full((h + 40, max(w, 480), 3), 245, np.uint8)
        panel[40:, :w] = goruntu
        cv2.putText(panel, baslik, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        paneller.append(panel)
    kaydet(cikti / "karsilastirma.png", np.hstack(paneller))
    rapor = "\n".join(satirlar) + "\n"
    (cikti / "sayim.txt").write_text(rapor, encoding="utf-8")
    print(rapor, end="")


if __name__ == "__main__":
    calistir()
