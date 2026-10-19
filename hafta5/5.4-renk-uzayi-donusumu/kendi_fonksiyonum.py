"""Hafta 5 / 5.4: renk uzayı dönüşümleri KENDİ fonksiyonlarımızla; OpenCV çıktısıyla hata karşılaştırması.

Python'da "fonksiyon pointer" karşılığı fonksiyon nesnesidir: her dönüşüm
DONUSUMLER sözlüğünde (kendi fonksiyon, OpenCV kodu) çiftine bağlanır ve döngüde çağrılır.
"""

from pathlib import Path

import cv2
import numpy as np

KLASOR = Path(__file__).resolve().parent
GIRDI = KLASOR.parent / "girdi" / "astronot.png"
CIKTI = KLASOR / "cikti"


def kanallar(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (bgr[:, :, i].astype(np.int32) for i in range(3))  # B, G, R


def sinirla(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 255).astype(np.uint8)


def bgr_gri(bgr: np.ndarray) -> np.ndarray:
    """Y = 0,299 R + 0,587 G + 0,114 B, 15 bit tamsayı ağırlıklarla."""
    b, g, r = kanallar(bgr)
    return sinirla((9798 * r + 19235 * g + 3735 * b + (1 << 14)) >> 15)


def _luma(r, g, b):
    """14 bit tamsayı Y; YCrCb ve YUV ortak kullanır."""
    return (4899 * r + 9617 * g + 1868 * b + (1 << 13)) >> 14


def bgr_ycrcb(bgr: np.ndarray) -> np.ndarray:
    b, g, r = kanallar(bgr)
    y = _luma(r, g, b)
    delta = (128 << 14) + (1 << 13)
    cr = ((r - y) * 11682 + delta) >> 14  # 0,713 * 2^14
    cb = ((b - y) * 9241 + delta) >> 14  # 0,564 * 2^14
    return sinirla(np.dstack([y, cr, cb]))


def bgr_yuv(bgr: np.ndarray) -> np.ndarray:
    b, g, r = kanallar(bgr)
    y = _luma(r, g, b)
    delta = (128 << 14) + (1 << 13)
    u = ((b - y) * 8061 + delta) >> 14  # 0,492 * 2^14
    v = ((r - y) * 14369 + delta) >> 14  # 0,877 * 2^14
    return sinirla(np.dstack([y, u, v]))


def bgr_xyz(bgr: np.ndarray) -> np.ndarray:
    b, g, r = kanallar(bgr)
    x = (1689 * r + 1465 * g + 739 * b + (1 << 11)) >> 12  # katsayı * 2^12
    y = (871 * r + 2929 * g + 296 * b + (1 << 11)) >> 12
    z = (79 * r + 488 * g + 3892 * b + (1 << 11)) >> 12
    return sinirla(np.dstack([x, y, z]))


def bgr_hsv(bgr: np.ndarray) -> np.ndarray:
    """H: 0-179 (derece/2), S ve V: 0-255."""
    b, g, r = kanallar(bgr)
    v = np.maximum(np.maximum(b, g), r)
    fark = v - np.minimum(np.minimum(b, g), r)
    guvenli_v = np.maximum(v, 1)
    guvenli_f = np.maximum(fark, 1)
    sdiv = ((255 << 12) + guvenli_v // 2) // guvenli_v  # 255 / v, 12 bit sabit nokta
    s = (fark * sdiv + (1 << 11)) >> 12
    s = np.where(v == 0, 0, s)
    # Hue: 6 dilimli renk çemberi; 180/(6*fark) çarpanı 12 bit sabit noktadır.
    bolen = np.rint((180 << 12) / (6.0 * guvenli_f)).astype(np.int64)
    h = np.where(v == r, g - b, np.where(v == g, b - r + 2 * fark, r - g + 4 * fark))
    h = (h.astype(np.int64) * bolen + (1 << 11)) >> 12
    h = np.where(h < 0, h + 180, h)
    h = np.where(fark == 0, 0, h)
    return sinirla(np.dstack([h, s, v]))


def bgr_hls(bgr: np.ndarray) -> np.ndarray:
    """H: 0-179, L ve S: 0-255 (OpenCV 8-bit için kayan noktayla hesaplar)."""
    b, g, r = (bgr[:, :, i].astype(np.float32) * np.float32(1 / 255) for i in range(3))
    vmax = np.maximum(np.maximum(b, g), r)
    vmin = np.minimum(np.minimum(b, g), r)
    fark = vmax - vmin
    l = (vmax + vmin) * np.float32(0.5)
    payda = np.where(l < 0.5, vmax + vmin, np.float32(2) - (vmax + vmin))
    s = np.where(fark > np.finfo(np.float32).eps, fark / np.maximum(payda, np.float32(1e-12)), 0)
    fark_g = np.where(fark > np.finfo(np.float32).eps, np.float32(60) / np.maximum(fark, np.float32(1e-12)), 0)
    h = np.where(vmax == r, (g - b) * fark_g, np.where(vmax == g, (b - r) * fark_g + 120, (r - g) * fark_g + 240))
    h = np.where(h < 0, h + 360, h)
    h = np.where(s == 0, 0, h)
    cikis = np.dstack([h * np.float32(0.5), l * 255, s * 255])
    return np.clip(np.rint(cikis), 0, 255).astype(np.uint8)


# Her dönüşüm: ad -> (kendi fonksiyonumuz, OpenCV dönüşüm kodu)
DONUSUMLER = {
    "gri": (bgr_gri, cv2.COLOR_BGR2GRAY),
    "ycrcb": (bgr_ycrcb, cv2.COLOR_BGR2YCrCb),
    "yuv": (bgr_yuv, cv2.COLOR_BGR2YUV),
    "xyz": (bgr_xyz, cv2.COLOR_BGR2XYZ),
    "hsv": (bgr_hsv, cv2.COLOR_BGR2HSV),
    "hls": (bgr_hls, cv2.COLOR_BGR2HLS),
}


def mutlak_fark_toplami(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.abs(a.astype(np.int64) - b.astype(np.int64)).sum())


def calistir() -> None:
    renkli = cv2.imread(str(GIRDI), cv2.IMREAD_COLOR)
    if renkli is None:
        raise FileNotFoundError(f"Renkli fotoğraf okunamadı: {GIRDI}")
    CIKTI.mkdir(exist_ok=True)

    satirlar = []
    for ad, (kendi_fonksiyon, opencv_kodu) in DONUSUMLER.items():
        kendi = kendi_fonksiyon(renkli)
        if not cv2.imwrite(str(CIKTI / f"{ad}_kendi.png"), kendi):
            raise OSError(f"{ad} kaydedilemedi")
        # Karşılaştırma görüntüsü; yalnızca hata hesabında kullanılır.
        opencv = cv2.cvtColor(renkli, opencv_kodu)
        fark = np.abs(kendi.astype(np.int64) - opencv.astype(np.int64))
        satirlar.append(f"{ad}: hata toplami = {int(fark.sum())}, farkli piksel = {int((fark > 0).sum())}, en buyuk fark = {int(fark.max())}")
    (CIKTI / "hata.txt").write_text("\n".join(satirlar) + "\n", encoding="utf-8")
    print("\n".join(satirlar))


if __name__ == "__main__":
    calistir()
