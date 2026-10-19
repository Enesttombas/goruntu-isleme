"""Hafta 5 / 5.3: RGB -> gri dönüşümü, KENDİ yazdığımız fonksiyonla; OpenCV ile aradaki hata."""

from pathlib import Path

import cv2
import numpy as np

KLASOR = Path(__file__).resolve().parent
GIRDI = KLASOR.parent / "girdi" / "astronot.png"
CIKTI = KLASOR / "cikti"


def kendi_gri(bgr: np.ndarray) -> np.ndarray:
    """Y = 0,299 R + 0,587 G + 0,114 B (ITU-R BT.601), 14 bit tamsayı ağırlıklarla.

    OpenCV'nin iç yuvarlaması birebir aynı olmayabilir; küçük bir hata toplamı çıkabilir.
    """
    mavi = bgr[:, :, 0].astype(np.int32)
    yesil = bgr[:, :, 1].astype(np.int32)
    kirmizi = bgr[:, :, 2].astype(np.int32)
    toplam = 4899 * kirmizi + 9617 * yesil + 1868 * mavi  # 0,299 / 0,587 / 0,114 * 2^14
    return ((toplam + (1 << 13)) >> 14).astype(np.uint8)


def mutlak_fark_toplami(a: np.ndarray, b: np.ndarray) -> int:
    """Eşleşen piksellerin mutlak farklarının toplamı (uint8 taşmasına karşı int64)."""
    return int(np.abs(a.astype(np.int64) - b.astype(np.int64)).sum())


def calistir() -> None:
    renkli = cv2.imread(str(GIRDI), cv2.IMREAD_COLOR)
    if renkli is None:
        raise FileNotFoundError(f"Renkli fotoğraf okunamadı: {GIRDI}")
    CIKTI.mkdir(exist_ok=True)

    kendi = kendi_gri(renkli)
    if not cv2.imwrite(str(CIKTI / "gri_kendi.png"), kendi):
        raise OSError("Gri görüntü kaydedilemedi")

    # Hata ölçümü için karşılaştırma görüntüsü; yalnızca hata hesabında kullanılır.
    hazir = cv2.cvtColor(renkli, cv2.COLOR_BGR2GRAY)
    hata = mutlak_fark_toplami(hazir, kendi)
    satir = f"Hata (cv2.cvtColor ile kendi_gri arasındaki mutlak fark toplamı): {hata}"
    (CIKTI / "hata.txt").write_text(satir + "\n", encoding="utf-8")
    print(satir)


if __name__ == "__main__":
    calistir()
