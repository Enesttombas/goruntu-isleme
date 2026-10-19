"""Hafta 5 / 5.3: RGB -> gri dönüşümü (OpenCV ve kendi fonksiyonum) ve mutlak fark toplamı."""

from pathlib import Path

import cv2
import numpy as np


KLASOR = Path(__file__).resolve().parent
GIRDI = KLASOR.parent / "girdi" / "astronot.png"
CIKTI = KLASOR / "cikti"


def kendi_gri(bgr: np.ndarray) -> np.ndarray:
    """Y = 0,299 R + 0,587 G + 0,114 B (ITU-R BT.601) ağırlıklarıyla gri üret.

    Ağırlıklar 14 bit tamsayıya çevrilip yuvarlanır. OpenCV'nin iç yuvarlaması
    birebir aynı olmayabilir, bu yüzden küçük bir hata toplamı çıkabilir.
    """
    mavi = bgr[:, :, 0].astype(np.int32)
    yesil = bgr[:, :, 1].astype(np.int32)
    kirmizi = bgr[:, :, 2].astype(np.int32)
    toplam = 4899 * kirmizi + 9617 * yesil + 1868 * mavi  # 0,299 / 0,587 / 0,114 * 2^14
    return ((toplam + (1 << 13)) >> 14).astype(np.uint8)


def kendi_gri_ondalik(bgr: np.ndarray) -> np.ndarray:
    """Aynı formülün ondalık sürümü; yalnızca yuvarlama farkını göstermek için."""
    b, g, r = (bgr[:, :, i].astype(np.float64) for i in range(3))
    return np.round(0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)


def mutlak_fark_toplami(a: np.ndarray, b: np.ndarray) -> int:
    """Eşleşen piksellerin mutlak farklarının toplamı (uint8 taşmasına karşı int64)."""
    return int(np.abs(a.astype(np.int64) - b.astype(np.int64)).sum())


def calistir() -> None:
    renkli = cv2.imread(str(GIRDI), cv2.IMREAD_COLOR)
    if renkli is None:
        raise FileNotFoundError(f"Renkli fotoğraf okunamadı: {GIRDI}")
    CIKTI.mkdir(exist_ok=True)

    hazir = cv2.cvtColor(renkli, cv2.COLOR_BGR2GRAY)
    kendi = kendi_gri(renkli)
    ondalik = kendi_gri_ondalik(renkli)
    for ad, resim in (("gri_hazir_fonksiyon", hazir), ("gri_kendi_fonksiyon", kendi)):
        if not cv2.imwrite(str(CIKTI / f"{ad}.png"), resim):
            raise OSError(f"{ad} kaydedilemedi")

    hata = mutlak_fark_toplami(hazir, kendi)
    hata_ondalik = mutlak_fark_toplami(hazir, ondalik)
    satirlar = [
        f"Görüntü: {renkli.shape[1]}x{renkli.shape[0]}, {hazir.size} piksel",
        f"Hata (cv2.cvtColor ile kendi_gri arasındaki mutlak fark toplamı): {hata}",
        f"Bilgi: ondalık yuvarlamalı sürümle fark toplamı: {hata_ondalik}",
    ]
    (CIKTI / "hata.txt").write_text("\n".join(satirlar) + "\n", encoding="utf-8")
    print("\n".join(satirlar))


if __name__ == "__main__":
    calistir()
