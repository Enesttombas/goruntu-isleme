"""
Hafta 10 - Ödev 3: Stereo Eşleştirmede SAD, SSD ve ZNCC Yöntemlerinin Karşılaştırılması
Ders: Görüntü İşleme

Middlebury Tsukuba veri setindeki 1. resim (scene1.row3.col1.ppm) ve
3. resim (scene1.row3.col3.ppm) kullanılarak, OpenCV StereoBM kullanmadan
3 farklı blok benzerlik metriği (SAD, SSD, ZNCC) test edilir.

Not: Ödevdeki "benzer parametre sayı ve türleri ile fonksiyon pointer" yapısına
uygun olarak, Python'da 3 metrik fonksiyonu da birebir aynı imzayla yazılmış
ve ana eşleştirme fonksiyonuna parametre (first-class function / Callable) olarak
aktarılmıştır.

Kullanım:
    python soru3_sad_ssd_zncc.py
    python soru3_sad_ssd_zncc.py --left tsukuba/scene1.row3.col1.ppm --right tsukuba/scene1.row3.col3.ppm --show
"""

import argparse
import time
from pathlib import Path
from typing import Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np

from stereo_eslestirme import load_stereo_pair, save_disparity, stereo_paths, validate_stereo

# Fonksiyon pointer karşılığı tip tanımı: (sol, kaydirilmis_sag, blok_boyutu) -> blok_maliyeti
MaliyetFonksiyonu = Callable[[np.ndarray, np.ndarray, int], np.ndarray]


def pencere_toplami_integral(matris: np.ndarray, block_size: int) -> np.ndarray:
    """İntegral görüntü (Summed-Area Table) ile (block_size x block_size) pencere toplamı hesaplar."""
    pad = block_size // 2
    padded = np.pad(matris, ((pad, pad), (pad, pad)), mode="edge")
    integral = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1), dtype=np.float64)
    integral[1:, 1:] = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    k = block_size
    return (
        integral[k:, k:]
        - integral[:-k, k:]
        - integral[k:, :-k]
        + integral[:-k, :-k]
    )


# -------------------------------------------------------------------------
# 3 FARKLI BLOK BENZERLİK FONKSİYONU (Birebir aynı parametre imzası ile)
# Hepsi "küçük değer = daha yüksek benzerlik (düşük maliyet)" döndürür.
# -------------------------------------------------------------------------

def maliyet_sad(left: np.ndarray, right_shifted: np.ndarray, block_size: int) -> np.ndarray:
    """
    1. SAD (Sum of Absolute Differences - Mutlak Farklar Toplamı):
       SAD = sum(|I_L - I_R|)
    """
    piksel_farki = np.abs(np.subtract(left, right_shifted, dtype=np.float64))
    return pencere_toplami_integral(piksel_farki, block_size).astype(np.float32)


def maliyet_ssd(left: np.ndarray, right_shifted: np.ndarray, block_size: int) -> np.ndarray:
    """
    2. SSD (Sum of Squared Differences - Kare Farklar Toplamı):
       SSD = sum((I_L - I_R)^2)
    """
    piksel_farki = np.subtract(left, right_shifted, dtype=np.float64) ** 2
    return pencere_toplami_integral(piksel_farki, block_size).astype(np.float32)


def maliyet_zncc(left: np.ndarray, right_shifted: np.ndarray, block_size: int) -> np.ndarray:
    """
    3. ZNCC (Zero-Mean Normalized Cross-Correlation - Sıfır Ortalamalı Normalize Çapraz Korelasyon):
       ZNCC in [-1, +1] aralığındadır (+1 en iyi eşleşme).
       Tüm yöntemlerle aynı "minimizasyon" arayüzünü kullanmak için maliyet olarak (1 - ZNCC) döndürülür.
    """
    left = np.asarray(left, dtype=np.float64)
    right_shifted = np.asarray(right_shifted, dtype=np.float64)
    n = float(block_size * block_size)

    sum_l = pencere_toplami_integral(left, block_size)
    sum_r = pencere_toplami_integral(right_shifted, block_size)
    sum_l2 = pencere_toplami_integral(left * left, block_size)
    sum_r2 = pencere_toplami_integral(right_shifted * right_shifted, block_size)
    sum_lr = pencere_toplami_integral(left * right_shifted, block_size)

    # Cebirsel özdeşlik: sum((L - mean_L)(R - mean_R)) = sum(L*R) - (sum_L * sum_R) / N
    pay = sum_lr - (sum_l * sum_r) / n
    var_l = np.maximum(sum_l2 - (sum_l * sum_l) / n, 0.0)
    var_r = np.maximum(sum_r2 - (sum_r * sum_r) / n, 0.0)

    payda = np.sqrt(var_l * var_r)
    valid = payda > 1e-6
    zncc = np.divide(pay, payda, out=np.zeros_like(pay), where=valid)
    # Varyansı sıfır olan bloklarda ZNCC tanımsızdır.
    return np.where(valid, 1.0 - np.clip(zncc, -1, 1), np.inf).astype(np.float32)


def blok_maliyetleri(left_gray, right_gray, maliyet_fn, max_disparity, block_size):
    """Her d için yalnız iki görüntüye de bütünüyle sığan blokları değerlendirir."""
    validate_stereo(left_gray, right_gray, max_disparity, block_size)
    left = left_gray.astype(np.float64)
    right = right_gray.astype(np.float64)
    h, w = left.shape
    r = block_size // 2
    for d in range(max_disparity):
        # Sol x ile sağ x-d eşleşir. İki kırpımda aynı sütun merkezleri kullanılır.
        cost = maliyet_fn(left[:, d:], right[:, :w - d], block_size)
        result = np.full((h, w), np.inf, dtype=np.float32)
        result[r:h-r, d+r:w-r] = cost[r:h-r, r:w-d-r]
        yield result


def blok_eslestirme(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    maliyet_fn: MaliyetFonksiyonu,
    max_disparity: int = 32,
    block_size: int = 7,
) -> np.ndarray:
    """En düşük maliyetli geçerli d'yi seçer. Aday yoksa -1 döndürür."""
    en_dusuk_maliyet = np.full(left_gray.shape, np.inf, dtype=np.float32)
    disparity_map = np.full(left_gray.shape, -1, dtype=np.float32)
    for d, cost in enumerate(blok_maliyetleri(left_gray, right_gray, maliyet_fn, max_disparity, block_size)):
        daha_iyi = cost < en_dusuk_maliyet
        en_dusuk_maliyet[daha_iyi] = cost[daha_iyi]
        disparity_map[daha_iyi] = d
    return disparity_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Ödev 3: SAD, SSD ve ZNCC blok eşleştirme karşılaştırması")
    parser.add_argument("--left", type=Path, default=None, help="1. resim (scene1.row3.col1.ppm)")
    parser.add_argument("--right", type=Path, default=None, help="3. resim (scene1.row3.col3.ppm)")
    parser.add_argument("--out", type=Path, default=Path("outputs"), help="Çıktı klasörü")
    parser.add_argument("--max-disparity", type=int, default=32, help="Maksimum disparity aralığı")
    parser.add_argument("--block-size", type=int, default=7, help="Blok boyutu (tek sayı, varsayılan: 7)")
    parser.add_argument("--show", action="store_true", help="Karşılaştırma grafiğini ekranda göster")
    args = parser.parse_args()

    left_path, right_path = stereo_paths(args.left, args.right)
    _, _, left_gray, right_gray = load_stereo_pair(left_path, right_path)

    # Aynı imzaya sahip 3 fonksiyonu sözlükte topla (C++ fonksiyon pointer tablosu karşılığı)
    yontemler: dict[str, MaliyetFonksiyonu] = {
        "SAD": maliyet_sad,
        "SSD": maliyet_ssd,
        "ZNCC": maliyet_zncc,
    }

    sonuclar_u8: dict[str, np.ndarray] = {}
    sonuclar_jet: dict[str, np.ndarray] = {}
    sureler_ms: dict[str, float] = {}

    args.out.mkdir(parents=True, exist_ok=True)

    for ad, maliyet_fn in yontemler.items():
        t0 = time.perf_counter()
        disp = blok_eslestirme(
            left_gray=left_gray,
            right_gray=right_gray,
            maliyet_fn=maliyet_fn,
            max_disparity=args.max_disparity,
            block_size=args.block_size,
        )
        sure_ms = (time.perf_counter() - t0) * 1000.0
        sureler_ms[ad] = sure_ms

        disp_u8, disp_jet = save_disparity(args.out / f"soru3_{ad.lower()}", disp, args.max_disparity)
        sonuclar_u8[ad] = disp_u8
        sonuclar_jet[ad] = disp_jet
        print(f"[{ad:4s}] Süre: {sure_ms:6.1f} ms | Geçerli piksel: {np.mean(disp >= 0):.1%}")

    # 2x3 Karşılaştırma Grafiği (Üst satır: Gri, Alt satır: Renkli JET)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for col, ad in enumerate(["SAD", "SSD", "ZNCC"]):
        axes[0, col].imshow(sonuclar_u8[ad], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axes[0, col].set_title(f"{ad} (Gri) - {sureler_ms[ad]:.0f} ms", fontsize=11)
        axes[0, col].axis("off")

        axes[1, col].imshow(cv2.cvtColor(sonuclar_jet[ad], cv2.COLOR_BGR2RGB), interpolation="nearest")
        axes[1, col].set_title(f"{ad} (Renkli JET) - block={args.block_size}x{args.block_size}", fontsize=11)
        axes[1, col].axis("off")

    fig.suptitle(
        f"Ödev 3: Stereo Eşleştirme Benzerlik Metrikleri Karşılaştırması (SAD vs SSD vs ZNCC)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    comparison_path = args.out / "soru3_sad_ssd_zncc_cikti.png"
    fig.savefig(comparison_path, dpi=135, bbox_inches="tight")
    if args.show:
        plt.show()
    plt.close(fig)

    print(f"[TAMAMLANDI] Karşılaştırma grafiği kaydedildi: {comparison_path}")


if __name__ == "__main__":
    main()
