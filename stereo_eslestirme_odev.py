"""
Hafta 10 - Stereo Eşleştirme (Soru 1: OpenCV StereoBM & Soru 2: Kendi Blok Eşleştirme)
Ders: Görüntü İşleme

- Soru 1: OpenCV'nin cv2.StereoBM_create() fonksiyonu ile ham (filtresiz) fark haritası oluşturur.
- Soru 2: StereoBM kullanmadan sıfırdan SAD (Sum of Absolute Differences) tabanlı
          blok eşleştirme fonksiyonu ile ham (filtresiz) fark haritası oluşturur.

Kullanım:
    python stereo_eslestirme_odev.py
    python stereo_eslestirme_odev.py --left tsukuba/scene1.row3.col1.ppm --right tsukuba/scene1.row3.col3.ppm --show
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


def find_image(candidates: list[str], search_dirs: list[Path]) -> Optional[Path]:
    """Aday dosya adlarını olası klasörlerde sırayla arar."""
    for directory in search_dirs:
        if not directory.exists():
            continue
        for name in candidates:
            path = directory / name
            if path.is_file():
                return path
    return None


def load_stereo_pair(left_path: Path, right_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """1. (sol) ve 3. (sağ) görüntüleri renkli ve gri tonlamalı olarak yükler."""
    left_bgr = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
    right_bgr = cv2.imread(str(right_path), cv2.IMREAD_COLOR)

    if left_bgr is None:
        raise FileNotFoundError(f"1. resim (sol) okunamadı: {left_path}")
    if right_bgr is None:
        raise FileNotFoundError(f"3. resim (sağ) okunamadı: {right_path}")
    if left_bgr.shape[:2] != right_bgr.shape[:2]:
        raise ValueError(f"Görüntü boyutları eşleşmiyor: {left_bgr.shape[:2]} vs {right_bgr.shape[:2]}")

    left_gray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
    return left_bgr, right_bgr, left_gray, right_gray


def soru1_opencv_stereobm(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    num_disparities: int = 32,
    block_size: int = 15,
) -> np.ndarray:
    """
    Soru 1: OpenCV StereoBM ile ham fark haritası (disparity map) hesaplar.
    Ekstra yumuşatma/medyan filtresi uygulanmaz; blok eşleştirmenin doğal
    gürültüsü ve eşleşemeyen (occlusion/dokusuz) bölgeleri korunur.
    """
    stereo_bm = cv2.StereoBM_create(
        numDisparities=num_disparities,
        blockSize=block_size,
    )
    # OpenCV çıktıyı 16 ile çarpılmış sabit noktalı (int16) döndürür
    disparity = stereo_bm.compute(left_gray, right_gray).astype(np.float32) / 16.0
    return disparity


def pencere_toplami_integral(maliyet: np.ndarray, block_size: int) -> np.ndarray:
    """İntegral görüntü (Summed-Area Table) ile (block_size x block_size) pencere toplamını alır."""
    pad = block_size // 2
    padded = np.pad(maliyet, ((pad, pad), (pad, pad)), mode="edge")
    integral = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1), dtype=np.float64)
    integral[1:, 1:] = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    k = block_size
    return (
        integral[k:, k:]
        - integral[:-k, k:]
        - integral[k:, :-k]
        + integral[:-k, :-k]
    ).astype(np.float32)


def soru2_kendi_blok_eslestirme(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    max_disparity: int = 32,
    block_size: int = 5,
    metric: str = "SAD",
) -> np.ndarray:
    """
    Soru 2: StereoBM kullanmadan sıfırdan Blok Eşleştirme (SAD veya SSD).
    Küçük blok boyutu (5x5) ve sonradan filtreleme olmadan çalıştırıldığı için
    dokusuz yüzeylerde ve nesne sınırlarında gerçek blok eşleştirme gürültüsü görülür.
    """
    if block_size < 3 or block_size % 2 == 0:
        raise ValueError("block_size 3 veya daha büyük tek sayı olmalıdır.")

    left = left_gray.astype(np.float32)
    right = right_gray.astype(np.float32)
    h, w = left.shape

    en_dusuk_maliyet = np.full((h, w), np.inf, dtype=np.float32)
    disparity_map = np.zeros((h, w), dtype=np.float32)
    metric_upper = metric.upper()

    for d in range(max_disparity):
        if d == 0:
            fark = left - right
        else:
            fark = np.full((h, w), 255.0, dtype=np.float32)
            fark[:, d:] = left[:, d:] - right[:, :-d]

        piksel_maliyeti = (fark ** 2) if metric_upper == "SSD" else np.abs(fark)
        blok_maliyeti = pencere_toplami_integral(piksel_maliyeti, block_size)

        daha_iyi = blok_maliyeti < en_dusuk_maliyet
        en_dusuk_maliyet[daha_iyi] = blok_maliyeti[daha_iyi]
        disparity_map[daha_iyi] = float(d)

    return disparity_map


def normalize_to_uint8(disparity: np.ndarray, max_disparity: int = 32) -> np.ndarray:
    """Negatif (geçersiz) değerleri 0'a çekip 0-255 aralığına ölçekler."""
    clipped = np.clip(disparity, 0, max_disparity - 1)
    return (clipped / float(max_disparity - 1) * 255.0).astype(np.uint8)


def save_outputs(
    bm_disp: np.ndarray,
    custom_disp: np.ndarray,
    out_dir: Path,
    max_disparity: int = 32,
    bm_block_size: int = 15,
    custom_block_size: int = 5,
    metric: str = "SAD",
    show: bool = False,
) -> Path:
    """Soru 1 ve Soru 2 çıktılarını tekil ve karşılaştırmalı olarak kaydeder."""
    out_dir.mkdir(parents=True, exist_ok=True)

    bm_u8 = normalize_to_uint8(bm_disp, max_disparity)
    bm_color = cv2.applyColorMap(bm_u8, cv2.COLORMAP_JET)

    custom_u8 = normalize_to_uint8(custom_disp, max_disparity)
    custom_color = cv2.applyColorMap(custom_u8, cv2.COLORMAP_JET)

    cv2.imwrite(str(out_dir / "soru1_stereobm_gray.png"), bm_u8)
    cv2.imwrite(str(out_dir / "soru1_stereobm_jet.png"), bm_color)
    cv2.imwrite(str(out_dir / "soru2_kendi_bm_gray.png"), custom_u8)
    cv2.imwrite(str(out_dir / "soru2_kendi_bm_jet.png"), custom_color)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    panels = [
        (bm_u8, f"Soru 1: OpenCV StereoBM (block={bm_block_size}, Gri)", "gray"),
        (cv2.cvtColor(bm_color, cv2.COLOR_BGR2RGB), f"Soru 1: OpenCV StereoBM (Renkli - JET)", None),
        (custom_u8, f"Soru 2: Kendi Blok Eşleştirme ({metric}, block={custom_block_size}, Gri)", "gray"),
        (cv2.cvtColor(custom_color, cv2.COLOR_BGR2RGB), f"Soru 2: Kendi Blok Eşleştirme ({metric}, JET)", None),
    ]

    for ax, (img, title, cmap) in zip(axes.flat, panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    fig.suptitle(
        "Tsukuba Stereo Eşleştirme - OpenCV StereoBM ve Sıfırdan Blok Eşleştirme (Ham Çıktılar)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    comparison_path = out_dir / "stereo_odev_cikti.png"
    plt.savefig(comparison_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return comparison_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Hafta 10 Stereo Eşleştirme (Soru 1 ve Soru 2)")
    parser.add_argument("--left", type=Path, default=None, help="1. resim (scene1.row3.col1.ppm)")
    parser.add_argument("--right", type=Path, default=None, help="3. resim (scene1.row3.col3.ppm)")
    parser.add_argument("--out", type=Path, default=Path("outputs"), help="Çıktı klasörü")
    parser.add_argument("--max-disparity", type=int, default=32, help="Maksimum disparity (16'nın katı)")
    parser.add_argument("--bm-block-size", type=int, default=15, help="OpenCV StereoBM blok boyutu")
    parser.add_argument("--custom-block-size", type=int, default=5, help="Kendi blok eşleştirme blok boyutu")
    parser.add_argument("--metric", type=str, default="SAD", choices=["SAD", "SSD"], help="Maliyet metriği")
    parser.add_argument("--show", action="store_true", help="Grafiği ekranda göster")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    search_dirs = [base_dir, base_dir / "tsukuba", base_dir / "data", base_dir / "images"]

    left_path = args.left or find_image(
        ["scene1.row3.col1.ppm", "im1.ppm", "im1.png", "left.png", "1.png"], search_dirs
    )
    right_path = args.right or find_image(
        ["scene1.row3.col3.ppm", "im3.ppm", "im3.png", "right.png", "3.png"], search_dirs
    )

    if left_path is None or right_path is None:
        raise FileNotFoundError(
            "Tsukuba 1. ve 3. resimleri bulunamadı! "
            "Lütfen 'scene1.row3.col1.ppm' ve 'scene1.row3.col3.ppm' dosyalarını klasöre ekleyin."
        )

    _, _, left_gray, right_gray = load_stereo_pair(left_path, right_path)

    bm_disp = soru1_opencv_stereobm(
        left_gray, right_gray, num_disparities=args.max_disparity, block_size=args.bm_block_size
    )
    custom_disp = soru2_kendi_blok_eslestirme(
        left_gray, right_gray, max_disparity=args.max_disparity, block_size=args.custom_block_size, metric=args.metric
    )

    out_path = save_outputs(
        bm_disp=bm_disp,
        custom_disp=custom_disp,
        out_dir=args.out,
        max_disparity=args.max_disparity,
        bm_block_size=args.bm_block_size,
        custom_block_size=args.custom_block_size,
        metric=args.metric,
        show=args.show,
    )
    print(f"[TAMAMLANDI] Çıktı kaydedildi: {out_path}")


if __name__ == "__main__":
    main()
