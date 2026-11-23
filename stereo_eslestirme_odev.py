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
import cv2
import matplotlib.pyplot as plt
import numpy as np

from soru3_sad_ssd_zncc import blok_eslestirme, maliyet_sad, maliyet_ssd
from stereo_eslestirme import compute_disparity_bm, load_stereo_pair, save_disparity, stereo_paths


def soru2_kendi_blok_eslestirme(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    max_disparity: int = 32,
    block_size: int = 5,
    metric: str = "SAD",
) -> np.ndarray:
    """Soru 2: OpenCV eşleştiricisi çağırmadan blok maliyetini en aza indirir."""
    costs = {"SAD": maliyet_sad, "SSD": maliyet_ssd}
    if metric.upper() not in costs:
        raise ValueError("Metrik SAD veya SSD olmalıdır.")
    return blok_eslestirme(left_gray, right_gray, costs[metric.upper()], max_disparity, block_size)


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

    bm_u8, bm_color = save_disparity(out_dir / "soru1_stereobm", bm_disp, max_disparity)
    custom_u8, custom_color = save_disparity(out_dir / "soru2_kendi_bm", custom_disp, max_disparity)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    panels = [
        (bm_u8, f"Soru 1: OpenCV StereoBM (block={bm_block_size}, Gri)", "gray"),
        (cv2.cvtColor(bm_color, cv2.COLOR_BGR2RGB), f"Soru 1: OpenCV StereoBM (Renkli - JET)", None),
        (custom_u8, f"Soru 2: Kendi Blok Eşleştirme ({metric}, block={custom_block_size}, Gri)", "gray"),
        (cv2.cvtColor(custom_color, cv2.COLOR_BGR2RGB), f"Soru 2: Kendi Blok Eşleştirme ({metric}, JET)", None),
    ]

    for ax, (img, title, cmap) in zip(axes.flat, panels):
        ax.imshow(img, cmap=cmap, vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    fig.suptitle(
        "OpenCV StereoBM ve Sıfırdan Blok Eşleştirme",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    comparison_path = out_dir / "stereo_odev_cikti.png"
    fig.savefig(comparison_path, dpi=135, bbox_inches="tight")
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

    left_path, right_path = stereo_paths(args.left, args.right)
    _, _, left_gray, right_gray = load_stereo_pair(left_path, right_path)
    bm_disp = compute_disparity_bm(
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
