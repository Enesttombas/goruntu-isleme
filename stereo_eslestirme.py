"""
Hafta 10 - Stereo Eşleştirme (Stereo Matching) ve Fark Haritası (Disparity Map)
Middlebury Tsukuba veri setindeki 1. resim (scene1.row3.col1.ppm) ve
3. resim (scene1.row3.col3.ppm) kullanılarak OpenCV StereoBM ve StereoSGBM
ile fark haritası oluşturur.
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


def find_image(candidates: list[str], search_dirs: list[Path]) -> Optional[Path]:
    for directory in search_dirs:
        if not directory.exists():
            continue
        for name in candidates:
            path = directory / name
            if path.is_file():
                return path
    return None


def load_stereo_pair(left_path: Path, right_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def compute_disparity_bm(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    num_disparities: int = 32,
    block_size: int = 15,
) -> np.ndarray:
    """OpenCV StereoBM (Block Matching) ile fark haritası hesaplar."""
    stereo_bm = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
    return stereo_bm.compute(left_gray, right_gray).astype(np.float32) / 16.0


def compute_disparity_sgbm(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    min_disparity: int = 0,
    num_disparities: int = 32,
    block_size: int = 5,
) -> np.ndarray:
    """OpenCV StereoSGBM (Semi-Global Block Matching) ile fark haritası hesaplar."""
    stereo_sgbm = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * (block_size ** 2),
        P2=32 * (block_size ** 2),
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    return stereo_sgbm.compute(left_gray, right_gray).astype(np.float32) / 16.0


def normalize_disparity(disparity: np.ndarray) -> np.ndarray:
    valid = np.clip(disparity, 0, None)
    return cv2.normalize(valid, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def save_comparison(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    bm_u8: np.ndarray,
    sgbm_u8: np.ndarray,
    sgbm_color_bgr: np.ndarray,
    output_path: Path,
    gt_gray: Optional[np.ndarray] = None,
    show: bool = False,
) -> None:
    left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)
    sgbm_color_rgb = cv2.cvtColor(sgbm_color_bgr, cv2.COLOR_BGR2RGB)

    if gt_gray is not None:
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        items = [
            (left_rgb, "1. Resim (Sol - col1)", None),
            (right_rgb, "3. Resim (Sağ - col3)", None),
            (gt_gray, "Referans Harita (Ground Truth)", "gray"),
            (bm_u8, "StereoBM (Gri)", "gray"),
            (sgbm_u8, "StereoSGBM (Gri)", "gray"),
            (sgbm_color_rgb, "StereoSGBM (Renkli - JET)", None),
        ]
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        items = [
            (left_rgb, "1. Resim (Sol - col1)", None),
            (right_rgb, "3. Resim (Sağ - col3)", None),
            (bm_u8, "StereoBM Fark Haritası", "gray"),
            (sgbm_color_rgb, "StereoSGBM Renkli Fark Haritası (JET)", None),
        ]

    for ax, (img, title, cmap) in zip(axes.flat, items):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle("Tsukuba Stereo Eşleştirme - Fark Haritası (Disparity Map)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tsukuba 1. ve 3. resimleriyle stereo fark haritası üretir.")
    parser.add_argument("--left", type=Path, default=None, help="1. resim (scene1.row3.col1.ppm)")
    parser.add_argument("--right", type=Path, default=None, help="3. resim (scene1.row3.col3.ppm)")
    parser.add_argument("--gt", type=Path, default=None, help="Opsiyonel ground truth (truedisp.row3.col3.pgm)")
    parser.add_argument("--out", type=Path, default=Path("outputs"), help="Çıktı klasörü")
    parser.add_argument("--num-disparities", type=int, default=32, help="Disparity aralığı (16'nın katı, col1-col3 için 32 önerilir)")
    parser.add_argument("--show", action="store_true", help="Çıktı grafiğini ekranda göster")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    search_dirs = [base_dir, base_dir / "tsukuba", base_dir / "data", base_dir / "images"]

    left_path = args.left or find_image(
        ["scene1.row3.col1.ppm", "im1.ppm", "im1.png", "left.png", "1.png"], search_dirs
    )
    right_path = args.right or find_image(
        ["scene1.row3.col3.ppm", "im3.ppm", "im3.png", "right.png", "3.png"], search_dirs
    )
    gt_path = args.gt or find_image(
        ["truedisp.row3.col3.pgm", "truedisp.png", "ground_truth.png"], search_dirs
    )

    if left_path is None or right_path is None:
        raise FileNotFoundError(
            "1. resim (scene1.row3.col1.ppm) ve 3. resim (scene1.row3.col3.ppm) bulunamadı. "
            "Dosyaları bu klasöre veya 'tsukuba/' altına koyun ya da --left / --right ile belirtin."
        )

    left_bgr, right_bgr, left_gray, right_gray = load_stereo_pair(left_path, right_path)

    disp_bm = compute_disparity_bm(left_gray, right_gray, num_disparities=args.num_disparities)
    bm_u8 = normalize_disparity(disp_bm)
    bm_color = cv2.applyColorMap(bm_u8, cv2.COLORMAP_JET)

    disp_sgbm = compute_disparity_sgbm(left_gray, right_gray, num_disparities=args.num_disparities)
    sgbm_u8 = normalize_disparity(disp_sgbm)
    sgbm_color = cv2.applyColorMap(sgbm_u8, cv2.COLORMAP_JET)

    gt_gray = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE) if (gt_path and gt_path.is_file()) else None

    args.out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out / "disparity_bm_gray.png"), bm_u8)
    cv2.imwrite(str(args.out / "disparity_bm_color.png"), bm_color)
    cv2.imwrite(str(args.out / "disparity_sgbm_gray.png"), sgbm_u8)
    cv2.imwrite(str(args.out / "disparity_sgbm_color.png"), sgbm_color)

    comparison_path = args.out / "stereo_comparison.png"
    save_comparison(left_bgr, right_bgr, bm_u8, sgbm_u8, sgbm_color, comparison_path, gt_gray, args.show)
    print(f"Kaydedildi: {comparison_path}")


if __name__ == "__main__":
    main()
