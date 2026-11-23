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
from matplotlib.colors import ListedColormap, Normalize


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


def stereo_paths(left: Path | None, right: Path | None) -> tuple[Path, Path]:
    base_dir = Path(__file__).resolve().parent
    search_dirs = [base_dir, base_dir / "tsukuba", base_dir / "data", base_dir / "images"]
    left = left or find_image(["scene1.row3.col1.ppm", "im1.ppm", "im1.png", "left.png", "1.png"], search_dirs)
    right = right or find_image(["scene1.row3.col3.ppm", "im3.ppm", "im3.png", "right.png", "3.png"], search_dirs)
    if left is None or right is None:
        raise FileNotFoundError("Stereo çifti bulunamadı. README'deki Tsukuba dosyalarını indirin veya --left ve --right verin.")
    return left, right


def validate_stereo(left: np.ndarray, right: np.ndarray, disparities: int, block_size: int) -> None:
    if left.ndim != 2 or left.shape != right.shape:
        raise ValueError("Sol ve sağ görüntü aynı boyutta gri tonlamalı diziler olmalıdır.")
    if block_size < 1 or block_size % 2 == 0 or block_size > min(left.shape):
        raise ValueError("Blok boyutu görüntüye sığan pozitif bir tek sayı olmalıdır.")
    if not 1 <= disparities <= left.shape[1] - block_size + 1:
        raise ValueError("Disparity sayısı en az 1 olmalı ve en büyük kaymada en az bir blok sığmalıdır.")


def validate_opencv(left: np.ndarray, right: np.ndarray, disparities: int, block_size: int) -> None:
    validate_stereo(left, right, disparities, block_size)
    if disparities % 16 or disparities >= left.shape[1]:
        raise ValueError("OpenCV disparity sayısı 16'nın katı ve görüntü genişliğinden küçük olmalıdır.")
    if left.dtype != np.uint8 or right.dtype != np.uint8:
        raise ValueError("OpenCV eşleştirme 8 bit gri tonlamalı görüntü bekler.")


def compute_disparity_bm(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    num_disparities: int = 32,
    block_size: int = 15,
) -> np.ndarray:
    """OpenCV StereoBM (Block Matching) ile fark haritası hesaplar."""
    validate_opencv(left_gray, right_gray, num_disparities, block_size)
    if not 5 <= block_size <= 255:
        raise ValueError("StereoBM blok boyutu 5 ile 255 arasında bir tek sayı olmalıdır.")
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
    validate_opencv(left_gray, right_gray, num_disparities, block_size)
    stereo_sgbm = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * (block_size ** 2),
        P2=32 * (block_size ** 2),
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    return stereo_sgbm.compute(left_gray, right_gray).astype(np.float32) / 16.0


def normalize_disparity(disparity: np.ndarray, disparities: int) -> np.ndarray:
    valid = np.isfinite(disparity) & (disparity >= 0)
    gray = np.zeros(disparity.shape, dtype=np.uint8)
    gray[valid] = np.rint(255 * np.clip(disparity[valid] / max(1, disparities - 1), 0, 1))
    return gray


def colorize_disparity(disparity: np.ndarray, disparities: int) -> np.ndarray:
    color = cv2.applyColorMap(normalize_disparity(disparity, disparities), cv2.COLORMAP_JET)
    color[~np.isfinite(disparity) | (disparity < 0)] = 0
    return color


def save_disparity(
    prefix: Path, disparity: np.ndarray, disparities: int, color_suffix: str = "jet"
) -> tuple[np.ndarray, np.ndarray]:
    gray = normalize_disparity(disparity, disparities)
    color = colorize_disparity(disparity, disparities)
    for suffix, image in (("gray", gray), (color_suffix, color)):
        path = prefix.with_name(f"{prefix.name}_{suffix}.png")
        if not cv2.imwrite(str(path), image):
            raise OSError(f"Görüntü kaydedilemedi: {path}")
    np.save(prefix.with_suffix(".npy"), disparity)
    return gray, color


def save_comparison(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    bm_color_bgr: np.ndarray,
    sgbm_color_bgr: np.ndarray,
    output_path: Path,
    disparities: int,
    gt_gray: Optional[np.ndarray] = None,
    show: bool = False,
) -> None:
    left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)
    sgbm_color_rgb = cv2.cvtColor(sgbm_color_bgr, cv2.COLOR_BGR2RGB)
    bm_color_rgb = cv2.cvtColor(bm_color_bgr, cv2.COLOR_BGR2RGB)

    if gt_gray is not None:
        fig, axes = plt.subplots(2, 3, figsize=(15, 7), layout="constrained")
        items = [
            (left_rgb, "1. Resim (Sol - col1)", None),
            (right_rgb, "3. Resim (Sağ - col3)", None),
            (gt_gray, "Sağlanan referans (ham gri ton)", "gray"),
            (bm_color_rgb, "StereoBM (JET)", None),
            (sgbm_color_rgb, "StereoSGBM (JET)", None),
        ]
        axes.flat[-1].axis("off")
    else:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
        items = [
            (left_rgb, "Sol görüntü", None),
            (right_rgb, "Sağ görüntü", None),
            (bm_color_rgb, "StereoBM (JET)", None),
            (sgbm_color_rgb, "StereoSGBM (JET)", None),
        ]

    for ax, (img, title, cmap) in zip(axes.flat, items):
        ax.imshow(img, cmap=cmap, interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle("Stereo Eşleştirme – Fark Haritası", fontsize=14, fontweight="bold")
    jet = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)[:, 0, ::-1] / 255
    scale = plt.cm.ScalarMappable(Normalize(0, max(1, disparities - 1)), cmap=ListedColormap(jet))
    fig.colorbar(scale, ax=list(axes.flat), shrink=0.85, label="Disparity (px) · Siyah = geçersiz")
    fig.savefig(output_path, dpi=135)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tsukuba 1. ve 3. resimleriyle stereo fark haritası üretir.")
    parser.add_argument("--left", type=Path, default=None, help="1. resim (scene1.row3.col1.ppm)")
    parser.add_argument("--right", type=Path, default=None, help="3. resim (scene1.row3.col3.ppm)")
    parser.add_argument("--gt", type=Path, default=None, help="Sol görüntüye hizalı isteğe bağlı referans")
    parser.add_argument("--out", type=Path, default=Path("outputs"), help="Çıktı klasörü")
    parser.add_argument("--num-disparities", type=int, default=32, help="Disparity aralığı (16'nın katı, col1-col3 için 32 önerilir)")
    parser.add_argument("--show", action="store_true", help="Çıktı grafiğini ekranda göster")
    args = parser.parse_args()

    left_path, right_path = stereo_paths(args.left, args.right)
    left_bgr, right_bgr, left_gray, right_gray = load_stereo_pair(left_path, right_path)

    disp_bm = compute_disparity_bm(left_gray, right_gray, num_disparities=args.num_disparities)
    disp_sgbm = compute_disparity_sgbm(left_gray, right_gray, num_disparities=args.num_disparities)
    gt_gray = None
    if args.gt is not None:
        gt_gray = cv2.imread(str(args.gt), cv2.IMREAD_GRAYSCALE)
        if gt_gray is None or gt_gray.shape != left_gray.shape:
            parser.error("--gt okunabilir ve sol görüntüyle aynı boyutta olmalıdır.")

    args.out.mkdir(parents=True, exist_ok=True)
    _, bm_color = save_disparity(args.out / "disparity_bm", disp_bm, args.num_disparities, "color")
    _, sgbm_color = save_disparity(args.out / "disparity_sgbm", disp_sgbm, args.num_disparities, "color")

    comparison_path = args.out / "stereo_comparison.png"
    save_comparison(left_bgr, right_bgr, bm_color, sgbm_color, comparison_path, args.num_disparities, gt_gray, args.show)
    print(f"Kaydedildi: {comparison_path}")


if __name__ == "__main__":
    main()
