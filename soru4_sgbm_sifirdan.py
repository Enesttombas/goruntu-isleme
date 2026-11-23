"""
Hafta 10 - Ödev 4: SGBM / SGBC (Semi-Global Block Matching) Algoritmasının Sıfırdan Gerçeklenmesi
Ders: Görüntü İşleme

OpenCV'nin StereoSGBM fonksiyonunu KULLANMADAN, Semi-Global Block Matching
(Hirschmüller SGM) algoritmasını sıfırdan uygular:
  1. Maliyet Hacmi (Cost Volume): Her piksel ve her d kayması için blok SAD maliyeti hesaplanır.
  2. Çok Yönlü Maliyet Toplama (8-Yönlü Dinamik Programlama):
     Sol->Sağ, Sağ->Sol, Yukarı->Aşağı, Aşağı->Yukarı ve 4 çapraz yönde
     Smoothness (yumuşak geçiş) cezaları P1 ve P2 kullanılarak enerji minimize edilir.
  3. Winner-Takes-All (WTA): Tüm yönlerden gelen toplam maliyetin en küçük olduğu d seçilir.

Kullanım:
    python soru4_sgbm_sifirdan.py
    python soru4_sgbm_sifirdan.py --left tsukuba/scene1.row3.col1.ppm --right tsukuba/scene1.row3.col3.ppm --show
    python soru4_sgbm_sifirdan.py --p1 15 --p2 150 --block-size 5 --show
"""

import argparse
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


from soru3_sad_ssd_zncc import blok_maliyetleri, maliyet_sad
from stereo_eslestirme import load_stereo_pair, save_disparity, stereo_paths


def maliyet_hacmi_olustur(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    max_disparity: int = 32,
    block_size: int = 5,
) -> np.ndarray:
    """C(y, x, d): ortalama SAD maliyeti, geometrik olarak geçersiz adaylarda sonsuz."""
    maliyetler = blok_maliyetleri(left_gray, right_gray, maliyet_sad, max_disparity, block_size)
    hacim = np.empty((*left_gray.shape, max_disparity), dtype=np.float32)
    for d, maliyet in enumerate(maliyetler):
        hacim[:, :, d] = maliyet / (block_size * block_size)
    return hacim


def _sgm_adim_guncelle(cur_cost: np.ndarray, prev_l: np.ndarray, p1: float, p2: float) -> np.ndarray:
    """
    Hirschmüller SGM dinamik programlama güncelleme kuralı:
      L(p, d) = C(p, d) + min(
          L(p-r, d),
          L(p-r, d-1) + P1,
          L(p-r, d+1) + P1,
          min_k L(p-r, k) + P2
      ) - min_k L(p-r, k)
    """
    prev_min = np.min(prev_l, axis=-1, keepdims=True)  # (N, 1)

    # d - 1 ve d + 1 komşu disparity maliyetleri
    l_minus1 = np.empty_like(prev_l)
    l_minus1[:, 0] = np.inf
    l_minus1[:, 1:] = prev_l[:, :-1] + p1

    l_plus1 = np.empty_like(prev_l)
    l_plus1[:, -1] = np.inf
    l_plus1[:, :-1] = prev_l[:, 1:] + p1

    # Diğer tüm büyük sıçramalar (|d - d'| > 1) için P2 cezası
    l_jump = prev_min + p2

    best_prev = np.minimum(np.minimum(prev_l, l_jump), np.minimum(l_minus1, l_plus1))
    return cur_cost + best_prev - prev_min


def yonlu_maliyet_topla(cost_volume: np.ndarray, dy: int, dx: int, p1: float, p2: float) -> np.ndarray:
    """
    Belirtilen (dy, dx) yönü boyunca SGM maliyet biriktirmesi (path aggregation) yapar.
    Desteklenen 8 yön: (0,1), (0,-1), (1,0), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)
    """
    h, w, _ = cost_volume.shape
    l_path = cost_volume.copy()

    # Yatay yönler: Soldan sağa (0, 1) ve Sağdan sola (0, -1)
    if dy == 0 and dx == 1:
        for x in range(1, w):
            l_path[:, x, :] = _sgm_adim_guncelle(cost_volume[:, x, :], l_path[:, x - 1, :], p1, p2)
    elif dy == 0 and dx == -1:
        for x in range(w - 2, -1, -1):
            l_path[:, x, :] = _sgm_adim_guncelle(cost_volume[:, x, :], l_path[:, x + 1, :], p1, p2)

    # Dikey yönler: Yukarıdan aşağı (1, 0) ve Aşağıdan yukarı (-1, 0)
    elif dy == 1 and dx == 0:
        for y in range(1, h):
            l_path[y, :, :] = _sgm_adim_guncelle(cost_volume[y, :, :], l_path[y - 1, :, :], p1, p2)
    elif dy == -1 and dx == 0:
        for y in range(h - 2, -1, -1):
            l_path[y, :, :] = _sgm_adim_guncelle(cost_volume[y, :, :], l_path[y + 1, :, :], p1, p2)

    # Çapraz yönler
    elif dy == 1 and dx == 1:  # Sol-Üst -> Sağ-Alt
        for y in range(1, h):
            l_path[y, 1:, :] = _sgm_adim_guncelle(cost_volume[y, 1:, :], l_path[y - 1, :-1, :], p1, p2)
    elif dy == -1 and dx == -1:  # Sağ-Alt -> Sol-Üst
        for y in range(h - 2, -1, -1):
            l_path[y, :-1, :] = _sgm_adim_guncelle(cost_volume[y, :-1, :], l_path[y + 1, 1:, :], p1, p2)
    elif dy == 1 and dx == -1:  # Sağ-Üst -> Sol-Alt
        for y in range(1, h):
            l_path[y, :-1, :] = _sgm_adim_guncelle(cost_volume[y, :-1, :], l_path[y - 1, 1:, :], p1, p2)
    elif dy == -1 and dx == 1:  # Sol-Alt -> Sağ-Üst
        for y in range(h - 2, -1, -1):
            l_path[y, 1:, :] = _sgm_adim_guncelle(cost_volume[y, 1:, :], l_path[y + 1, :-1, :], p1, p2)
    else:
        raise ValueError(f"Desteklenmeyen yön: ({dy}, {dx})")

    return l_path


def kendi_sgbm(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    max_disparity: int = 32,
    block_size: int = 5,
    p1: float = 10.0,
    p2: float = 120.0,
    num_paths: int = 8,
) -> np.ndarray:
    """
    Sıfırdan Semi-Global Block Matching (SGBM / SGBC) algoritması.

    Parametreler:
      - p1 (Smoothness - Küçük geçiş cezası): Komşu pikseller arasında 1 piksellik
        disparity değişimine (eğimli yüzeyler) verilen düşük yumuşatma cezası.
      - p2 (Smoothness - Büyük sıçrama cezası): Komşu pikseller arasında 1'den büyük
        disparity sıçramalarına verilen yüksek ceza (p2 > p1 olmalıdır).
    """
    if not (np.isfinite(p1) and np.isfinite(p2) and 0 <= p1 <= p2):
        raise ValueError("Cezalar sonlu olmalı ve 0 <= P1 <= P2 koşulunu sağlamalıdır.")
    if num_paths not in (4, 8):
        raise ValueError("Yön sayısı 4 veya 8 olmalıdır.")

    cost_volume = maliyet_hacmi_olustur(left_gray, right_gray, max_disparity, block_size)
    h, w = left_gray.shape
    radius = block_size // 2
    # Sınırda hiçbir blok sığmaz. Her iç pikselde en az d=0 sonlu bir adaydır.
    interior = np.s_[radius:h-radius, radius:w-radius]
    valid_costs = cost_volume[interior]

    if p1 == 0 and p2 == 0:
        toplam_maliyet = valid_costs
    else:
        paths = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        if num_paths == 8:
            paths += [(1, 1), (-1, -1), (1, -1), (-1, 1)]

        toplam_maliyet = np.zeros(valid_costs.shape, dtype=np.float32)
        for dy, dx in paths:
            toplam_maliyet += yonlu_maliyet_topla(valid_costs, dy, dx, p1, p2)

    disparity_map = np.full((h, w), -1, dtype=np.float32)
    disparity_map[interior] = np.argmin(toplam_maliyet, axis=2)
    return disparity_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Ödev 4: Sıfırdan SGBM (SGBC) ve Smoothness analizi")
    parser.add_argument("--left", type=Path, default=None, help="1. resim (scene1.row3.col1.ppm)")
    parser.add_argument("--right", type=Path, default=None, help="3. resim (scene1.row3.col3.ppm)")
    parser.add_argument("--out", type=Path, default=Path("outputs"), help="Çıktı klasörü")
    parser.add_argument("--max-disparity", type=int, default=32, help="Maksimum disparity aralığı")
    parser.add_argument("--block-size", type=int, default=5, help="Blok boyutu (tek sayı, varsayılan: 5)")
    parser.add_argument("--p1", type=float, default=10.0, help="Smoothness P1 (1 px değişim cezası)")
    parser.add_argument("--p2", type=float, default=120.0, help="Smoothness P2 (>1 px sıçrama cezası)")
    parser.add_argument("--paths", type=int, default=8, choices=[4, 8], help="SGM yön sayısı (4 veya 8)")
    parser.add_argument("--show", action="store_true", help="Karşılaştırma grafiğini ekranda göster")
    args = parser.parse_args()

    left_path, right_path = stereo_paths(args.left, args.right)
    _, _, left_gray, right_gray = load_stereo_pair(left_path, right_path)
    args.out.mkdir(parents=True, exist_ok=True)

    # Smoothness parametresinin etkisini göstermek için 3 farklı ayar test edilir:
    # 1) Smoothness yok (P1=0, P2=0) -> Saf Blok Eşleştirme
    # 2) Kullanıcının seçtiği cezalar
    # 3) Aynı maliyet üzerinde daha yüksek cezalar
    ayarlar = [
        ("Smoothness Kapalı (P1=0, P2=0)", 0.0, 0.0, "sgbm_smooth_yok"),
        (f"SGBM (P1={args.p1:g}, P2={args.p2:g})", args.p1, args.p2, "sgbm_dengeli"),
        (f"Yüksek Smoothness (P1={args.p1*5:g}, P2={args.p2*4:g})", args.p1 * 5, args.p2 * 4, "sgbm_yuksek"),
    ]

    sonuclar = []
    for baslik, p1_val, p2_val, dosya_eki in ayarlar:
        t0 = time.perf_counter()
        disp = kendi_sgbm(
            left_gray=left_gray,
            right_gray=right_gray,
            max_disparity=args.max_disparity,
            block_size=args.block_size,
            p1=p1_val,
            p2=p2_val,
            num_paths=args.paths,
        )
        sure_ms = (time.perf_counter() - t0) * 1000.0
        u8, jet = save_disparity(args.out / f"soru4_{dosya_eki}", disp, args.max_disparity)
        sonuclar.append((baslik, u8, jet, sure_ms))
        print(f"[{baslik}] Süre: {sure_ms:.1f} ms | Ort. geçerli disparity: {disp[disp >= 0].mean():.2f} px")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for col, (baslik, u8, jet, sure_ms) in enumerate(sonuclar):
        axes[0, col].imshow(u8, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axes[0, col].set_title(f"{baslik}\n(Gri - {sure_ms:.0f} ms)", fontsize=10)
        axes[0, col].axis("off")

        axes[1, col].imshow(cv2.cvtColor(jet, cv2.COLOR_BGR2RGB), interpolation="nearest")
        axes[1, col].set_title(f"{baslik}\n(Renkli JET)", fontsize=10)
        axes[1, col].axis("off")

    fig.suptitle(
        f"Ödev 4: Sıfırdan SGBM ({args.paths}-Yönlü) ve Smoothness (P1, P2) Parametresinin Etkisi",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    out_path = args.out / "soru4_sgbm_smoothness_cikti.png"
    fig.savefig(out_path, dpi=135, bbox_inches="tight")
    if args.show:
        plt.show()
    plt.close(fig)
    print(f"[TAMAMLANDI] Karşılaştırma grafiği kaydedildi: {out_path}")


if __name__ == "__main__":
    main()
