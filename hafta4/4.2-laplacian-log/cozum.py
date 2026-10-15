"""EBLG351 Hafta 4 - Ödev 4.2: Laplacian ve Laplacian of Gaussian (LoG) Kenar Tespiti.

Bu betik:
1. Gauss yumuşatmalı ve yumuşatmasız 3x3 Laplacian filtresini uygular.
2. 7x7 analitik LoG çekirdeği ile LoG yanıtını hesaplar.
3. İkinci türev haritalarında 8-komşuluklu sıfır geçişi (zero-crossing) ile kenar belirler.
4. Gauss yumuşatmanın 2. türev filtrelerindeki gürültü bastırma etkisini görselleştirir.
"""

from pathlib import Path
import sys
import cv2
import numpy as np

dosya_yolu = Path(__file__).resolve()
hafta4_dizin = dosya_yolu.parent.parent
repo_kok = hafta4_dizin.parent
for d in [str(repo_kok), str(hafta4_dizin)]:
    if d not in sys.path:
        sys.path.insert(0, d)

from hafta4.kenarlar import (
    isaretli_goruntu_uint8,
    laplacian,
    log_filtresi,
    mutlak_uint8,
    sifir_gecis,
)


def panel_olustur(gorseller: list, basliklar: list, grid_boyut: tuple) -> np.ndarray:
    """Görselleri ve başlıklarını grid panel halinde birleştirir."""
    satir, sutun = grid_boyut
    h, w = gorseller[0].shape[:2]
    banner_h = 30
    hucre_h = h + banner_h
    panel = np.full((satir * hucre_h, sutun * w, 3), 245, dtype=np.uint8)

    for idx, (img, baslik) in enumerate(zip(gorseller, basliklar)):
        r = idx // sutun
        c = idx % sutun
        y0 = r * hucre_h
        x0 = c * w

        cv2.rectangle(
            panel, (x0, y0), (x0 + w, y0 + banner_h), (40, 40, 40), thickness=-1
        )
        cv2.putText(
            panel,
            baslik,
            (x0 + 10, y0 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        if len(img.shape) == 2:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = img
        panel[y0 + banner_h : y0 + hucre_h, x0 : x0 + w] = img_bgr

    return panel


def calistir():
    cikti_klasor = dosya_yolu.parent / "cikti"
    cikti_klasor.mkdir(parents=True, exist_ok=True)
    genel_cikti = hafta4_dizin / "cikti"
    genel_cikti.mkdir(parents=True, exist_ok=True)

    girdi_yol = hafta4_dizin / "girdi" / "kamera.png"
    if not girdi_yol.exists():
        raise FileNotFoundError(f"Girdi dosyası bulunamadı: {girdi_yol}")

    gri = cv2.imread(str(girdi_yol), cv2.IMREAD_GRAYSCALE)
    if gri is None:
        raise ValueError("kamera.png okunamadı!")

    # 1. Laplacian (Gauss'suz) - Gürültü hassasiyetini gösterir
    lap_nogauss = laplacian(gri, yumusat=False)
    lap_nogauss_vis = isaretli_goruntu_uint8(lap_nogauss)
    lap_nogauss_zc = sifir_gecis(lap_nogauss, esik_orani=0.05)

    # 2. Laplacian (Gauss'lu)
    lap_gauss = laplacian(gri, yumusat=True)
    lap_gauss_vis = isaretli_goruntu_uint8(lap_gauss)
    lap_gauss_abs = mutlak_uint8(lap_gauss)
    lap_gauss_zc = sifir_gecis(lap_gauss, esik_orani=0.05)

    # 3. LoG (Analitik 7x7 çekirdek)
    log_res = log_filtresi(gri)
    log_vis = isaretli_goruntu_uint8(log_res)
    log_abs = mutlak_uint8(log_res)
    log_zc = sifir_gecis(log_res, esik_orani=0.05)

    kayitlar = {
        "kamera_laplacian_nogauss_isaretli.png": lap_nogauss_vis,
        "kamera_laplacian_nogauss_ikili.png": lap_nogauss_zc,
        "kamera_laplacian_gauss_isaretli.png": lap_gauss_vis,
        "kamera_laplacian_gauss_mutlak.png": lap_gauss_abs,
        "kamera_laplacian_gauss_ikili.png": lap_gauss_zc,
        "kamera_log_isaretli.png": log_vis,
        "kamera_log_mutlak.png": log_abs,
        "kamera_log_ikili.png": log_zc,
    }

    for ad, gorsel in kayitlar.items():
        cv2.imwrite(str(cikti_klasor / ad), gorsel)
        cv2.imwrite(str(genel_cikti / ad), gorsel)

    # Karşılaştırma Paneli (2x3)
    panel = panel_olustur(
        gorseller=[
            lap_nogauss_vis,
            lap_gauss_vis,
            log_vis,
            lap_nogauss_zc,
            lap_gauss_zc,
            log_zc,
        ],
        basliklar=[
            "Laplacian (Gauss Yok) Isaretli",
            "Laplacian (Gauss Var) Isaretli",
            "LoG (7x7 sigma=1) Isaretli",
            "Laplacian (Gauss Yok) Sifir Gecis",
            "Laplacian (Gauss Var) Sifir Gecis",
            "LoG (7x7 sigma=1) Sifir Gecis",
        ],
        grid_boyut=(2, 3),
    )
    cv2.imwrite(str(cikti_klasor / "karsilastirma_laplacian_log.png"), panel)
    cv2.imwrite(str(genel_cikti / "karsilastirma_laplacian_log.png"), panel)

    print("Ödev 4.2 (Laplacian - LoG) başarıyla tamamlandı.")


if __name__ == "__main__":
    calistir()
