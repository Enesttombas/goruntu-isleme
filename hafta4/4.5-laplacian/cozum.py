"""EBLG351 Hafta 4 - Ödev 4.5: 3x3 Laplacian Kenar Tespiti.

Bu betik:
1. Girdi kamera görüntüsüne Gauss'lu ve Gauss'suz 3x3 Laplacian operatörünü uygular.
2. 8-komşuluklu sıfır geçişi (zero-crossing) ile ikili kenar haritasını üretir.
3. Çıktı görsellerini ve panelini kaydeder.
4. Prewitt ve Sobel ile karşılaştıran karsilastirma.md ve karsilastirma.pdf belgelerini üretir.
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
    mutlak_uint8,
    panel_olustur,
    sifir_gecis,
)
from hafta4.rapor_uret import rapor_45_uret


def calistir():
    """Ödev 4.5 Laplacian adımlarını yürütür, görselleri ve raporları kaydeder."""
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

    # 1. Laplacian (Gauss'suz) - Gürültü duyarlılığını gösterir
    lap_no = laplacian(gri, yumusat=False)
    lap_no_vis = isaretli_goruntu_uint8(lap_no)
    lap_no_zc = sifir_gecis(lap_no, esik_orani=0.05)

    # 2. Laplacian (Gauss'lu)
    lap_g = laplacian(gri, yumusat=True)
    lap_g_vis = isaretli_goruntu_uint8(lap_g)
    lap_g_abs = mutlak_uint8(lap_g)
    lap_g_zc = sifir_gecis(lap_g, esik_orani=0.05)

    # 3. Görsel Panel: 2x3 Grid
    panel = panel_olustur(
        [gri, lap_no_vis, lap_no_zc, lap_g_vis, lap_g_abs, lap_g_zc],
        [
            "1. Girdi (Orijinal)",
            "2. Lap (Gauss-suz Isaretli)",
            "3. Lap (Gauss-suz Sifir Gecis)",
            "4. Lap (Gauss-lu Isaretli)",
            "5. Lap (Gauss-lu Mutlak)",
            "6. Lap (Gauss-lu Sifir Gecis)",
        ],
        (2, 3),
    )

    # Çıktıları kaydet
    cv2.imwrite(str(cikti_klasor / "kamera_laplacian_nogauss_isaretli.png"), lap_no_vis)
    cv2.imwrite(str(cikti_klasor / "kamera_laplacian_nogauss_ikili.png"), lap_no_zc)
    cv2.imwrite(str(cikti_klasor / "kamera_laplacian_gauss_isaretli.png"), lap_g_vis)
    cv2.imwrite(str(cikti_klasor / "kamera_laplacian_gauss_mutlak.png"), lap_g_abs)
    cv2.imwrite(str(cikti_klasor / "kamera_laplacian_gauss_ikili.png"), lap_g_zc)
    cv2.imwrite(str(cikti_klasor / "panel_laplacian.png"), panel)

    cv2.imwrite(str(genel_cikti / "kamera_laplacian_nogauss_isaretli.png"), lap_no_vis)
    cv2.imwrite(str(genel_cikti / "kamera_laplacian_nogauss_ikili.png"), lap_no_zc)
    cv2.imwrite(str(genel_cikti / "kamera_laplacian_gauss_isaretli.png"), lap_g_vis)
    cv2.imwrite(str(genel_cikti / "kamera_laplacian_gauss_mutlak.png"), lap_g_abs)
    cv2.imwrite(str(genel_cikti / "kamera_laplacian_gauss_ikili.png"), lap_g_zc)

    # 4. Prewitt ve Sobel ile karşılaştıran raporları üret
    rapor_45_uret(hedef_klasor=dosya_yolu.parent)

    print("Ödev 4.5 (Laplacian) tamamlandı.")


if __name__ == "__main__":
    calistir()
