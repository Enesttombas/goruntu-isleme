"""EBLG351 Hafta 4 - Ödev 4.6: 7x7 Analitik LoG Kenar Tespiti.

Bu betik:
1. Girdi kamera görüntüsüne 7x7 analitik LoG filtresini uygular.
2. İşaretli, mutlak ve 8-komşuluklu sıfır geçişi (zero-crossing) ile ikili kenar haritasını üretir.
3. Çıktı görsellerini ve panelini kaydeder.
4. Diğer yöntemlerle karşılaştıran karsilastirma.md ve karsilastirma.pdf belgelerini üretir.
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
    log_filtresi,
    mutlak_uint8,
    panel_olustur,
    sifir_gecis,
)
from hafta4.rapor_uret import rapor_46_uret


def calistir():
    """Ödev 4.6 LoG adımlarını yürütür, görselleri ve raporları kaydeder."""
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

    # 1. 7x7 LoG filtresi
    log_res = log_filtresi(gri)
    log_vis = isaretli_goruntu_uint8(log_res)
    log_abs = mutlak_uint8(log_res)
    log_zc = sifir_gecis(log_res, esik_orani=0.05)

    # 2. Görsel Panel: 1x4 Grid
    panel = panel_olustur(
        [gri, log_vis, log_abs, log_zc],
        [
            "1. Girdi (Orijinal)",
            "2. LoG (Isaretli)",
            "3. LoG (Mutlak)",
            "4. LoG (Sifir Gecis)",
        ],
        (1, 4),
    )

    # Çıktıları kaydet
    cv2.imwrite(str(cikti_klasor / "kamera_log_isaretli.png"), log_vis)
    cv2.imwrite(str(cikti_klasor / "kamera_log_mutlak.png"), log_abs)
    cv2.imwrite(str(cikti_klasor / "kamera_log_ikili.png"), log_zc)
    cv2.imwrite(str(cikti_klasor / "panel_log.png"), panel)

    cv2.imwrite(str(genel_cikti / "kamera_log_isaretli.png"), log_vis)
    cv2.imwrite(str(genel_cikti / "kamera_log_mutlak.png"), log_abs)
    cv2.imwrite(str(genel_cikti / "kamera_log_ikili.png"), log_zc)

    # 3. Karşılaştırma raporlarını üret
    rapor_46_uret(hedef_klasor=dosya_yolu.parent)

    print("Ödev 4.6 (LoG) tamamlandı.")


if __name__ == "__main__":
    calistir()
