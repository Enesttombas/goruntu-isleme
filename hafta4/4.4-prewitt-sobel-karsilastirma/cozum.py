"""EBLG351 Hafta 4 - Ödev 4.4: Prewitt ve Sobel Karşılaştırması.

Bu betik:
1. Girdi kamera görüntüsünde Prewitt ve Sobel gradyanlarını hesaplar.
2. Karşılaştırma panosunu (karsilastirma_prewitt_sobel.png) oluşturur.
3. Kendi klasöründe karsilastirma.md ve tam 1 sayfalık karsilastirma.pdf üretir.
"""

from pathlib import Path
import sys
import cv2
import numpy as np
import pymupdf

dosya_yolu = Path(__file__).resolve()
hafta4_dizin = dosya_yolu.parent.parent
repo_kok = hafta4_dizin.parent
for d in [str(repo_kok), str(hafta4_dizin)]:
    if d not in sys.path:
        sys.path.insert(0, d)

from hafta4.kenarlar import (
    gradyan_ikili,
    mutlak_uint8,
    panel_olustur,
    prewitt,
    sobel,
)
from hafta4.rapor_uret import rapor_44_uret


def calistir():
    """Ödev 4.4 karşılaştırma paneli ve 1 sayfalık raporları üretir."""
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

    # 1. Prewitt ve Sobel hesaplamaları (Gauss'lu)
    _, _, p_mag = prewitt(gri, yumusat=True)
    p_mag_vis = mutlak_uint8(p_mag)
    p_ikili = gradyan_ikili(p_mag, esik_orani=0.2)

    _, _, s_mag = sobel(gri, yumusat=True)
    s_mag_vis = mutlak_uint8(s_mag)
    s_ikili = gradyan_ikili(s_mag, esik_orani=0.2)

    # 2. Karşılaştırma Paneli
    panel = panel_olustur(
        [gri, p_mag_vis, p_ikili, s_mag_vis, s_ikili],
        [
            "1. Girdi (Kamera)",
            "2. Prewitt Buyukluk",
            "3. Prewitt Ikili (%20)",
            "4. Sobel Buyukluk",
            "5. Sobel Ikili (%20)",
        ],
        (1, 5),
    )

    # Çıktıları kaydet
    cv2.imwrite(str(cikti_klasor / "kamera_prewitt_hypot.png"), p_mag_vis)
    cv2.imwrite(str(cikti_klasor / "kamera_prewitt_ikili.png"), p_ikili)
    cv2.imwrite(str(cikti_klasor / "karsilastirma_prewitt_sobel.png"), panel)

    cv2.imwrite(str(genel_cikti / "kamera_prewitt_hypot.png"), p_mag_vis)
    cv2.imwrite(str(genel_cikti / "kamera_prewitt_ikili.png"), p_ikili)
    cv2.imwrite(str(genel_cikti / "karsilastirma_prewitt_sobel.png"), panel)

    # 3. 1 Sayfalık Rapor Üretimi (karsilastirma.md ve tam 1 sayfa karsilastirma.pdf)
    rapor_44_uret(hedef_klasor=dosya_yolu.parent)

    pdf_yol = dosya_yolu.parent / "karsilastirma.pdf"
    doc = pymupdf.open(str(pdf_yol))
    sayfa_sayisi = len(doc)
    doc.close()
    if sayfa_sayisi != 1:
        raise ValueError(f"Hata: 4.4 karsilastirma.pdf tam 1 sayfa olmalıdır, mevcut: {sayfa_sayisi}")

    print("Ödev 4.4 (Prewitt-Sobel Karşılaştırma) tamamlandı.")


if __name__ == "__main__":
    calistir()
