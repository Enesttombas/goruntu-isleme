"""EBLG351 Hafta 4 - Ödev 4.3: 3x3 Sobel Gradyan Operatörü.

Bu betik:
1. Girdi kamera görüntüsüne 5x5 Gauss yumuşatma (sigma=1.0) uygular.
2. 3x3 Sobel operatörü ile gx (dikey kenar), gy (yatay kenar) ve gradyan büyüklüğünü (hypot) hesaplar.
3. Tepe büyüklüğün %20'si eşik değeri ile ikili kenar haritasını üretir.
4. Girdi, gx, gy, büyüklük ve ikili kenar haritasından oluşan paneli kaydeder.
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
    gauss_yumusat,
    gradyan_ikili,
    isaretli_goruntu_uint8,
    mutlak_uint8,
    panel_olustur,
    sobel,
)


def calistir():
    """Ödev 4.3 Sobel adımlarını yürütür ve çıktıları kaydeder."""
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

    # 1. 5x5 Gauss Yumuşatma + Sobel (gx, gy, hypot)
    gx, gy, mag = sobel(gri, yumusat=True)
    gx_vis = isaretli_goruntu_uint8(gx)
    gy_vis = isaretli_goruntu_uint8(gy)
    mag_vis = mutlak_uint8(mag)
    ikili = gradyan_ikili(mag, esik_orani=0.2)

    # 2. Panel: 2x3 Grid
    # Girdi, Gauss Yumuşatılmış, Sobel Gx, Sobel Gy, Büyüklük, İkili Kenar
    yumusak = gauss_yumusat(gri)
    panel = panel_olustur(
        [gri, yumusak, gx_vis, gy_vis, mag_vis, ikili],
        [
            "1. Girdi (Orijinal)",
            "2. Gauss Yumusatma (5x5)",
            "3. Sobel Gx (Dikey Kenar)",
            "4. Sobel Gy (Yatay Kenar)",
            "5. Gradyan Buyuklugu",
            "6. Ikili Kenar (%20 Esik)",
        ],
        (2, 3),
    )

    # Çıktıları kaydet
    cv2.imwrite(str(cikti_klasor / "kamera_sobel_gx.png"), gx_vis)
    cv2.imwrite(str(cikti_klasor / "kamera_sobel_gy.png"), gy_vis)
    cv2.imwrite(str(cikti_klasor / "kamera_sobel_hypot.png"), mag_vis)
    cv2.imwrite(str(cikti_klasor / "kamera_sobel_ikili.png"), ikili)
    cv2.imwrite(str(cikti_klasor / "panel_sobel.png"), panel)

    # Genel çıktı dizini için de güncelle
    cv2.imwrite(str(genel_cikti / "kamera_sobel_gx.png"), gx_vis)
    cv2.imwrite(str(genel_cikti / "kamera_sobel_gy.png"), gy_vis)
    cv2.imwrite(str(genel_cikti / "kamera_sobel_hypot.png"), mag_vis)
    cv2.imwrite(str(genel_cikti / "kamera_sobel_ikili.png"), ikili)

    print("Ödev 4.3 (Sobel) tamamlandı.")


if __name__ == "__main__":
    calistir()
