"""EBLG351 Hafta 4 - Ödev 4.1: Prewitt ve Sobel Kenar Tespiti ve Karşılaştırması.

Bu betik:
1. Girdi kamera görüntüsüne hafif Gauss yumuşatma (sigma=1.0) uygular.
2. Prewitt ve Sobel X/Y gradyanlarını ve büyüklüklerini hesaplar.
3. İşaretli türevleri negatif değerleri kaybetmeden uint8 olarak görselleştirir.
4. Tepe büyüklüğün %20'si eşik değeri ile ikili kenar haritalarını üretir.
5. İki yöntemi görsel ve nitel olarak karşılaştırıp çıktıları kaydeder.
"""

from pathlib import Path
import sys
import cv2
import numpy as np

# Hem repo kökünden hem de kendi dizininden çalışabilmesi için sys.path ayarı
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
    prewitt,
    sobel,
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

        # Başlık bandı
        cv2.rectangle(
            panel, (x0, y0), (x0 + w, y0 + banner_h), (40, 40, 40), thickness=-1
        )
        cv2.putText(
            panel,
            baslik,
            (x0 + 10, y0 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Görsel (BGR formatına çevrilerek yerleştirilir)
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

    # Ön işleme: Gauss yumuşatma
    yumusatilmis = gauss_yumusat(gri)
    cv2.imwrite(str(cikti_klasor / "kamera_gauss.png"), yumusatilmis)
    cv2.imwrite(str(genel_cikti / "kamera_gauss.png"), yumusatilmis)

    # 1. Prewitt Hesaplama (Gauss'lu)
    p_gx, p_gy, p_mag = prewitt(gri, yumusat=True)
    p_gx_vis = isaretli_goruntu_uint8(p_gx)
    p_gy_vis = isaretli_goruntu_uint8(p_gy)
    p_mag_vis = mutlak_uint8(p_mag)
    p_ikili = gradyan_ikili(p_mag, esik_orani=0.2)

    # 2. Sobel Hesaplama (Gauss'lu)
    s_gx, s_gy, s_mag = sobel(gri, yumusat=True)
    s_gx_vis = isaretli_goruntu_uint8(s_gx)
    s_gy_vis = isaretli_goruntu_uint8(s_gy)
    s_mag_vis = mutlak_uint8(s_mag)
    s_ikili = gradyan_ikili(s_mag, esik_orani=0.2)

    # Dosyaları kaydet
    kayitlar = {
        "kamera_prewitt_gx.png": p_gx_vis,
        "kamera_prewitt_gy.png": p_gy_vis,
        "kamera_prewitt_hypot.png": p_mag_vis,
        "kamera_prewitt_ikili.png": p_ikili,
        "kamera_sobel_gx.png": s_gx_vis,
        "kamera_sobel_gy.png": s_gy_vis,
        "kamera_sobel_hypot.png": s_mag_vis,
        "kamera_sobel_ikili.png": s_ikili,
    }

    for ad, gorsel in kayitlar.items():
        cv2.imwrite(str(cikti_klasor / ad), gorsel)
        cv2.imwrite(str(genel_cikti / ad), gorsel)

    # Karşılaştırma Paneli (2 satır x 4 sütun)
    panel = panel_olustur(
        gorseller=[
            p_gx_vis,
            p_gy_vis,
            p_mag_vis,
            p_ikili,
            s_gx_vis,
            s_gy_vis,
            s_mag_vis,
            s_ikili,
        ],
        basliklar=[
            "Prewitt Gx (Yatay)",
            "Prewitt Gy (Dikey)",
            "Prewitt Buyukluk",
            "Prewitt Ikili (Esik 0.2)",
            "Sobel Gx (Yatay)",
            "Sobel Gy (Dikey)",
            "Sobel Buyukluk",
            "Sobel Ikili (Esik 0.2)",
        ],
        grid_boyut=(2, 4),
    )
    cv2.imwrite(str(cikti_klasor / "karsilastirma_prewitt_sobel.png"), panel)
    cv2.imwrite(str(genel_cikti / "karsilastirma_prewitt_sobel.png"), panel)

    print("Ödev 4.1 (Prewitt - Sobel) başarıyla tamamlandı.")
    print(f"Prewitt max gradyan: {p_mag.max():.2f}, Sobel max gradyan: {s_mag.max():.2f}")


if __name__ == "__main__":
    calistir()
