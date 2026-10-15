"""EBLG351 Hafta 4 - Ödev 4.7: OpenCV Canny ve Tüm Yöntemlerin Karşılaştırılması.

Bu betik:
1. OpenCV Canny kenar tespitini (histerezis 50/100, L2gradient) uygular.
2. 9 farklı varyantı (Prewitt True/False, Sobel True/False, Laplacian True/False,
   LoG tek varyant, Canny True/False) 3 test görüntüsünde çalıştırır:
   - kamera (512x512)
   - sentetik (512x512 temiz)
   - sentetik_gurultulu (512x512, RNG=42 sigma=10)
3. Her yöntem için 3 ısınma + 20 ölçüm ile medyan ms süresini ölçer.
4. Sentetik görüntülerde bilinen geometrik referans ile 1 piksel toleranslı
   Precision, Recall ve F1 puanlarını hesaplar.
5. Sonuçları hafta4/cikti/olcumler.csv dosyasına kaydeder.
"""

import csv
from pathlib import Path
import sys
import time
import cv2
import numpy as np

dosya_yolu = Path(__file__).resolve()
hafta4_dizin = dosya_yolu.parent.parent
repo_kok = hafta4_dizin.parent
for d in [str(repo_kok), str(hafta4_dizin)]:
    if d not in sys.path:
        sys.path.insert(0, d)

from hafta4.kenarlar import (
    canny,
    gradyan_ikili,
    kenar_skoru,
    laplacian,
    log_filtresi,
    prewitt,
    sentetik_veri_uret,
    sifir_gecis,
    sobel,
)


def sure_olc(fonksiyon, tekrar: int = 20, isinma: int = 3) -> float:
    """Fonksiyonun medyan çalışma süresini (milisaniye) ölçer."""
    for _ in range(isinma):
        fonksiyon()
    sureler = []
    for _ in range(tekrar):
        t0 = time.perf_counter()
        fonksiyon()
        t1 = time.perf_counter()
        sureler.append((t1 - t0) * 1000.0)
    return float(np.median(sureler))


def calistir():
    """Ödev 4.7 adımlarını yürütür, süre ve doğruluk ölçümlerini kaydeder."""
    cv2.setNumThreads(1)
    cikti_klasor = dosya_yolu.parent / "cikti"
    cikti_klasor.mkdir(parents=True, exist_ok=True)
    genel_cikti = hafta4_dizin / "cikti"
    genel_cikti.mkdir(parents=True, exist_ok=True)

    girdi_yol = hafta4_dizin / "girdi" / "kamera.png"
    if not girdi_yol.exists():
        raise FileNotFoundError(f"Girdi bulunamadı: {girdi_yol}")

    kamera_img = cv2.imread(str(girdi_yol), cv2.IMREAD_GRAYSCALE)
    if kamera_img is None:
        raise ValueError("kamera.png okunamadı!")

    sentetik_img, ref_kenar, gurultulu_img = sentetik_veri_uret()

    # Sentetik ve gürültülü görselleri diske kaydet
    cv2.imwrite(str(genel_cikti / "sentetik.png"), sentetik_img)
    cv2.imwrite(str(genel_cikti / "sentetik_referans.png"), ref_kenar)
    cv2.imwrite(str(genel_cikti / "sentetik_gurultulu.png"), gurultulu_img)

    # 9 Varyant Tanımı
    varyantlar = [
        ("Prewitt", False),
        ("Prewitt", True),
        ("Sobel", False),
        ("Sobel", True),
        ("Laplacian", False),
        ("Laplacian", True),
        ("LoG", True),
        ("Canny", False),
        ("Canny", True),
    ]

    test_goruntuleri = [
        ("kamera", kamera_img, None),
        ("sentetik", sentetik_img, ref_kenar),
        ("sentetik_gurultulu", gurultulu_img, ref_kenar),
    ]

    def yontem_calistir(img, yontem, yumusat):
        if yontem == "Prewitt":
            _, _, mag = prewitt(img, yumusat=yumusat)
            return gradyan_ikili(mag, esik_orani=0.2)
        elif yontem == "Sobel":
            _, _, mag = sobel(img, yumusat=yumusat)
            return gradyan_ikili(mag, esik_orani=0.2)
        elif yontem == "Laplacian":
            lap = laplacian(img, yumusat=yumusat)
            return sifir_gecis(lap, esik_orani=0.05)
        elif yontem == "LoG":
            log_res = log_filtresi(img)
            return sifir_gecis(log_res, esik_orani=0.05)
        elif yontem == "Canny":
            return canny(img, yumusat=yumusat)
        else:
            raise ValueError(f"Bilinmeyen yöntem: {yontem}")

    satirlar = []

    for g_ad, img, ref in test_goruntuleri:
        for yontem, yumusat in varyantlar:
            # Süre ölçümü (Gauss adımı dahil, disk IO hariç)
            ms = sure_olc(lambda: yontem_calistir(img, yontem, yumusat))

            # Kenar haritası üretimi
            tahmin = yontem_calistir(img, yontem, yumusat)

            # Doğruluk hesaplama (yalnızca sentetik referanslı görüntüler için)
            if ref is not None:
                p, r, f1 = kenar_skoru(tahmin, ref, tolerans=1)
                p_str = f"{p:.4f}"
                r_str = f"{r:.4f}"
                f1_str = f"{f1:.4f}"
            else:
                p_str = ""
                r_str = ""
                f1_str = ""

            satirlar.append(
                {
                    "goruntu": g_ad,
                    "yontem": yontem,
                    "yumusatma": str(yumusat),
                    "sure_ms": f"{ms:.3f}",
                    "precision": p_str,
                    "recall": r_str,
                    "f1": f1_str,
                }
            )

    # CSV olarak kaydet
    csv_yol = genel_cikti / "olcumler.csv"
    alanlar = ["goruntu", "yontem", "yumusatma", "sure_ms", "precision", "recall", "f1"]
    with open(csv_yol, "w", newline="", encoding="utf-8") as f:
        yazici = csv.DictWriter(f, fieldnames=alanlar)
        yazici.writeheader()
        yazici.writerows(satirlar)

    # Canny görsel çıktılarını kaydet
    kamera_canny_gauss = canny(kamera_img, yumusat=True)
    kamera_canny_nogauss = canny(kamera_img, yumusat=False)

    cv2.imwrite(str(cikti_klasor / "kamera_canny_gauss.png"), kamera_canny_gauss)
    cv2.imwrite(str(cikti_klasor / "kamera_canny_nogauss.png"), kamera_canny_nogauss)
    cv2.imwrite(str(genel_cikti / "kamera_canny_gauss.png"), kamera_canny_gauss)
    cv2.imwrite(str(genel_cikti / "kamera_canny_nogauss.png"), kamera_canny_nogauss)

    # Karşılaştırma Paneli (Tüm yöntemlerin kamera ve gürültülü sentetik üzerindeki ikili çıktıları)
    panel_gorseller = [
        kamera_img,
        gradyan_ikili(prewitt(kamera_img, True)[2]),
        gradyan_ikili(sobel(kamera_img, True)[2]),
        sifir_gecis(laplacian(kamera_img, True)),
        sifir_gecis(log_filtresi(kamera_img)),
        kamera_canny_gauss,
        gurultulu_img,
        gradyan_ikili(prewitt(gurultulu_img, True)[2]),
        gradyan_ikili(sobel(gurultulu_img, True)[2]),
        sifir_gecis(laplacian(gurultulu_img, True)),
        sifir_gecis(log_filtresi(gurultulu_img)),
        canny(gurultulu_img, yumusat=True),
    ]
    basliklar = [
        "Kamera Orijinal",
        "Prewitt (Kamera)",
        "Sobel (Kamera)",
        "Laplacian (Kamera)",
        "LoG (Kamera)",
        "Canny (Kamera)",
        "Gurultulu Sentetik",
        "Prewitt (Gurultulu)",
        "Sobel (Gurultulu)",
        "Laplacian (Gurultulu)",
        "LoG (Gurultulu)",
        "Canny (Gurultulu)",
    ]

    h, w = panel_gorseller[0].shape[:2]
    panel = np.full((2 * (h + 30), 6 * w, 3), 245, dtype=np.uint8)
    for idx, (img, b) in enumerate(zip(panel_gorseller, basliklar)):
        r = idx // 6
        c = idx % 6
        y0 = r * (h + 30)
        x0 = c * w
        cv2.rectangle(panel, (x0, y0), (x0 + w, y0 + 30), (40, 40, 40), -1)
        cv2.putText(
            panel,
            b,
            (x0 + 8, y0 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        if len(img.shape) == 2:
            img_b = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_b = img
        panel[y0 + 30 : y0 + h + 30, x0 : x0 + w] = img_b

    cv2.imwrite(str(cikti_klasor / "karsilastirma_tum_yontemler.png"), panel)
    cv2.imwrite(str(genel_cikti / "karsilastirma_tum_yontemler.png"), panel)

    print("Ödev 4.7 (Canny ve Ölçümler) başarıyla tamamlandı.")
    print(f"Toplam {len(satirlar)} ölçüm satırı hafta4/cikti/olcumler.csv'ye yazıldı.")


if __name__ == "__main__":
    calistir()
