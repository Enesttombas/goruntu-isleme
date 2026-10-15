"""EBLG351 Hafta 4 - Karşılaştırma Raporu Üretici (Markdown ve PDF).

Bu modül:
- hafta4/cikti/olcumler.csv dosyasını okur.
- Dinamik olarak tüm süre ve doğruluk metriklerini CSV verilerinden türetir.
- hafta4/raporlar/KARSILASTIRMA.md dosyasını üretir.
- ReportLab kullanarak tam 3 sayfalık hafta4/raporlar/KARSILASTIRMA.pdf üretir:
    Sayfa 1: Prewitt - Sobel Karşılaştırması (yalnızca Prewitt ve Sobel)
    Sayfa 2: Laplacian, LoG ve Gauss Ön Filtreleme Etkisi
    Sayfa 3: Canny, Performans & Doğruluk Ölçümleri Tablosu ve Sınırlılıklar
- PyMuPDF (fitz) ile PDF'in tam 3 sayfa olduğunu doğrular.
"""

import csv
import os
from pathlib import Path
import sys
import cv2
import numpy as np
import pymupdf
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

dosya_yolu = Path(__file__).resolve()
hafta4_dizin = dosya_yolu.parent
genel_cikti = hafta4_dizin / "cikti"
rapor_klasor = hafta4_dizin / "raporlar"
rapor_klasor.mkdir(parents=True, exist_ok=True)


def olcumleri_oku():
    """olcumler.csv dosyasını okuyup liste olarak döner."""
    csv_yol = genel_cikti / "olcumler.csv"
    if not csv_yol.exists():
        return []
    with open(csv_yol, mode="r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def olcum_bul(olcumler: list, goruntu: str, yontem: str, yumusatma=None):
    """Belirtilen koşullara uyan ölçüm kaydını bulur."""
    for r in olcumler:
        if r.get("goruntu") == goruntu and r.get("yontem") == yontem:
            if yumusatma is None or str(r.get("yumusatma")).lower() == str(yumusatma).lower():
                return r
    return {}


def metrik_analizi(olcumler: list) -> dict:
    """CSV verilerinden dinamik çalışma parametreleri ve metriklerini hesaplar."""
    py_ver = sys.version.split()[0]
    cv_ver = cv2.__version__
    np_ver = np.__version__

    # Kamera Gauss=True ölçümleri
    kam_prewitt_t = olcum_bul(olcumler, "kamera", "Prewitt", True)
    kam_sobel_t = olcum_bul(olcumler, "kamera", "Sobel", True)
    kam_lap_t = olcum_bul(olcumler, "kamera", "Laplacian", True)
    kam_log_t = olcum_bul(olcumler, "kamera", "LoG", True)
    kam_canny_t = olcum_bul(olcumler, "kamera", "Canny", True)

    # Kamera Gauss=False ölçümleri
    kam_prewitt_f = olcum_bul(olcumler, "kamera", "Prewitt", False)
    kam_sobel_f = olcum_bul(olcumler, "kamera", "Sobel", False)
    kam_lap_f = olcum_bul(olcumler, "kamera", "Laplacian", False)
    kam_canny_f = olcum_bul(olcumler, "kamera", "Canny", False)

    # Süre sayısal dönüşümleri (ms)
    kp_t_ms = float(kam_prewitt_t.get("sure_ms", 0.0) or 0.0)
    ks_t_ms = float(kam_sobel_t.get("sure_ms", 0.0) or 0.0)
    kl_t_ms = float(kam_lap_t.get("sure_ms", 0.0) or 0.0)
    klog_t_ms = float(kam_log_t.get("sure_ms", 0.0) or 0.0)
    kc_t_ms = float(kam_canny_t.get("sure_ms", 0.0) or 0.0)

    kp_f_ms = float(kam_prewitt_f.get("sure_ms", 0.0) or 0.0)
    ks_f_ms = float(kam_sobel_f.get("sure_ms", 0.0) or 0.0)
    kl_f_ms = float(kam_lap_f.get("sure_ms", 0.0) or 0.0)
    kc_f_ms = float(kam_canny_f.get("sure_ms", 0.0) or 0.0)

    # Kamera Gauss=Evet kıyaslama listesi
    gauss_true_list = [
        ("Canny", kc_t_ms),
        ("Prewitt", kp_t_ms),
        ("Sobel", ks_t_ms),
        ("Laplacian", kl_t_ms),
        ("LoG", klog_t_ms),
    ]
    gauss_true_list.sort(key=lambda x: x[1])
    en_hizli_yontem, en_hizli_sure = gauss_true_list[0]

    # Sentetik gürültülü veri doğruluk metrikleri
    sg_c_f = olcum_bul(olcumler, "sentetik_gurultulu", "Canny", False)
    sg_c_t = olcum_bul(olcumler, "sentetik_gurultulu", "Canny", True)
    sg_p_f = olcum_bul(olcumler, "sentetik_gurultulu", "Prewitt", False)
    sg_p_t = olcum_bul(olcumler, "sentetik_gurultulu", "Prewitt", True)
    sg_s_f = olcum_bul(olcumler, "sentetik_gurultulu", "Sobel", False)
    sg_s_t = olcum_bul(olcumler, "sentetik_gurultulu", "Sobel", True)
    sg_l_f = olcum_bul(olcumler, "sentetik_gurultulu", "Laplacian", False)
    sg_l_t = olcum_bul(olcumler, "sentetik_gurultulu", "Laplacian", True)
    sg_log_t = olcum_bul(olcumler, "sentetik_gurultulu", "LoG", True)

    return {
        "py_ver": py_ver,
        "cv_ver": cv_ver,
        "np_ver": np_ver,
        "kp_t_ms": kp_t_ms,
        "ks_t_ms": ks_t_ms,
        "kl_t_ms": kl_t_ms,
        "klog_t_ms": klog_t_ms,
        "kc_t_ms": kc_t_ms,
        "kp_f_ms": kp_f_ms,
        "ks_f_ms": ks_f_ms,
        "kl_f_ms": kl_f_ms,
        "kc_f_ms": kc_f_ms,
        "en_hizli_yontem": en_hizli_yontem,
        "en_hizli_sure": en_hizli_sure,
        "gauss_true_sirali": gauss_true_list,
        # Canny
        "canny_gur_f_p": sg_c_f.get("precision", "-"),
        "canny_gur_f_r": sg_c_f.get("recall", "-"),
        "canny_gur_f_f1": sg_c_f.get("f1", "-"),
        "canny_gur_t_p": sg_c_t.get("precision", "-"),
        "canny_gur_t_r": sg_c_t.get("recall", "-"),
        "canny_gur_t_f1": sg_c_t.get("f1", "-"),
        # Prewitt
        "prewitt_gur_f_p": sg_p_f.get("precision", "-"),
        "prewitt_gur_f_f1": sg_p_f.get("f1", "-"),
        "prewitt_gur_t_p": sg_p_t.get("precision", "-"),
        "prewitt_gur_t_f1": sg_p_t.get("f1", "-"),
        # Sobel
        "sobel_gur_f_p": sg_s_f.get("precision", "-"),
        "sobel_gur_f_f1": sg_s_f.get("f1", "-"),
        "sobel_gur_t_p": sg_s_t.get("precision", "-"),
        "sobel_gur_t_f1": sg_s_t.get("f1", "-"),
        # Laplacian
        "lap_gur_f_p": sg_l_f.get("precision", "-"),
        "lap_gur_f_f1": sg_l_f.get("f1", "-"),
        "lap_gur_t_p": sg_l_t.get("precision", "-"),
        "lap_gur_t_f1": sg_l_t.get("f1", "-"),
        # LoG
        "log_gur_t_p": sg_log_t.get("precision", "-"),
        "log_gur_t_f1": sg_log_t.get("f1", "-"),
    }


def markdown_rapor_uret(olcumler: list):
    """KARSILASTIRMA.md raporunu oluşturur."""
    md_yol = rapor_klasor / "KARSILASTIRMA.md"
    m = metrik_analizi(olcumler)

    # Tablo satırlarını hazırla
    tablo_satirlari = []
    for r in olcumler:
        g = r.get("goruntu", "")
        y = r.get("yontem", "")
        ym = "Evet" if r.get("yumusatma") == "True" else "Hayır"
        s = r.get("sure_ms", "")
        p = r.get("precision", "-") if r.get("precision") != "" else "-"
        rec = r.get("recall", "-") if r.get("recall") != "" else "-"
        f1 = r.get("f1", "-") if r.get("f1") != "" else "-"
        tablo_satirlari.append(
            f"| {g} | {y} | {ym} | {s} | {p} | {rec} | {f1} |"
        )
    tablo_metni = "\n".join(tablo_satirlari)

    icerik = f"""# EBLG351 Görüntü İşleme - Hafta 4 Kenar Tespit Raporu

**Tarih:** 2026-10-15
**Öğrenci / Depo:** EBLG351 Hafta 4 Kenar Algoritmaları
**Çalışma Ortamı:** Python {m['py_ver']}, OpenCV {m['cv_ver']}, NumPy {m['np_ver']}, Tek İş Parçacığı (`cv2.setNumThreads(1)`), Görüntü Boyutları: 512×512 piksel
**Ölçüm Protokolü:** 3 ısınma + 20 tekrar `time.perf_counter()` medyanı
**Ölçülen Pipeline:** Ön filtre (varsa Gauss) + çekirdek/gradyan konvolüsyonu + ikili kenar haritası dönüşümü (Prewitt/Sobel için NumPy hypot ve tepe değerin %20'si eşikleme; Laplacian/LoG için 8-komşuluk sıfır geçişi; Canny için cv2.Canny dahili NMS ve çift eşikli histerezis). Dosya disk okuma/yazma (I/O) ve F1 skorlama süre ölçümüne dahil değildir.

---

## 1. Prewitt ve Sobel Karşılaştırması

### Kuramsal Temeller ve Çekirdekler
1. mertebeden yönlü türev operatörleri görüntüdeki ani yoğunluk değişimlerini hesaplar.
- **Prewitt Operatörü:**
  $$\\mathbf{{G_x}} = \\begin{{bmatrix}} -1 & 0 & 1 \\\\ -1 & 0 & 1 \\\\ -1 & 0 & 1 \\end{{bmatrix}}, \\quad \\mathbf{{G_y}} = \\begin{{bmatrix}} -1 & -1 & -1 \\\\ 0 & 0 & 0 \\\\ 1 & 1 & 1 \\end{{bmatrix}}$$
- **Sobel Operatörü:**
  $$\\mathbf{{G_x}} = \\begin{{bmatrix}} -1 & 0 & 1 \\\\ -2 & 0 & 2 \\\\ -1 & 0 & 1 \\end{{bmatrix}}, \\quad \\mathbf{{G_y}} = \\begin{{bmatrix}} -1 & -2 & -1 \\\\ 0 & 0 & 0 \\\\ 1 & 2 & 1 \\end{{bmatrix}}$$
- **Gradyan Büyüklüğü:**
  $$M(x, y) = \\sqrt{{G_x(x,y)^2 + G_y(x,y)^2}}$$

### Farklılıklar ve Nitel Analiz
- **Yumuşatma Etkisi ve Gürültü Davranışı:** Sobel çekirdeği merkez katsayısına $2$ ağırlığını vererek türev eksenine dik doğrultuda yerel bir üçgensel (1-2-1) yumuşatma gerçekleştirir. Prewitt'te ise tüm katsayılar eşit ($1$) ağırlıktadır. Teorik olarak Sobel yerel düzleştirme avantajı sunsa da, bu deneyde kullanılan test sahnelerinde Prewitt ve Sobel benzer kenar sürekliliği ve gürültü davranışı sergilemiştir; bu çalışma tek başına genel bir 'üstünlük' iddiası olarak yorumlanamaz.
- **Türev İşareti ve Görselleştirme:** Türev matrisleri işaretli (float32) üretilmiştir. Görselleştirmede negatif değerleri sıfıra kırpmak yerine doğrusal gerdirme (min-max normalization) veya mutlak değer alınarak tüm gradyan yönleri eksiksiz korunmuştur.
- **İkili Eşikleme Sınırlılığı:** İkili kenar haritası tepe gradyanın %20'si ($T = 0.2 \\times \\max(M)$) ile üretilmiştir. Bu dinamik ölçekleme sabit resimlerde doğru olarak 0 kenar üretir; ancak sahnede çok parlak tekil bir yansıma varsa diğer gerçek kenarlar eşik altında kalabilir.
- **Hesaplama Süreleri:** Kamera görüntüsünde ölçülen tam pipeline süreleri Prewitt için Gauss'suz {m['kp_f_ms']:.2f} ms / Gauss'lu {m['kp_t_ms']:.2f} ms; Sobel için Gauss'suz {m['ks_f_ms']:.2f} ms / Gauss'lu {m['ks_t_ms']:.2f} ms düzeyindedir. Her iki filtre de 3×3 konvolüsyon ve NumPy tabanlı ikili eşikleme adımlarına sahip olduğundan süreleri birbirine çok yakındır.

Görsel: [Prewitt - Sobel Karşılaştırma Paneli](../cikti/karsilastirma_prewitt_sobel.png)

---

## 2. Laplacian, LoG ve Gauss Ön Filtreleme Etkisi

### 2. Mertebe İzotropik Türev ve Sıfır Geçişleri
- **4-Komşuluk 3x3 Laplacian:**
  $$\\nabla^2 = \\begin{{bmatrix}} 0 & 1 & 0 \\\\ 1 & -4 & 1 \\\\ 0 & 1 & 0 \\end{{bmatrix}}$$
  Merkez katsayı negatiftir (-4). Çekirdek matrisi **ayrılabilir (separable) değildir**; doğrudan 2B konvolüsyon ile uygulanır.
- **7x7 Analitik LoG (Laplacian of Gaussian) Çekirdeği ($\\sigma=1.0$):**
  $$LoG(x, y) = \\frac{{r^2 - 2\\sigma^2}}{{\\sigma^4}} \\exp\\left(-\\frac{{r^2}}{{2\\sigma^2}}\\right), \\quad r^2 = x^2 + y^2$$
  Çekirdek ortalaması $0$'a çekilmiş ve mutlak toplamı $1$'e normalize edilmiştir. Merkez katsayısı negatiftir.
- **Sıfır Geçişi (Zero-Crossing):** İkinci türevin işareti kenar sınırında artıdan eksiye geçer. 8-komşulukta işaret değişimi olan ve mutlak farkı dinamik eşiği aşan pikseller kenar olarak işaretlenir. Sıfır geçişine yakın pikseller işaretlenir; kenarlar kalın veya kesintili olabilir, NMS veya kontur inceltme/kapatma uygulanmaz (1 piksel kalınlığında kapalı kontur garantisi yoktur).

### Gauss Ön Filtrelemenin Rolü ve Sınırları
2. mertebeden türev filtresi frekans düzleminde $\\omega^2$ çarpanı sebebiyle yüksek frekanslı gürültüyü büyütür ve sahte sıfır geçişlerinin sayısını aşırı artırabilir. 5x5 Gauss ($\\sigma=1.0$) uygulandığında veya dahili Gauss içeren LoG kullanıldığında bu sahte sıfır geçişleri bastırılır. Ancak Gauss ön filtrelemenin her veri ve eşik değerinde her metriği otomatik olarak iyileştireceği varsayılmamalıdır; nitekim sentetik gürültülü veride kenar yayılması nedeniyle Laplacian ve LoG doğruluk metriklerinde düşüş kaydedilmiştir.

Görsel: [Laplacian - LoG Karşılaştırma Paneli](../cikti/karsilastirma_laplacian_log.png)

---

## 3. Canny, Sayısal Ölçümler ve Yöntem Sınırlılıkları

### Canny Algoritması
Canny algoritması (OpenCV `cv2.Canny`, histerezis eşikleri 50 ve 100, `L2gradient=True`) yönlü gradyan hesabı, Non-Maximum Suppression (NMS ile 1 piksel inceltme) ve çift eşikli histerezis adımlarından oluşur.

### Deneysel Ölçüm Sonuçları (3 Isınma + 20 Ölçüm Medyanı, cv2.setNumThreads(1))

| Görüntü | Yöntem | Gauss Yumuşatma | Süre (ms) | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
{tablo_metni}

### Sonuçların Değerlendirilmesi ve Sınırlılıklar
1. **Çalışma Süreleri:** Aynı kamera görüntüsü ve Gauss=Evet varyantlarında (LoG dahili Gauss içerir), ölçülen pipeline sürelerine göre bu çalıştırmada en hızlı yöntem {m['en_hizli_sure']:.2f} ms ile **{m['en_hizli_yontem']}** olmuştur. Canny'nin hız avantajı OpenCV'nin C++ düzeyindeki optimize kodundan kaynaklanmaktadır. Prewitt ({m['kp_t_ms']:.2f} ms) ve Sobel ({m['ks_t_ms']:.2f} ms) Python/NumPy ikili eşikleme adımlarıyla birbirine çok yakın seyrederken; piksel bazlı 8-komşuluk sıfır geçişi taraması yapan Laplacian ({m['kl_t_ms']:.2f} ms) ve LoG ({m['klog_t_ms']:.2f} ms) en yüksek işlem sürelerine sahiptir.
2. **Doğruluk Analizi ve Gürültü Tepkisi:** Sentetik gürültülü ($\\sigma=10$) görüntüde Canny'ye Gauss ön filtreleme uygulandığında Precision {m['canny_gur_f_p']}'den {m['canny_gur_t_p']}'ye, F1 skoru {m['canny_gur_f_f1']}'den {m['canny_gur_t_f1']}'e yükselmiştir. Ancak Gaussian yumuşatma her yöntemi iyileştirmemiştir: Prewitt'te F1 skoru {m['prewitt_gur_f_f1']}'den {m['prewitt_gur_t_f1']}'ye, Sobel'de {m['sobel_gur_f_f1']}'den {m['sobel_gur_t_f1']}'ye, Laplacian'da ise {m['lap_gur_f_f1']}'den {m['lap_gur_t_f1']}'ye gerilemiştir. Bu düşüş, Gauss filtresinin keskin sentetik sınırları yayarak kenar bölgesini 1 piksel tolerans bandının ötesine taşıması ve eşikleme ile birlikte sınır kalınlaşması oluşturmasıyla ilişkilendirilebilir; nedenleri ayırmak için ek deney gerekir. Canny'de ise NMS adımı kenarları yönlü maksimumda 1 piksele incelttiği için bu yayılma avantaja dönüşmektedir.
3. **Sınırlılıklar:**
   - Precision, recall ve F1, bilinen sentetik referans kenar maskesine göre 1 piksel toleransla morfolojik genişletme kullanılarak ölçülmüştür; gerçek fotoğraflarda zemin gerçeği bulunmadığı için doğruluk uydurulmamıştır.
   - Elde edilen metrik ve süreler bu tek sahne, sentetik geometri ve belirlenen parametre setine (eşik oranları, $\\sigma$) özgüdür; evrensel bir algoritma üstünlük sıralaması olarak genellenemez.

### Kaynaklar ve Dokümantasyon
- OpenCV GaussianBlur: [docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html)
- OpenCV Sobel & Laplacian: [docs.opencv.org/4.x/d2/d2c/tutorial_sobel_derivatives.html](https://docs.opencv.org/4.x/d2/d2c/tutorial_sobel_derivatives.html)
- OpenCV Canny: [docs.opencv.org/4.x/da/d22/tutorial_py_canny.html](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html)
- LoG Kuramı ve Laplacian of Gaussian: [scipy.ndimage.gaussian_laplace](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_laplace.html)
"""
    md_yol.write_text(icerik, encoding="utf-8")
    print(f"Markdown raporu oluşturuldu: {md_yol}")


def pdf_rapor_uret(olcumler: list):
    """Tam 3 sayfalık KARSILASTIRMA.pdf dosyasını üretir."""
    pdf_yol = rapor_klasor / "KARSILASTIRMA.pdf"
    m = metrik_analizi(olcumler)

    # Arial Unicode Font kaydı (Türkçe karakter desteği için)
    font_ad = "Helvetica"
    font_bold = "Helvetica-Bold"

    if os.path.exists("C:/Windows/Fonts/arial.ttf") and os.path.exists(
        "C:/Windows/Fonts/arialbd.ttf"
    ):
        pdfmetrics.registerFont(TTFont("ArialTR", "C:/Windows/Fonts/arial.ttf"))
        pdfmetrics.registerFont(TTFont("ArialTR-Bold", "C:/Windows/Fonts/arialbd.ttf"))
        font_ad = "ArialTR"
        font_bold = "ArialTR-Bold"

    doc = SimpleDocTemplate(
        str(pdf_yol),
        pagesize=A4,
        leftMargin=32,
        rightMargin=32,
        topMargin=28,
        bottomMargin=28,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Baslik",
        fontName=font_bold,
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=4,
    )
    h2_style = ParagraphStyle(
        "AltBaslik",
        fontName=font_bold,
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=4,
        spaceAfter=2,
    )
    body_style = ParagraphStyle(
        "Govde",
        fontName=font_ad,
        fontSize=8.3,
        leading=11.3,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=3,
    )
    bullet_style = ParagraphStyle(
        "Madde",
        fontName=font_ad,
        fontSize=8,
        leading=10.8,
        textColor=colors.HexColor("#2D3748"),
        leftIndent=8,
        spaceAfter=2.5,
    )
    table_cell = ParagraphStyle(
        "TabloHucre",
        fontName=font_ad,
        fontSize=7,
        leading=8.5,
        textColor=colors.black,
        alignment=1,
    )
    table_cell_h = ParagraphStyle(
        "TabloHucreBaslik",
        fontName=font_bold,
        fontSize=7,
        leading=8.5,
        textColor=colors.white,
        alignment=1,
    )

    hikaye = []

    # ==================== SAYFA 1: Prewitt ve Sobel Karşılaştırması ====================
    hikaye.append(Paragraph("EBLG351 Görüntü İşleme — Hafta 4 Kenar Tespit Raporu", title_style))
    hikaye.append(Paragraph("<b>Sayfa 1: Prewitt ve Sobel Kenar Tespiti Karşılaştırması</b>", h2_style))
    hikaye.append(
        Paragraph(
            "Bu bölümde 1. mertebeden yönlü türev operatörleri olan Prewitt ve Sobel algoritmaları "
            "karşılaştırılmıştır. Testler 512×512 boyutundaki standart kamera fotoğrafı üzerinde yürütülmüştür.",
            body_style,
        )
    )

    hikaye.append(Paragraph("1.1 Matematiksel Altyapı ve Çekirdek Matrisleri", h2_style))
    hikaye.append(
        Paragraph(
            "Türev filtreleri, komşu pikseller arasındaki yoğunluk gradyanını ölçer. 3×3 ayrılabilir çekirdekler kullanılır:<br/>"
            "• <b>Prewitt X & Y:</b> Gx = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], Gy = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]<br/>"
            "• <b>Sobel X & Y:</b> Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], Gy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]<br/>"
            "• <b>Büyüklük & Yön:</b> M(x, y) = hypot(Gx, Gy) = sqrt(Gx² + Gy²). Gradyan açısı: θ = arctan2(Gy, Gx).",
            bullet_style,
        )
    )

    hikaye.append(Paragraph("1.2 Temel Farklılıklar ve Nitel Analiz", h2_style))
    hikaye.append(
        Paragraph(
            "• <b>Yumuşatma ve Gürültü Davranışı:</b> Sobel operatörü, merkez satır veya sütun katsayılarına 2 ağırlığını verir. "
            "Bu ağırlıklandırma, türev eksenine dik doğrultuda yerel bir 1-2-1 üçgensel yumuşatma sağlar. Prewitt'te ise katsayılar "
            "düzgün (1, 1, 1) dağılmıştır. Teorik olarak bu fark yerel gürültü yumuşatma avantajı sunsa da, bu deneyde kullanılan test "
            "sahnelerinde iki yöntemin kenar yanıtı ve gürültü direnci birbirine oldukça yakın seyretmiştir; bu deney tek başına genel "
            "bir 'üstünlük' kanıtı olarak yorumlanamaz.",
            bullet_style,
        )
    )
    hikaye.append(
        Paragraph(
            "• <b>İşaretli Türevlerin Korunması:</b> Gradyan hesaplamaları işaretli float32 matrislerinde tutulmuştur. Negatif "
            "türevlerin doğrudan sıfıra kırpılması (clipping) yönlü bilgi kaybına yol açar. Görselleştirmede min-max gerdirme "
            "(negatifler koyu, sıfır orta gri, pozitifler parlak) ve mutlak büyüklük normalizasyonu kullanılmıştır.",
            bullet_style,
        )
    )
    hikaye.append(
        Paragraph(
            "• <b>İkili Eşikleme ve Ölçek Sınırlılığı:</b> İkili kenar haritası tepe büyüklüğün %20'si (T = 0.2 × max(M)) "
            "alınarak elde edilmiştir. Düz (sabit) görüntülerde gradyan 0 olduğundan sıfır kenar üreterek beklentiyi karşılar. "
            "Ancak sahnede aşırı parlak tek bir yansıma varsa maksimum değer şişeceğinden zayıf fakat gerçek nesne kenarları "
            "eşik altında kalıp kaybolabilir. Bu durum bağıl tepe eşiklemenin kuramsal bir sınırlılığıdır.",
            bullet_style,
        )
    )
    hikaye.append(
        Paragraph(
            f"• <b>Hesaplama ve Süre Değerlendirmesi:</b> Her iki operatör de 3×3 konvolüsyon matrisi kullanır. Kamera görüntüsünde "
            f"ölçülen tam pipeline süreleri Prewitt için Gauss'suz {m['kp_f_ms']:.2f} ms / Gauss'lu {m['kp_t_ms']:.2f} ms; Sobel için "
            f"Gauss'suz {m['ks_f_ms']:.2f} ms / Gauss'lu {m['ks_t_ms']:.2f} ms düzeyindedir. Ölçülen süre yalnızca çekirdek konvolüsyonunu "
            f"değil; Gauss ön filtrelemeyi, gradyan matrislerini, NumPy hypot büyüklüğünü ve tepe değerin %20'si ile ikili eşikleme "
            f"adımlarını kapsar (görüntü I/O ve F1 skorlama süre ölçümüne dahil değildir). Her iki filtrenin hesaplama maliyeti birbirine çok yakındır.",
            bullet_style,
        )
    )

    # SAYFA 2 GEÇİŞİ
    hikaye.append(PageBreak())

    # ==================== SAYFA 2: Laplacian, LoG ve Gauss Ön Filtreleme ====================
    hikaye.append(Paragraph("<b>Sayfa 2: Laplacian, LoG ve Gauss Ön Filtreleme Etkisi</b>", h2_style))
    hikaye.append(
        Paragraph(
            "Bu bölümde 2. mertebeden izotropik (yön bağımsız) türev filtreleri ve kenar konumlandırmada kullanılan "
            "sıfır geçişi (zero-crossing) tekniği ele alınmıştır.",
            body_style,
        )
    )

    hikaye.append(Paragraph("2.1 4-Komşuluk Laplacian ve 7x7 Analitik LoG Çekirdeği", h2_style))
    hikaye.append(
        Paragraph(
            "• <b>3×3 Laplacian Çekirdeği:</b> [[0, 1, 0], [1, -4, 1], [0, 1, 0]]. Merkez katsayı negatif (-4), doğrudan "
            "komşular pozitiftir. Çekirdek toplamı 0 olup sabit bölgelerde sıfır yanıt verir. Çekirdek matrisi ayrılabilir "
            "(separable) değildir; doğrudan 2B konvolüsyon ile uygulanır.<br/>"
            "• <b>7×7 Analitik LoG (Laplacian of Gaussian, σ=1.0):</b><br/>"
            "  LoG(x, y) = ((r² - 2σ²) / σ⁴) × exp(-r² / (2σ²)), r² = x² + y² (-3 ≤ x,y ≤ 3).<br/>"
            "  Analitik formül sonrasında çekirdek ortalaması sıfıra çekilmiş ve mutlak katsayı toplamı 1'e normalize edilmiştir. "
            "Merkez piksel negatif tepeye sahiptir.",
            bullet_style,
        )
    )

    hikaye.append(Paragraph("2.2 Sıfır Geçişi (Zero-Crossing) Prensibi", h2_style))
    hikaye.append(
        Paragraph(
            "1. mertebe türev kenarda tepe yaparken, 2. mertebe türev kenarın tam orta noktasında pozitiften negatife geçer "
            "(sıfır noktasını keser). Uygulamada 8-komşuluk kontrol edilerek pozitif piksellerin negatif komşularıyla aralarındaki "
            "mutlak fark dinamik eşiği (maksimum genliğin %5'i) aşıyorsa kenar olarak işaretlenir. Sıfır geçişine yakın pikseller "
            "işaretlenir; kenarlar kalın veya kesintili olabilir, Non-Maximum Suppression (NMS) veya kontur inceltme/kapatma "
            "adımları uygulanmaz (1 piksel kalınlığında kapalı kontur garantisi yoktur).",
            body_style,
        )
    )

    hikaye.append(Paragraph("2.3 Gauss Ön Filtreleme ve Gürültü Etkisi", h2_style))
    hikaye.append(
        Paragraph(
            "İkinci türev filtresi, frekans düzleminde frekansın karesi (ω²) ile çarpar; bu durum yüksek frekanslı sensör "
            "gürültüsünü büyütür ve sahte sıfır geçişlerinin sayısını aşırı artırabilir. 5×5 Gauss (σ=1.0) ön filtresi "
            "veya dahili Gauss yumuşatmasına sahip LoG bu gürültüyü bastırır. Bununla birlikte Gauss ön filtrelemenin her veri seti ve "
            "eşik değerinde tüm metrikleri otomatik olarak iyileştireceği iddia edilemez; nitekim sentetik gürültülü veride kenar yayılması "
            "ve bağıl eşik etkileşimi nedeniyle Laplacian ve LoG doğruluk metriklerinde gerileme görülmektedir.",
            body_style,
        )
    )

    # SAYFA 3 GEÇİŞİ
    hikaye.append(PageBreak())

    # ==================== SAYFA 3: Canny, Sayısal Tablo ve Değerlendirme ====================
    hikaye.append(Paragraph("<b>Sayfa 3: Canny Algoritması, Sayısal Karşılaştırma ve Sınırlılıklar</b>", h2_style))
    hikaye.append(
        Paragraph(
            "Canny algoritması dört aşamadan oluşur: (1) Gauss ön yumuşatma, (2) Sobel gradyan genlik ve yön hesabı, "
            "(3) Gradyan yönünde yerel olmayan maksimumların bastırılması (NMS) ile 1 piksel inceltme, (4) Çift eşikli "
            "(50 / 100) histerezis eşikleme ile kenar sürekliliğinin takibi.",
            body_style,
        )
    )

    hikaye.append(
        Paragraph(
            f"<b>Ölçüm Ortamı & Protokolü:</b> Python {m['py_ver']}, OpenCV {m['cv_ver']}, NumPy {m['np_ver']} | "
            f"512×512 piksel, Tek İş Parçacığı (cv2.setNumThreads(1)) | 3 ısınma + 20 tekrar perf_counter medyanı.<br/>"
            f"<b>Ölçülen Pipeline:</b> Ön filtre (varsa Gauss) + çekirdek konvolüsyonu + ikili kenar haritası dönüşümü "
            f"(Prewitt/Sobel: np.hypot + %20 eşikleme; Laplacian/LoG: 8-komşu sıfır geçişi; Canny: cv2.Canny). Dosya I/O ve F1 hariçtir.",
            bullet_style,
        )
    )

    hikaye.append(Paragraph("3.1 Performans ve Doğruluk Ölçümleri Tablosu", h2_style))

    # 9 Varyant için satırları birleştirerek kompakt ve okunur tablo hazırla
    varyant_sozlugu = {}
    for r in olcumler:
        anahtar = (r.get("yontem", ""), r.get("yumusatma", ""))
        if anahtar not in varyant_sozlugu:
            varyant_sozlugu[anahtar] = {}
        varyant_sozlugu[anahtar][r.get("goruntu", "")] = r

    tablo_data = [
        [
            Paragraph("Yöntem", table_cell_h),
            Paragraph("Gauss", table_cell_h),
            Paragraph("Kamera (ms)", table_cell_h),
            Paragraph("Sentetik (ms)", table_cell_h),
            Paragraph("Sentetik F1", table_cell_h),
            Paragraph("Gürültü (ms)", table_cell_h),
            Paragraph("Gürültü Prec", table_cell_h),
            Paragraph("Gürültü Rec", table_cell_h),
            Paragraph("Gürültü F1", table_cell_h),
        ]
    ]

    for (yontem, yumusat), g_data in varyant_sozlugu.items():
        ym = "Var" if yumusat == "True" else "Yok"
        kam = g_data.get("kamera", {})
        sen = g_data.get("sentetik", {})
        gur = g_data.get("sentetik_gurultulu", {})

        kam_ms = kam.get("sure_ms", "-")
        sen_ms = sen.get("sure_ms", "-")
        sen_f1 = sen.get("f1", "-")
        gur_ms = gur.get("sure_ms", "-")
        gur_p = gur.get("precision", "-")
        gur_r = gur.get("recall", "-")
        gur_f1 = gur.get("f1", "-")

        tablo_data.append(
            [
                Paragraph(yontem, table_cell),
                Paragraph(ym, table_cell),
                Paragraph(kam_ms, table_cell),
                Paragraph(sen_ms, table_cell),
                Paragraph(sen_f1, table_cell),
                Paragraph(gur_ms, table_cell),
                Paragraph(gur_p, table_cell),
                Paragraph(gur_r, table_cell),
                Paragraph(gur_f1, table_cell),
            ]
        )

    t = Table(
        tablo_data,
        colWidths=[
            1.8 * cm,
            1.1 * cm,
            1.8 * cm,
            1.8 * cm,
            1.6 * cm,
            1.8 * cm,
            1.7 * cm,
            1.7 * cm,
            1.7 * cm,
        ],
    )
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
                ("TOPPADDING", (0, 0), (-1, -1), 1.5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
            ]
        )
    )
    hikaye.append(t)
    hikaye.append(Spacer(1, 3))

    hikaye.append(Paragraph("3.2 Deneysel Bulgular ve Yöntem Sınırlılıkları", h2_style))
    hikaye.append(
        Paragraph(
            f"• <b>Süre Analizi:</b> Aynı kamera görüntüsü ve Gauss=Evet varyantlarında (LoG dahili Gauss içerir), ölçülen tam pipeline "
            f"sürelerine göre bu çalıştırmada en hızlı yöntem <b>{m['en_hizli_sure']:.2f} ms ile {m['en_hizli_yontem']}</b> olmuştur. "
            f"Canny'nin hız avantajı OpenCV'nin optimize C++ çekirdeğinden kaynaklanmaktadır. Prewitt ({m['kp_t_ms']:.2f} ms) ve "
            f"Sobel ({m['ks_t_ms']:.2f} ms) NumPy ikili eşikleme adımlarıyla birbirine çok yakın seyrederken; piksel bazlı 8-komşuluk sıfır "
            f"geçişi taraması yapan Laplacian ({m['kl_t_ms']:.2f} ms) ve LoG ({m['klog_t_ms']:.2f} ms) en yüksek işlem sürelerine sahiptir.<br/>"
            f"• <b>Doğruluk ve Gürültü Tepkisi:</b> Sentetik gürültülü (σ=10) veride Canny'ye Gauss ön filtreleme uygulandığında "
            f"Precision {m['canny_gur_f_p']}'den {m['canny_gur_t_p']}'ye, F1 skoru {m['canny_gur_f_f1']}'den {m['canny_gur_t_f1']}'e "
            f"yükselmiştir. Ancak Gaussian yumuşatma her yöntemi iyileştirmemiştir: Prewitt'te F1 skoru {m['prewitt_gur_f_f1']}'den {m['prewitt_gur_t_f1']}'ye, "
            f"Sobel'de {m['sobel_gur_f_f1']}'den {m['sobel_gur_t_f1']}'ye, Laplacian'da ise {m['lap_gur_f_f1']}'den {m['lap_gur_t_f1']}'ye "
            f"gerilemiştir. Bu düşüş, Gauss yumuşatmanın keskin sentetik sınırları yayarak kenar bölgesini 1 piksel tolerans bandının ötesine "
            f"taşıması ve eşikleme ile birlikte sınır kalınlaşması oluşturmasıyla ilişkilendirilebilir; nedenleri ayırmak için ek deney gerekir. Canny'de ise NMS adımı "
            f"kenarları yönlü maksimumda 1 piksele incelttiği için bu yayılma avantaja dönüşmektedir.<br/>"
            f"• <b>Sınırlılıklar:</b> Precision, recall ve F1, bilinen sentetik referans maskesine göre 1 piksel toleransla ölçülmüştür; gerçek fotoğraflarda zemin gerçeği bulunmadığından "
            f"doğruluk uydurulmamıştır. Elde edilen metrik ve süreler bu tek sahne, sentetik geometri ve belirlenen parametre setine özgüdür; "
            f"algoritmalar için evrensel bir üstünlük sıralaması olarak genellenemez.",
            bullet_style,
        )
    )

    hikaye.append(Paragraph("3.3 Resmi Dokümantasyon ve Kaynaklar", h2_style))
    hikaye.append(
        Paragraph(
            "• OpenCV GaussianBlur: docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html<br/>"
            "• OpenCV Sobel & Laplacian: docs.opencv.org/4.x/d2/d2c/tutorial_sobel_derivatives.html<br/>"
            "• OpenCV Canny: docs.opencv.org/4.x/da/d22/tutorial_py_canny.html<br/>"
            "• SciPy LoG Kuramı: docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_laplace.html",
            bullet_style,
        )
    )

    doc.build(hikaye)
    print(f"PDF raporu üretildi: {pdf_yol}")

    # Sayfa sayısını doğrula
    doc_fitz = pymupdf.open(str(pdf_yol))
    sayfa_sayisi = len(doc_fitz)
    print(f"PDF Sayfa Sayısı: {sayfa_sayisi}")
    if sayfa_sayisi != 3:
        raise ValueError(f"HATA: PDF tam 3 sayfa olmalıdır, mevcut sayfa sayısı: {sayfa_sayisi}!")


def calistir():
    olcumler = olcumleri_oku()
    markdown_rapor_uret(olcumler)
    pdf_rapor_uret(olcumler)


if __name__ == "__main__":
    calistir()
