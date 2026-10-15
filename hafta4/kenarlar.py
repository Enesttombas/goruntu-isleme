"""EBLG351 Hafta 4 - Kenar Tespit Fonksiyonları Kütüphanesi.

Bu modül, kenar tespiti ödevinde (4.1, 4.2 ve 4.3) kullanılan ortak
fonksiyonları, filtre çekirdeklerini ve başarım ölçüm metriklerini içerir.
"""

from typing import Tuple
import cv2
import numpy as np


def gauss_yumusat(gri: np.ndarray) -> np.ndarray:
    """Giriş gri tonlama görüntüsünü 5x5 Gauss çekirdeği ile yumuşatır.

    Girdi:
        gri (np.ndarray): 2 boyutlu gri seviye görüntü (uint8 veya float32).

    Sonuç:
        np.ndarray: Gürültüsü bastırılmış yumuşatılmış görüntü.

    Neden Yapılır:
        Türev tabanlı kenar bulucular (özellikle 2. türev) yüksek frekanslı
        gürültüye karşı çok hassastır. Hafif Gauss ön filtreleme (sigma=1.0),
        kenarları aşırı kaybetmeden sahte kenar oluşumunu engeller.
    """
    return cv2.GaussianBlur(gri, (5, 5), 1.0, borderType=cv2.BORDER_REFLECT_101)


def prewitt(
    gri: np.ndarray, yumusat: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prewitt 1. türev operatörleri ile X, Y gradyanları ve büyüklük hesaplar.

    Girdi:
        gri (np.ndarray): Gri seviye görüntü.
        yumusat (bool): İşlem öncesi Gauss yumuşatma uygulanıp uygulanmayacağı.

    Sonuç:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            gx: Yatay gradyan (işaretli float32).
            gy: Dikey gradyan (işaretli float32).
            hypot: Gradyan büyüklüğü sqrt(gx^2 + gy^2) (float32).

    Neden Yapılır:
        Prewitt, komşu piksellerin basit farkı ile kenar yönünü ve şiddetini
        ölçer. X çekirdeği düşey kenarları, Y çekirdeği yatay kenarları vurgular.
    """
    islenmis = gauss_yumusat(gri) if yumusat else gri
    girdi_f = islenmis.astype(np.float32, copy=False)

    kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)

    gx = cv2.filter2D(girdi_f, cv2.CV_32F, kx, borderType=cv2.BORDER_REFLECT_101)
    gy = cv2.filter2D(girdi_f, cv2.CV_32F, ky, borderType=cv2.BORDER_REFLECT_101)
    hypot = np.hypot(gx, gy).astype(np.float32)

    return gx, gy, hypot


def sobel(
    gri: np.ndarray, yumusat: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sobel operatörü ile X, Y gradyanları ve gradyan büyüklüğünü hesaplar.

    Girdi:
        gri (np.ndarray): Gri seviye görüntü.
        yumusat (bool): İşlem öncesi Gauss yumuşatma uygulanıp uygulanmayacağı.

    Sonuç:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            gx: Yatay gradyan (işaretli float32).
            gy: Dikey gradyan (işaretli float32).
            hypot: Gradyan büyüklüğü sqrt(gx^2 + gy^2) (float32).

    Neden Yapılır:
        Sobel, Prewitt'e benzer ancak merkez piksele 2 ağırlığı vererek
        türev alırken yerel bir Gauss benzeri yumuşatma etkisi sunar;
        bu sayede gürültüye karşı Prewitt'ten daha dirençlidir.
    """
    islenmis = gauss_yumusat(gri) if yumusat else gri
    girdi_f = islenmis.astype(np.float32, copy=False)

    gx = cv2.Sobel(
        girdi_f, cv2.CV_32F, 1, 0, ksize=3, scale=1, borderType=cv2.BORDER_REFLECT_101
    )
    gy = cv2.Sobel(
        girdi_f, cv2.CV_32F, 0, 1, ksize=3, scale=1, borderType=cv2.BORDER_REFLECT_101
    )
    hypot = np.hypot(gx, gy).astype(np.float32)

    return gx, gy, hypot


def laplacian(gri: np.ndarray, yumusat: bool = True) -> np.ndarray:
    """4-komşuluk 3x3 Laplacian 2. türev filtresi uygular.

    Girdi:
        gri (np.ndarray): Gri seviye görüntü.
        yumusat (bool): İşlem öncesi Gauss yumuşatma uygulanıp uygulanmayacağı.

    Sonuç:
        np.ndarray: İşaretli 2. türev matrisi (float32).

    Neden Yapılır:
        Laplacian yön bağımsız (izotropik) ikinci türev filtresidir.
        Görüntüdeki hızlı yoğunluk değişimlerinde sıfır geçişi (zero-crossing)
        noktaları oluşturarak kenar merkezlerini belirler.
    """
    islenmis = gauss_yumusat(gri) if yumusat else gri
    girdi_f = islenmis.astype(np.float32, copy=False)

    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    return cv2.filter2D(girdi_f, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT_101)


def log_cekirdegi() -> np.ndarray:
    """7x7 analitik LoG (Laplacian of Gaussian) çekirdeği üretir (sigma=1.0).

    Formül:
        LoG(x, y) = ((r^2 - 2*sigma^2) / sigma^4) * exp(-r^2 / (2*sigma^2))
        Burada r^2 = x^2 + y^2, x ve y in [-3, 3].

    Sonuç:
        np.ndarray: Ortalaması 0'a çekilmiş ve mutlak toplamı 1'e normalize edilmiş
        7x7 float32 çekirdek. Merkez piksel negatiftir.

    Neden Yapılır:
        Gauss yumuşatma ile Laplacian türevini tek bir analitik konvolüsyon
        çekirdeğinde birleştirir. Düzgün bölgelerde sıfır tepki vermesi için
        ortalama çıkarılır.
    """
    x, y = np.meshgrid(np.arange(-3, 4), np.arange(-3, 4))
    r2 = (x**2 + y**2).astype(np.float64)
    sigma = 1.0
    sigma2 = sigma**2
    sigma4 = sigma**4

    k = ((r2 - 2.0 * sigma2) / sigma4) * np.exp(-r2 / (2.0 * sigma2))
    # Düzgün alanlarda toplamın sıfır olması için ortalamayı çıkar
    k = k - k.mean()
    # Mutlak toplamı 1'e normalize et
    mutlak_toplam = np.sum(np.abs(k))
    if mutlak_toplam > 0:
        k = k / mutlak_toplam

    return k.astype(np.float32)


def log_filtresi(gri: np.ndarray) -> np.ndarray:
    """Giriş görüntüsüne 7x7 LoG filtresi uygular.

    Girdi:
        gri (np.ndarray): Gri seviye görüntü.

    Sonuç:
        np.ndarray: İşaretli float32 2. türev LoG yanıtı.

    Neden Yapılır:
        LoG çekirdeği Gauss yumuşatmayı matematiksel yapısında zaten barındırdığı
        için ikinci bir harici Gauss yumuşatması uygulanmaz.
    """
    girdi_f = gri.astype(np.float32, copy=False)
    return cv2.filter2D(
        girdi_f, cv2.CV_32F, log_cekirdegi(), borderType=cv2.BORDER_REFLECT_101
    )


def canny(gri: np.ndarray, yumusat: bool = True) -> np.ndarray:
    """OpenCV Canny kenar tespit algoritmasını uygular.

    Girdi:
        gri (np.ndarray): Gri seviye görüntü.
        yumusat (bool): İşlem öncesi harici Gauss uygulanıp uygulanmayacağı.

    Sonuç:
        np.ndarray: İkili (0 veya 255) uint8 kenar haritası.

    Neden Yapılır:
        Canny algoritması çok adımlıdır: Gradyan hesaplama, non-maximum
        suppression (yerel olmayan maksimumların bastırılması ile 1 piksel
        inceltme) ve histerezis eşikleme (alt=50, üst=100) adımlarını uygular.
    """
    islenmis = gauss_yumusat(gri) if yumusat else gri
    if islenmis.dtype != np.uint8:
        islenmis = np.clip(islenmis, 0, 255).astype(np.uint8)

    return cv2.Canny(islenmis, 50, 100, apertureSize=3, L2gradient=True)


def gradyan_ikili(hypot: np.ndarray, esik_orani: float = 0.2) -> np.ndarray:
    """Gradyan büyüklük haritasından tepe değerin oranı ile ikili kenar üretir.

    Girdi:
        hypot (np.ndarray): float32 gradyan büyüklüğü.
        esik_orani (float): Maksimum değerin eşikleme katsayısı (varsayılan 0.2).

    Sonuç:
        np.ndarray: 0 veya 255 değerli uint8 ikili kenar görüntüsü.
        Görüntü sabit/düz ise sıfır kenar döner.
    """
    maks = float(hypot.max())
    if maks <= 1e-6:
        return np.zeros(hypot.shape, dtype=np.uint8)
    esik = maks * esik_orani
    return ((hypot >= esik).astype(np.uint8)) * 255


def sifir_gecis(turev2: np.ndarray, esik_orani: float = 0.05) -> np.ndarray:
    """İkinci türev haritasında 8-komşuluklu sıfır geçişlerini bulur.

    Girdi:
        turev2 (np.ndarray): İşaretli 2. türev yanıtı (Laplacian veya LoG).
        esik_orani (float): Dinamik eşik katsayısı (maks mutlak değer oranı).

    Sonuç:
        np.ndarray: 0 veya 255 değerli uint8 kenar haritası.

    Mantık:
        İkinci türevin işareti kenar geçişinde artıdan eksiye döner.
        Bir piksel pozitifken 8 komşusundan biri negatifse ve aralarındaki
        fark dinamik eşiği aşıyorsa o piksel kenar olarak kabul edilir.
        Görüntü sabitse (tüm türevler sıfır) hiç kenar üretilmez.
    """
    maks_abs = float(np.max(np.abs(turev2)))
    if maks_abs <= 1e-6:
        return np.zeros(turev2.shape, dtype=np.uint8)

    esik = maks_abs * esik_orani
    h, w = turev2.shape
    pad = np.pad(turev2, 1, mode="reflect")

    center = pad[1:-1, 1:-1]
    zc_mask = np.zeros((h, w), dtype=bool)

    shifts = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    for dy, dx in shifts:
        neigh = pad[1 + dy : h + 1 + dy, 1 + dx : w + 1 + dx]
        # Pozitif piksel ile negatif komşu arasındaki sert zıtlaşma
        cond = (center > 0) & (neigh < 0) & ((center - neigh) >= esik)
        zc_mask |= cond

    sonuc = np.zeros((h, w), dtype=np.uint8)
    sonuc[zc_mask] = 255
    return sonuc


def isaretli_goruntu_uint8(dizi: np.ndarray) -> np.ndarray:
    """İşaretli türevi negatifleri kırpmadan 0-255 aralığına gererek uint8 yapar.

    Negatifler koyu, 0 orta gri (~128), pozitifler parlak görünür.
    """
    return cv2.normalize(dizi, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def mutlak_uint8(dizi: np.ndarray) -> np.ndarray:
    """İşaretli dizinin mutlak değerini alır ve 0-255 aralığına normalize eder."""
    mutlak = np.abs(dizi)
    maks = float(mutlak.max())
    if maks <= 1e-6:
        return np.zeros(dizi.shape, dtype=np.uint8)
    return np.clip((mutlak / maks) * 255.0, 0, 255).astype(np.uint8)


def kenar_skoru(
    tahmin: np.ndarray, referans: np.ndarray, tolerans: int = 1
) -> Tuple[float, float, float]:
    """Tahmin edilen ikili kenar ile referans kenar haritasını karşılaştırır.

    Girdi:
        tahmin (np.ndarray): Algoritmanın bulduğu ikili kenar (0 / 255).
        referans (np.ndarray): Bilinen zemin gerçeklik (ground truth) ikili kenarı.
        tolerans (int): Konum esneme toleransı (piksel cinsinden, varsayılan 1).

    Sonuç:
        Tuple[float, float, float]: (precision, recall, f1)

    Açıklama:
        1 piksel tolerans için morfolojik genişletme (dilation) kullanılır.
        Tahmin kenarının referansın 1 piksel komşuluğuna düşmesi doğru kabul edilir.
        İkisi de boş ise (kenarsız düz alan) kusursuz kabul edilir: (1.0, 1.0, 1.0).
        Biri boş diğeri dolu veya tamamen ayrık ise: (0.0, 0.0, 0.0).
    """
    tahmin_bin = (tahmin > 0)
    referans_bin = (referans > 0)
    tahmin_sayisi = int(np.count_nonzero(tahmin_bin))
    referans_sayisi = int(np.count_nonzero(referans_bin))

    if tahmin_sayisi == 0 and referans_sayisi == 0:
        return 1.0, 1.0, 1.0
    if tahmin_sayisi == 0 or referans_sayisi == 0:
        return 0.0, 0.0, 0.0

    ksize = 2 * tolerans + 1
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))

    ref_dil = cv2.dilate(referans_bin.astype(np.uint8), k)
    tah_dil = cv2.dilate(tahmin_bin.astype(np.uint8), k)

    tp_prec = int(np.count_nonzero(tahmin_bin & (ref_dil > 0)))
    precision = tp_prec / float(tahmin_sayisi)

    tp_rec = int(np.count_nonzero(referans_bin & (tah_dil > 0)))
    recall = tp_rec / float(referans_sayisi)

    if precision + recall > 0:
        f1 = (2.0 * precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return float(precision), float(recall), float(f1)


def sentetik_veri_uret() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Test ve başarım ölçümü için sentetik test görüntüleri ve referans üretir.

    Sonuç:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            sentetik: Temiz 512x512 geometrik şekiller içeren gri görüntü.
            referans: Geometrik sınırların tam referans kenar haritası (ground truth).
            gurultulu: RNG(42) ile sigma=10 Gauss gürültüsü eklenmiş görüntü.
    """
    boyut = (512, 512)
    sentetik = np.full(boyut, 50, dtype=np.uint8)

    # Daire: merkez (160, 160), r=80, dolgu 200
    cv2.circle(sentetik, (160, 160), 80, 200, thickness=-1)
    # Dikdörtgen: (280, 80) -> (440, 240), dolgu 150
    cv2.rectangle(sentetik, (280, 80), (440, 240), 150, thickness=-1)
    # Çizgi: (80, 360) -> (440, 420), dolgu 230, kalınlık 6
    cv2.line(sentetik, (80, 360), (440, 420), 230, thickness=6)

    # Referans kenar haritası: Temiz ikili nesne sınırlarından türetilir
    # Her nesnenin sınırını geometrik olarak 1 piksel konturla çiziyoruz
    referans = np.zeros(boyut, dtype=np.uint8)
    cv2.circle(referans, (160, 160), 80, 255, thickness=1)
    cv2.rectangle(referans, (280, 80), (440, 240), 255, thickness=1)

    # Kalın çizginin dış sınırları (ikili maskenin morfolojik gradyanı ile)
    cizgi_maskesi = np.zeros(boyut, dtype=np.uint8)
    cv2.line(cizgi_maskesi, (80, 360), (440, 420), 255, thickness=6)
    cizgi_sinir = cv2.morphologyEx(
        cizgi_maskesi, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
    )
    referans = cv2.bitwise_or(referans, cizgi_sinir)

    # Gürültülü varyant: RNG(42) sigma=10
    rng = np.random.default_rng(42)
    gurultu = rng.normal(0.0, 10.0, boyut)
    gurultulu = np.clip(sentetik.astype(np.float32) + gurultu, 0, 255).astype(np.uint8)

    return sentetik, referans, gurultulu


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
            0.55,
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
