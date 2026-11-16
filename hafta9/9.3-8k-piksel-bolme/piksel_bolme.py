"""9.3: Bütün BGR değerlerini ikiye tamsayı bölme."""

import platform
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import cv2
import numpy as np

KLASOR = Path(__file__).resolve().parent
GENISLIK, YUKSEKLIK = 7680, 4320


def resim_oku(yol):
    resim = cv2.imread(str(yol), cv2.IMREAD_COLOR)
    if resim is None:
        raise OSError(f"Resim okunamadı: {yol}")
    return resim


def kaydet(yol, resim):
    if not cv2.imwrite(str(yol), resim):
        raise OSError(f"Resim kaydedilemedi: {yol}")


def ikiye_bol(resim, sonuc):
    """B, G ve R dahil bütün elemanları önceden ayrılmış tampona yazar."""
    np.floor_divide(resim, 2, out=sonuc)


def olc(resim, sonuc):
    ikiye_bol(resim, sonuc)  # Bir ısınma; ölçüme dahil değil.
    sureler = []
    for _ in range(7):
        baslangic = perf_counter_ns()
        ikiye_bol(resim, sonuc)
        bitis = perf_counter_ns()
        sureler.append(bitis - baslangic)
    return sureler


def dogrula(resim, sonuc, blok_satir=64):
    if resim.shape != sonuc.shape or resim.dtype != np.uint8 or sonuc.dtype != np.uint8:
        raise ValueError("Bölme doğrulama: boyut veya veri türü uyuşmuyor.")
    if blok_satir <= 0:
        raise ValueError("Blok satır sayısı pozitif olmalı.")
    for bas in range(0, resim.shape[0], blok_satir):
        son = bas + blok_satir
        if not np.array_equal(sonuc[bas:son], resim[bas:son] // 2):
            raise ValueError(f"Bölme doğrulama başarısız: {bas}. satırdan başlayan blok.")


def resim_uret():
    """Tekrarlanabilir, üç kanalı farklı renk geçişleri olan 8K görüntü."""
    resim = np.empty((YUKSEKLIK, GENISLIK, 3), dtype=np.uint8)
    x = np.arange(GENISLIK, dtype=np.uint16)[None, :]
    y = np.arange(YUKSEKLIK, dtype=np.uint16)[:, None]
    resim[:, :, 0] = x % 256
    resim[:, :, 1] = y % 256
    for bas in range(0, YUKSEKLIK, 64):
        resim[bas:bas + 64, :, 2] = (x + y[bas:bas + 64]) % 256
    return resim


def calistir():
    girdi = KLASOR / "girdi"
    cikti = KLASOR / "cikti"
    girdi.mkdir(exist_ok=True)
    cikti.mkdir(exist_ok=True)
    girdi_yol = girdi / "8k.bmp"
    sonuc_yol = cikti / "8k_yari.bmp"
    uretilen = resim_uret()
    kaydet(girdi_yol, uretilen)
    del uretilen
    resim = resim_oku(girdi_yol)
    if resim.shape != (YUKSEKLIK, GENISLIK, 3) or resim.dtype != np.uint8:
        raise ValueError("Diskten okunan görüntü 7680x4320, uint8 ve üç kanallı olmalı.")
    sonuc = np.empty_like(resim)  # Ayırma ve sayfalara ilk erişim ölçüm dışında.
    sonuc.fill(0)
    sureler = olc(resim, sonuc)
    dogrula(resim, sonuc)
    kaydet(sonuc_yol, sonuc)
    # Kaydedilen BMP'nin de bütün kanallarını satır bloklarıyla kontrol et.
    kaydedilen = resim_oku(sonuc_yol)
    dogrula(resim, kaydedilen)
    del kaydedilen
    onizleme = np.hstack([
        cv2.resize(resim, (960, 540), interpolation=cv2.INTER_AREA),
        cv2.resize(sonuc, (960, 540), interpolation=cv2.INTER_AREA),
    ])
    kaydet(cikti / "yan_yana.png", onizleme)
    satirlar = [
        f"Çözünürlük: {GENISLIK} x {YUKSEKLIK}; uint8, 3 kanal (BGR).",
        f"Python: {platform.python_version()}; OpenCV: {cv2.__version__}; NumPy: {np.__version__}",
        "Isınma: 1; ölçüm sayısı: 7; saat: perf_counter_ns.",
        "Her tur aynı orijinalden ayrılmış çıktı tamponuna tamsayı bölme.",
        "Üretme, okuma, yazma, tampon hazırlığı, doğrulama ve önizleme ölçüm dışında.",
    ]
    for sira, ns in enumerate(sureler, 1):
        satirlar.append(f"Ölçüm {sira}: {ns} ns = {ns / 1_000_000:.6f} ms")
    for ad, ns in [("Medyan", median(sureler)), ("Minimum", min(sureler)), ("Maksimum", max(sureler))]:
        satirlar.append(f"{ad}: {ns} ns = {ns / 1_000_000:.6f} ms")
    satirlar.append("Tam x//2 doğrulaması: bütün satırlar ve üç kanal, 64 satırlık bloklarla geçti.")
    (cikti / "sure.txt").write_text("\n".join(satirlar) + "\n", encoding="utf-8")


if __name__ == "__main__":
    calistir()
