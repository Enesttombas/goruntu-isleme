"""Hafta 6 / 6.5: Eş merkezli çemberleri bulan Hough dönüşümü, tamamen KENDİ yazdığımız kodla.

Fikir: Eş merkezli çemberlerin hepsinde kenar normali (gradyan yönü) aynı merkezden geçer.
Bu yüzden oylamayı gradyan yönüne göre yapıyoruz: her kenar pikseli, 6.4'teki gibi bütün
açılara değil, yalnız kendi gradyan doğrultusundaki merkezlere (her yarıçap için) oy verir.
Böylece aynı merkezli bütün halkaların oyları 3B oylama matrisinde aynı (a, b) sütununa
düşer; tek çember sütunda tek tepe, eş merkezli küme ise birden çok tepe bırakır.
Önce çember bulup sonra merkezleri karşılaştırmak (post processing) yok.
"""

from pathlib import Path

import cv2
import numpy as np

KLASOR = Path(__file__).resolve().parent
GIRDI = KLASOR.parent / "girdi" / "es_merkezli.png"
CIKTI = KLASOR / "cikti"

YARICAPLAR = np.arange(10, 111)  # aranan yarıçaplar (piksel)
ACI_ADIM = 360                   # gradyan açısı 1 derecelik adımla
ACI_PENCERE = 2                  # gradyan açısındaki hataya tolerans: ±2 derece
HALKA_ESIK = 0.3                 # bir halka için oy / çevre >= 0,3
HALKA_ARALIGI = 6                # aynı sütundaki iki tepe en az 6 px farklı yarıçapta olmalı
MIN_HALKA = 2                    # eş merkezli sayılmak için en az halka sayısı
MIN_MERKEZ_UZAKLIGI = 15

RENKLER = [(255, 0, 255), (0, 200, 0), (0, 165, 255), (255, 255, 0)]


def kaydet(ad: str, goruntu: np.ndarray) -> None:
    if not cv2.imwrite(str(CIKTI / ad), goruntu):
        raise OSError(f"Kaydedilemedi: {ad}")


def sin_cos_tablosu(adim: int) -> tuple[np.ndarray, np.ndarray]:
    """Sin ve cos değerleri baştan bir kez hesaplanır ve bellekte tutulur."""
    thetalar = np.arange(adim) * 2 * np.pi / adim
    return np.cos(thetalar), np.sin(thetalar)


def gradyan_aci_indeksi(gri: np.ndarray) -> np.ndarray:
    """Sobel gradyanının açısı, sin/cos tablosundaki indekse (0..ACI_ADIM-1) çevrilir."""
    gx = cv2.Sobel(gri, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gri, cv2.CV_32F, 0, 1, ksize=3)
    aci = np.arctan2(gy, gx)  # -pi..pi
    return np.rint(aci / (2 * np.pi) * ACI_ADIM).astype(np.int32) % ACI_ADIM


def es_merkezli_oylama(kenar: np.ndarray, aci_indeksi: np.ndarray,
                       cos_t: np.ndarray, sin_t: np.ndarray) -> np.ndarray:
    """3 boyutlu oylama matrisi [b, a, r]: oy yalnız gradyan doğrultusundaki merkezlere."""
    yukseklik, genislik = kenar.shape
    oylama = np.zeros((yukseklik, genislik, len(YARICAPLAR)), dtype=np.int32)
    pencere = np.arange(-ACI_PENCERE, ACI_PENCERE + 1)
    r_indeks = np.arange(len(YARICAPLAR))

    ys, xs = np.nonzero(kenar)
    for x, y in zip(xs, ys):
        acilar = (aci_indeksi[y, x] + pencere) % ACI_ADIM
        # Gradyan halkanın içine de dışına da bakabilir (açık zemin/koyu halka): iki yön (+1, -1).
        yon_cos = np.concatenate([cos_t[acilar], -cos_t[acilar]])
        yon_sin = np.concatenate([sin_t[acilar], -sin_t[acilar]])
        a = np.rint(x - YARICAPLAR[:, None] * yon_cos).astype(np.int32)
        b = np.rint(y - YARICAPLAR[:, None] * yon_sin).astype(np.int32)
        r = np.broadcast_to(r_indeks[:, None], a.shape)
        icerde = (a >= 0) & (a < genislik) & (b >= 0) & (b < yukseklik)
        hucreler = np.unique(np.ravel_multi_index((b[icerde], a[icerde], r[icerde]), oylama.shape))
        oylama.ravel()[hucreler] += 1
    return oylama


def sutun_halkalari(oran_sutunu: np.ndarray) -> list[int]:
    """Bir (a, b) sütunundaki halka tepeleri: eşiği geçen, ±HALKA_ARALIGI içinde en büyük olan r'ler."""
    halkalar = []
    for i in np.argsort(-oran_sutunu):
        if oran_sutunu[i] < HALKA_ESIK:
            break
        if all(abs(i - j) >= HALKA_ARALIGI for j in halkalar):
            halkalar.append(int(i))
    return sorted(halkalar)


def merkezleri_bul(oylama: np.ndarray) -> list[tuple[int, int, list[int], float]]:
    """Merkez skoru = sütundaki eşiği geçen oranların toplamı. Her merkezin halkaları sütundan okunur."""
    oran = oylama / (2 * np.pi * YARICAPLAR)
    skor = np.where(oran >= HALKA_ESIK, oran, 0).sum(axis=2)
    b_i, a_i = np.nonzero(skor >= HALKA_ESIK)
    adaylar = sorted(zip(b_i, a_i), key=lambda h: -skor[h])

    secilen: list[tuple[int, int, list[int], float]] = []
    for b, a in adaylar:
        if any(np.hypot(a - sa, b - sb) < MIN_MERKEZ_UZAKLIGI for sa, sb, _, _ in secilen):
            continue
        halkalar = sutun_halkalari(oran[b, a])
        if halkalar:
            secilen.append((int(a), int(b), [int(YARICAPLAR[i]) for i in halkalar], float(skor[b, a])))
    return secilen


def merkez_skoru_resmi(oylama: np.ndarray) -> np.ndarray:
    """Her (a, b) için sütundaki toplam oy oranı: eş merkezli merkezler en parlak noktalar."""
    toplam = (oylama / (2 * np.pi * YARICAPLAR)).sum(axis=2)
    olcekli = (255 * np.sqrt(toplam / toplam.max())).astype(np.uint8)
    return cv2.applyColorMap(olcekli, cv2.COLORMAP_INFERNO)


def calistir() -> None:
    renkli = cv2.imread(str(GIRDI), cv2.IMREAD_COLOR)
    if renkli is None:
        raise FileNotFoundError(f"Resim okunamadı: {GIRDI}")
    CIKTI.mkdir(exist_ok=True)

    gri = cv2.GaussianBlur(cv2.cvtColor(renkli, cv2.COLOR_BGR2GRAY), (5, 5), 1.2)
    kenar = cv2.Canny(gri, 30, 90)
    kaydet("kenar_canny.png", kenar)

    cos_t, sin_t = sin_cos_tablosu(ACI_ADIM)
    oylama = es_merkezli_oylama(kenar, gradyan_aci_indeksi(gri), cos_t, sin_t)
    kaydet("oylama_merkez_skoru.png", merkez_skoru_resmi(oylama))

    merkezler = merkezleri_bul(oylama)
    kumeler = [m for m in merkezler if len(m[2]) >= MIN_HALKA]
    tekler = [m for m in merkezler if len(m[2]) < MIN_HALKA]
    cizim = renkli.copy()
    for a, b, yaricaplar, _ in tekler:  # tek halkalı: eş merkezli değil, ince gri
        cv2.circle(cizim, (a, b), yaricaplar[0], (128, 128, 128), 1)
    for k, (a, b, yaricaplar, _) in enumerate(kumeler):
        renk = RENKLER[k % len(RENKLER)]
        for r in yaricaplar:
            cv2.circle(cizim, (a, b), r, renk, 2)
        cv2.circle(cizim, (a, b), 3, renk, -1)
    kaydet("cemberler_es_merkezli.png", cizim)

    print(f"Oylama matrisi boyutu (b x a x r): {oylama.shape}, en yüksek oy: {oylama.max()}")
    print(f"Bulunan eş merkezli küme: {len(kumeler)}")
    for a, b, yaricaplar, skor in kumeler:
        print(f"  merkez=({a:3d}, {b:3d})  yarıçaplar={yaricaplar}  skor={skor:.2f}")
    print(f"Tek halkalı (eş merkezli sayılmayan) merkez: {len(tekler)}")
    for a, b, yaricaplar, skor in tekler:
        print(f"  merkez=({a:3d}, {b:3d})  yarıçap={yaricaplar[0]}  skor={skor:.2f}")


if __name__ == "__main__":
    calistir()
