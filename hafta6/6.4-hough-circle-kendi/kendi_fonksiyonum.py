"""Hafta 6 / 6.4: Çember bulan Hough dönüşümü, tamamen KENDİ yazdığımız kodla.

Çember denklemi: x = a + r*cos(theta), y = b + r*sin(theta)
Her kenar pikseli (x, y), her yarıçap r ve her açı theta için olası merkezi
a = x - r*cos(theta), b = y - r*sin(theta) bulur ve 3 boyutlu oylama matrisinde
(b, a, r) hücresine 1 oy verir.
"""

from pathlib import Path

import cv2
import numpy as np

KLASOR = Path(__file__).resolve().parent
GIRDI = KLASOR.parent / "girdi" / "smarties.png"
CIKTI = KLASOR / "cikti"

YARICAPLAR = np.arange(20, 41)   # aranan yarıçaplar (piksel)
ACI_ADIM = 360                   # 1 derecelik adım
ESIK_ORAN = 0.20                 # oy / çevre >= 0,20 : çevrenin en az %20'si kenar olmalı
MIN_MERKEZ_UZAKLIGI = 20         # aynı çemberi iki kez saymamak için


def kaydet(ad: str, goruntu: np.ndarray) -> None:
    if not cv2.imwrite(str(CIKTI / ad), goruntu):
        raise OSError(f"Kaydedilemedi: {ad}")


def sin_cos_tablosu(adim: int) -> tuple[np.ndarray, np.ndarray]:
    """Sin ve cos değerleri baştan bir kez hesaplanır ve bellekte tutulur."""
    thetalar = np.arange(adim) * 2 * np.pi / adim
    return np.cos(thetalar), np.sin(thetalar)


def merkez_kaydirmalari(cos_t: np.ndarray, sin_t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Her (r, theta) için r*cos ve r*sin de bir kez hesaplanır: piksel başına yalnız toplama kalır."""
    r_izgara = YARICAPLAR[:, None]
    da = np.rint(r_izgara * cos_t).astype(np.int32)
    db = np.rint(r_izgara * sin_t).astype(np.int32)
    r_indeks = np.broadcast_to(np.arange(len(YARICAPLAR))[:, None], da.shape)
    return da.ravel(), db.ravel(), r_indeks.ravel()


def hough_cember_oylama(kenar: np.ndarray, cos_t: np.ndarray, sin_t: np.ndarray) -> np.ndarray:
    """3 boyutlu oylama matrisi: [b (satır), a (sütun), r indeksi]."""
    yukseklik, genislik = kenar.shape
    oylama = np.zeros((yukseklik, genislik, len(YARICAPLAR)), dtype=np.int32)
    da, db, r_indeks = merkez_kaydirmalari(cos_t, sin_t)

    ys, xs = np.nonzero(kenar)
    for x, y in zip(xs, ys):
        a = x - da
        b = y - db
        icerde = (a >= 0) & (a < genislik) & (b >= 0) & (b < yukseklik)
        # Aynı piksel aynı hücreye bir kez oy verir (küçük r'de iki açı aynı merkeze yuvarlanabilir).
        hucreler = np.unique(np.ravel_multi_index((b[icerde], a[icerde], r_indeks[icerde]), oylama.shape))
        oylama.ravel()[hucreler] += 1
    return oylama


def cemberleri_sec(oylama: np.ndarray) -> list[tuple[int, int, int, float]]:
    """Eşik: oy / (2*pi*r). Güçlüden zayıfa sıralayıp yakın merkezli tekrarları atar."""
    cevre = 2 * np.pi * YARICAPLAR
    oran = oylama / cevre
    b_i, a_i, r_i = np.nonzero(oran >= ESIK_ORAN)
    adaylar = sorted(zip(b_i, a_i, r_i), key=lambda h: -oran[h])

    secilen: list[tuple[int, int, int, float]] = []
    for b, a, r in adaylar:
        if all(np.hypot(a - sa, b - sb) >= MIN_MERKEZ_UZAKLIGI for sa, sb, _, _ in secilen):
            secilen.append((int(a), int(b), int(YARICAPLAR[r]), float(oran[b, a, r])))
    return secilen


def oylama_resmi(oylama: np.ndarray) -> np.ndarray:
    """3B matrisin r ekseninde en büyük oranı: her (a, b) merkezinin en güçlü yarıçaptaki oyu."""
    oran = (oylama / (2 * np.pi * YARICAPLAR)).max(axis=2)
    olcekli = (255 * np.clip(oran / oran.max(), 0, 1)).astype(np.uint8)
    return cv2.applyColorMap(olcekli, cv2.COLORMAP_INFERNO)


def calistir() -> None:
    renkli = cv2.imread(str(GIRDI), cv2.IMREAD_COLOR)
    if renkli is None:
        raise FileNotFoundError(f"Resim okunamadı: {GIRDI}")
    CIKTI.mkdir(exist_ok=True)

    gri = cv2.cvtColor(renkli, cv2.COLOR_BGR2GRAY)
    kenar = cv2.medianBlur(cv2.Canny(gri, 50, 150), 3)  # 6.3 ile aynı kenar resmi

    cos_t, sin_t = sin_cos_tablosu(ACI_ADIM)
    oylama = hough_cember_oylama(kenar, cos_t, sin_t)
    kaydet("oylama_matrisi_merkezler.png", oylama_resmi(oylama))

    cemberler = cemberleri_sec(oylama)
    cizim = renkli.copy()
    for a, b, r, _ in cemberler:
        cv2.circle(cizim, (a, b), r, (255, 0, 255), 2)
        cv2.circle(cizim, (a, b), 2, (0, 0, 255), 3)
    kaydet("cemberler_kendi.png", cizim)

    print(f"Oylama matrisi boyutu (b x a x r): {oylama.shape}, en yüksek oy: {oylama.max()}")
    print(f"Eşik oy/çevre >= {ESIK_ORAN} ile bulunan çember: {len(cemberler)}")
    for a, b, r, oran in cemberler:
        print(f"  merkez=({a:3d}, {b:3d})  r={r:2d}  oy/çevre={oran:.2f}")


if __name__ == "__main__":
    calistir()
