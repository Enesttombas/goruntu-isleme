"""Hafta 6 / 6.2: Doğru bulan Hough dönüşümü, tamamen KENDİ yazdığımız kodla.

Doğru denklemi: rho = x*cos(theta) + y*sin(theta)
Her kenar pikseli, her theta için bir rho hesaplar ve oylama matrisinde (rho, theta) hücresine 1 oy verir.
"""

from pathlib import Path

import cv2
import numpy as np

KLASOR = Path(__file__).resolve().parent
GIRDI = KLASOR.parent / "girdi" / "sudoku.png"
CIKTI = KLASOR / "cikti"

THETA_ADIM = 180   # 0..179 derece, 1 derecelik adım
ESIK = 100         # bir doğrunun kabul edilmesi için gereken en az oy


def kaydet(ad: str, goruntu: np.ndarray) -> None:
    if not cv2.imwrite(str(CIKTI / ad), goruntu):
        raise OSError(f"Kaydedilemedi: {ad}")


def sin_cos_tablosu(adim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sin ve cos değerleri baştan bir kez hesaplanır ve bellekte tutulur."""
    thetalar = np.arange(adim) * np.pi / adim
    return thetalar, np.cos(thetalar), np.sin(thetalar)


def hough_oylama(kenar: np.ndarray, cos_t: np.ndarray, sin_t: np.ndarray) -> tuple[np.ndarray, int]:
    """Oylama matrisi: satır = rho (negatifler için kaydırılmış), sütun = theta."""
    yukseklik, genislik = kenar.shape
    kosegen = int(np.ceil(np.hypot(yukseklik, genislik)))
    oylama = np.zeros((2 * kosegen + 1, len(cos_t)), dtype=np.int32)
    theta_indeks = np.arange(len(cos_t))

    ys, xs = np.nonzero(kenar)
    for x, y in zip(xs, ys):
        # Bu piksel, her theta için kendi rho'sunu bulur ve o hücreye 1 oy verir.
        rho = np.rint(x * cos_t + y * sin_t).astype(np.int32)
        oylama[rho + kosegen, theta_indeks] += 1
    return oylama, kosegen


def tepeleri_bul(oylama: np.ndarray, esik: int) -> list[tuple[int, int, int]]:
    """Eşiği geçen ve 3x3 komşuluğunun en büyüğü olan hücreler (aynı doğruyu iki kez çizmemek için)."""
    komsu_max = cv2.dilate(oylama.astype(np.float32), np.ones((3, 3), np.uint8))
    tepe = (oylama >= esik) & (oylama == komsu_max)
    rho_i, theta_i = np.nonzero(tepe)
    sirali = sorted(zip(rho_i, theta_i), key=lambda h: -oylama[h])
    return [(int(r), int(t), int(oylama[r, t])) for r, t in sirali]


def dogru_ciz(resim: np.ndarray, rho: float, cos_v: float, sin_v: float) -> None:
    uzunluk = 2 * max(resim.shape)
    x0, y0 = rho * cos_v, rho * sin_v
    p1 = (int(x0 - uzunluk * sin_v), int(y0 + uzunluk * cos_v))
    p2 = (int(x0 + uzunluk * sin_v), int(y0 - uzunluk * cos_v))
    cv2.line(resim, p1, p2, (255, 0, 255), 2)


def oylama_resmi(oylama: np.ndarray) -> np.ndarray:
    """Oy sayısını 0-255'e ölçekleyip renk haritası ile kaydedilecek resme çevirir.

    Karekök ölçek: az oylu sinüs eğrileri de görünsün, tepeler yine en parlak kalsın.
    """
    olcekli = (255 * np.sqrt(oylama / oylama.max())).astype(np.uint8)
    olcekli = cv2.resize(olcekli, (720, olcekli.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.applyColorMap(olcekli, cv2.COLORMAP_INFERNO)


def calistir() -> None:
    renkli = cv2.imread(str(GIRDI), cv2.IMREAD_COLOR)
    if renkli is None:
        raise FileNotFoundError(f"Resim okunamadı: {GIRDI}")
    CIKTI.mkdir(exist_ok=True)

    gri = cv2.cvtColor(renkli, cv2.COLOR_BGR2GRAY)
    kenar = cv2.medianBlur(cv2.Canny(gri, 50, 150), 3)  # 6.1 ile aynı kenar resmi

    thetalar, cos_t, sin_t = sin_cos_tablosu(THETA_ADIM)
    oylama, kosegen = hough_oylama(kenar, cos_t, sin_t)
    kaydet("oylama_matrisi.png", oylama_resmi(oylama))

    tepeler = tepeleri_bul(oylama, ESIK)
    cizim = renkli.copy()
    for rho_i, theta_i, oy in tepeler:
        dogru_ciz(cizim, rho_i - kosegen, cos_t[theta_i], sin_t[theta_i])
    kaydet("dogrular_kendi.png", cizim)

    print(f"Oylama matrisi boyutu (rho x theta): {oylama.shape}, en yüksek oy: {oylama.max()}")
    print(f"Eşik {ESIK} ile bulunan doğru: {len(tepeler)}")
    for rho_i, theta_i, oy in tepeler:
        print(f"  rho={rho_i - kosegen:5d}  theta={np.degrees(thetalar[theta_i]):5.1f}°  oy={oy}")


if __name__ == "__main__":
    calistir()
