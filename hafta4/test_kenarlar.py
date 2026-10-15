import sys
from pathlib import Path

# Repo kokunu import yoluna ekle
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pytest
from hafta4.kenarlar import (
    canny,
    gauss_yumusat,
    gradyan_ikili,
    kenar_skoru,
    laplacian,
    log_cekirdegi,
    log_filtresi,
    prewitt,
    sifir_gecis,
    sobel,
)


def test_log_cekirdegi_ozellikleri():
    """7x7 LoG çekirdeğinin analitik özelliklerini doğrular."""
    k = log_cekirdegi()
    assert k.shape == (7, 7), "LoG çekirdeği 7x7 boyutunda olmalıdır."
    assert k.dtype == np.float32, "LoG float32 olmalıdır."
    assert k[3, 3] < 0, "LoG merkez pikseli negatif olmalıdır."
    assert np.isclose(float(k.sum()), 0.0, atol=1e-5), "LoG toplamı sıfır olmalıdır."
    assert np.isclose(
        float(np.sum(np.abs(k))), 1.0, atol=1e-5
    ), "LoG mutlak toplamı 1 olmalıdır."


def test_sabit_resim_sifir_kenar():
    """Sabit (düz) bir görüntünün hiçbir filtrede kenar üretmediğini test eder."""
    sabit = np.full((64, 64), 120, dtype=np.uint8)

    # Prewitt ve Sobel
    _, _, p_hypot = prewitt(sabit, yumusat=False)
    _, _, s_hypot = sobel(sabit, yumusat=False)
    assert np.all(p_hypot == 0), "Sabit resimde Prewitt gradyanı sıfır olmalıdır."
    assert np.all(s_hypot == 0), "Sabit resimde Sobel gradyanı sıfır olmalıdır."
    assert np.all(gradyan_ikili(p_hypot) == 0), "Prewitt ikili kenar boş olmalıdır."
    assert np.all(gradyan_ikili(s_hypot) == 0), "Sobel ikili kenar boş olmalıdır."

    # Laplacian ve LoG
    lap = laplacian(sabit, yumusat=False)
    log_res = log_filtresi(sabit)
    assert np.allclose(lap, 0, atol=1e-4), "Sabit resimde Laplacian sıfıra yakın olmalıdır."
    assert np.allclose(log_res, 0, atol=1e-4), "Sabit resimde LoG sıfıra yakın olmalıdır."
    assert np.all(sifir_gecis(lap) == 0), "Laplacian sıfır geçişi boş olmalıdır."
    assert np.all(sifir_gecis(log_res) == 0), "LoG sıfır geçişi boş olmalıdır."

    # Canny
    cn = canny(sabit, yumusat=False)
    assert np.all(cn == 0), "Sabit resimde Canny sıfır olmalıdır."


def test_basamak_yonu_ve_signed_float():
    """Dikey ve yatay basamaklarda türev yönlerini ve işaretli float yapısını test eder."""
    # Soldan sağa aydınlanan dikey basamak (50 -> 200)
    dikey = np.full((32, 32), 50, dtype=np.uint8)
    dikey[:, 16:] = 200

    px, py, _ = prewitt(dikey, yumusat=False)
    sx, sy, _ = sobel(dikey, yumusat=False)

    # Türevler signed float olmalı
    assert issubclass(px.dtype.type, np.floating)
    assert issubclass(sx.dtype.type, np.floating)

    # Dikey kenarda x türevi pozitif tepe yapmalı, y türevi sıfır kalmalı
    assert px[16, 16] > 0
    assert np.all(py[1:-1, 1:-1] == 0)
    assert sx[16, 16] > 0
    assert np.all(sy[1:-1, 1:-1] == 0)

    # Ters basamak (200 -> 50) ile negatif işaret testi
    ters_dikey = np.full((32, 32), 200, dtype=np.uint8)
    ters_dikey[:, 16:] = 50
    tpx, _, _ = prewitt(ters_dikey, yumusat=False)
    assert tpx[16, 16] < 0, "Düşen basamakta gx negatif türev üretmelidir."

    # Laplacian da negatif ve pozitif değerleri barındırmalı
    lap = laplacian(dikey, yumusat=False)
    assert issubclass(lap.dtype.type, np.floating)
    assert np.any(lap > 0) and np.any(lap < 0), "Laplacian işaretli iki yönlü olmalıdır."


def test_girdiye_yazmama():
    """Fonksiyonların girdi matrisini yerinde (in-place) değiştirmediğini doğrular."""
    orijinal = np.full((32, 32), 100, dtype=np.uint8)
    kopya = orijinal.copy()

    _ = gauss_yumusat(orijinal)
    assert np.array_equal(orijinal, kopya)

    _ = prewitt(orijinal)
    assert np.array_equal(orijinal, kopya)

    _ = sobel(orijinal)
    assert np.array_equal(orijinal, kopya)

    _ = laplacian(orijinal)
    assert np.array_equal(orijinal, kopya)

    _ = log_filtresi(orijinal)
    assert np.array_equal(orijinal, kopya)

    _ = canny(orijinal)
    assert np.array_equal(orijinal, kopya)


def test_kenar_skoru_metrikleri():
    """Kenar skoru fonksiyonunun tolerans, boşluk ve kayma durumlarını test eder."""
    # Boş / Boş mükemmel
    bos1 = np.zeros((20, 20), dtype=np.uint8)
    bos2 = np.zeros((20, 20), dtype=np.uint8)
    assert kenar_skoru(bos1, bos2) == (1.0, 1.0, 1.0)

    # Biri boş diğeri dolu -> 0.0
    dolu = np.zeros((20, 20), dtype=np.uint8)
    dolu[10, :] = 255
    assert kenar_skoru(bos1, dolu) == (0.0, 0.0, 0.0)
    assert kenar_skoru(dolu, bos1) == (0.0, 0.0, 0.0)

    # Tam eşleşme -> 1.0
    assert kenar_skoru(dolu, dolu) == (1.0, 1.0, 1.0)

    # 1 piksel kayma: tolerans=1 iken mükemmel sayılır
    kaymis_1px = np.zeros((20, 20), dtype=np.uint8)
    kaymis_1px[9, :] = 255
    prec, rec, f1 = kenar_skoru(kaymis_1px, dolu, tolerans=1)
    assert np.isclose(prec, 1.0) and np.isclose(rec, 1.0) and np.isclose(f1, 1.0)

    # 2 piksel kayma: tolerans=1 dışına çıkar -> 0.0
    kaymis_2px = np.zeros((20, 20), dtype=np.uint8)
    kaymis_2px[8, :] = 255
    p2, r2, f2 = kenar_skoru(kaymis_2px, dolu, tolerans=1)
    assert p2 == 0.0 and r2 == 0.0 and f2 == 0.0
