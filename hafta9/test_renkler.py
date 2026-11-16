"""Küçük, elle denetlenebilir renkler ve gerçek BMP dosyalarıyla kabul testleri."""

import importlib.util
from pathlib import Path
import tempfile

import cv2
import numpy as np
import pytest

KLASOR = Path(__file__).resolve().parent


def yukle(ad, alt, dosya):
    spec = importlib.util.spec_from_file_location(ad, KLASOR / alt / dosya)
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)
    return modul


@pytest.fixture
def renk():
    return yukle("renk", "9.4-renk-uzayi-donusumu", "renk_uzaylari.py")


@pytest.fixture
def ten():
    return yukle("ten", "9.5-ten-rengi", "ten_rengi.py")


def test_hsi_elle_hesaplanan_renkler(renk):
    # RGB: ana/ara renkler, siyah, beyaz, gri ve eşit olmayan kanallar.
    rgb = np.array([[(255, 0, 0), (255, 255, 0), (0, 255, 0),
                     (0, 255, 255), (0, 0, 255), (255, 0, 255),
                     (0, 0, 0), (255, 255, 255), (90, 90, 90),
                     (120, 60, 0)]], dtype=np.uint8)
    beklenen = [[[0, 255, 85], [30, 255, 170], [60, 255, 85],
                 [90, 255, 170], [120, 255, 85], [150, 255, 170],
                 [0, 0, 0], [0, 0, 255], [0, 0, 90], [15, 255, 60]]]
    with np.errstate(all="raise"):
        sonuc = renk.hsi_donustur(rgb[:, :, ::-1])
    np.testing.assert_array_equal(sonuc, np.array(beklenen, np.uint8))


def test_hsi_doygunluk_yuvarlama_ve_sarilma(renk):
    bgr = np.array([[(60, 60, 120), (1, 0, 255)]], np.uint8)
    sonuc = renk.hsi_donustur(bgr)
    # I=80, S=1-60/80=0.25; 0.25*255=63.75 -> 64.
    np.testing.assert_array_equal(sonuc[0, 0], [0, 64, 80])
    assert sonuc[0, 1, 0] == 179


def test_bgr_donusumleri(renk):
    bgr = np.array([[(0, 0, 255), (255, 0, 0), (30, 90, 160)]], np.uint8)
    sonuc = renk.donustur(bgr)
    for ad, kod in [("hsv", cv2.COLOR_BGR2HSV), ("yuv", cv2.COLOR_BGR2YUV),
                    ("lab", cv2.COLOR_BGR2Lab)]:
        np.testing.assert_array_equal(sonuc[ad], cv2.cvtColor(bgr, kod))
    np.testing.assert_array_equal(sonuc["hsv"][0, 0], [0, 255, 255])


def test_rgb_tasma_ve_ten_ornegi(ten):
    # İkinci pikselde R-G=-10; uint8 çıkarma 246 yapıp yanlış kabul edebilir.
    bgr = np.array([[(80, 120, 180), (60, 130, 120), (90, 105, 115),
                     (255, 0, 0), (0, 0, 0)]], np.uint8)
    np.testing.assert_array_equal(ten.rgb_maskesi(bgr), [[True, False, False, False, False]])


def test_ycrcb_sira_ve_dahil_sinirlar(ten):
    ycrcb = np.array([[(120, 133, 77), (120, 173, 127), (120, 132, 100),
                       (120, 174, 100), (120, 150, 76), (120, 150, 128),
                       (120, 100, 150)]], np.uint8)
    bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    np.testing.assert_array_equal(cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb), ycrcb)
    np.testing.assert_array_equal(ten.ycbcr_maskesi(bgr),
                                  [[True, True, False, False, False, False, False]])


def test_hsv_sarilma_ve_alt_esikler(ten):
    hsv = np.array([[(0, 150, 180), (20, 150, 180), (21, 150, 180),
                     (169, 150, 180), (170, 150, 180), (179, 150, 180),
                     (10, 20, 180), (10, 150, 40), (0, 0, 255)]], np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    np.testing.assert_array_equal(ten.hsv_maskesi(bgr),
                                  [[True, True, False, False, True, True, False, False, False]])


@pytest.mark.parametrize("yontem", ["rgb", "hsv", "ycbcr"])
def test_ten_korunur_digerleri_saf_kirmizi(ten, yontem):
    bgr = np.array([[(80, 120, 180), (255, 0, 0), (0, 255, 0), (0, 0, 0)]], np.uint8)
    once = bgr.copy()
    maske = getattr(ten, yontem + "_maskesi")(bgr)
    assert maske.dtype == np.bool_
    np.testing.assert_array_equal(maske, [[True, False, False, False]])
    sonuc = ten.kirmizi_yap(bgr, maske)
    np.testing.assert_array_equal(sonuc[maske], bgr[maske])
    np.testing.assert_array_equal(sonuc[~maske], np.tile([0, 0, 255], (3, 1)))
    np.testing.assert_array_equal(bgr, once)


def test_gercek_bmp_ciktilari(renk, ten):
    with tempfile.TemporaryDirectory(dir=KLASOR, prefix="test_") as gecici:
        kok = Path(gecici)
        girdi = kok / "girdi.bmp"
        bgr = np.array([[(80, 120, 180), (255, 0, 0), (0, 0, 0)],
                        [(0, 0, 255), (255, 255, 255), (17, 89, 201)]], np.uint8)
        assert cv2.imwrite(str(girdi), bgr)
        renk.calistir(girdi, kok / "renk")
        ten.calistir(girdi, kok / "ten")
        for ad, ham in renk.donustur(bgr).items():
            yol = kok / "renk" / (ad + ".bmp")
            assert yol.read_bytes()[:2] == b"BM"
            np.testing.assert_array_equal(cv2.imread(str(yol), cv2.IMREAD_UNCHANGED), ham)
        for ad in ["rgb", "hsv", "ycbcr"]:
            maske = getattr(ten, ad + "_maskesi")(bgr)
            np.testing.assert_array_equal(cv2.imread(str(kok / "ten" / (ad + "_maske.bmp")), -1),
                                          maske.astype(np.uint8) * 255)
            np.testing.assert_array_equal(cv2.imread(str(kok / "ten" / (ad + ".bmp")), -1),
                                          ten.kirmizi_yap(bgr, maske))
        for alt in ["renk", "ten"]:
            assert cv2.imread(str(kok / alt / "karsilastirma.png")) is not None
        rapor = (kok / "ten" / "sayim.txt").read_text(encoding="utf-8")
        for ad in ["RGB", "HSV", "YCbCr"]:
            assert ad in rapor


@pytest.mark.parametrize("ad,alt,dosya", [
    ("renk", "9.4-renk-uzayi-donusumu", "renk_uzaylari.py"),
    ("ten", "9.5-ten-rengi", "ten_rengi.py")])
def test_import_yan_etkisiz(monkeypatch, ad, alt, dosya):
    def yasak(*args, **kwargs):
        pytest.fail("Import sırasında dosya okuma/yazma veya klasör oluşturma")
    monkeypatch.setattr(cv2, "imread", yasak)
    monkeypatch.setattr(cv2, "imwrite", yasak)
    monkeypatch.setattr(Path, "mkdir", yasak)
    yukle(ad, alt, dosya)


@pytest.mark.parametrize("modul", ["renk", "ten"])
def test_hata_denetimi(request, modul, monkeypatch):
    m = request.getfixturevalue(modul)
    with tempfile.TemporaryDirectory(dir=KLASOR, prefix="test_") as gecici:
        kok = Path(gecici)
        with pytest.raises((FileNotFoundError, ValueError)):
            m.calistir(kok / "yok.bmp", kok / "cikti")
        monkeypatch.setattr(cv2, "imwrite", lambda *args: False)
        with pytest.raises(OSError):
            m.kaydet(kok / "a.bmp", np.zeros((1, 1, 3), np.uint8))
