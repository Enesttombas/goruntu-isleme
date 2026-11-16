"""Küçük verilerle gerçek dosya ve sayısal davranış kontrolleri; 8K üretmez."""

import importlib.util
import struct
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

KLASOR = Path(__file__).resolve().parent


@pytest.fixture
def tmp_path():
    # Geçici test dosyaları da yalnız çalışma klasöründe kalır.
    with tempfile.TemporaryDirectory(prefix=".test-", dir=KLASOR) as yol:
        yield Path(yol)


def yukle(yol):
    tanim = importlib.util.spec_from_file_location(yol.stem, yol)
    modul = importlib.util.module_from_spec(tanim)
    tanim.loader.exec_module(modul)
    return modul


@pytest.fixture
def gri():
    return yukle(KLASOR / "9.2-bmp-gri/bmp_gri.py")


@pytest.fixture
def bolme():
    return yukle(KLASOR / "9.3-8k-piksel-bolme/piksel_bolme.py")


def test_elle_hesaplanmis_tum_kanallar_ve_girdi(bolme):
    resim = np.array([[[0, 1, 2], [3, 254, 255]],
                      [[255, 3, 1], [254, 2, 0]]], dtype=np.uint8)
    once = resim.copy()
    sonuc = np.empty_like(resim)
    bolme.ikiye_bol(resim, sonuc)
    beklenen = np.array([[[0, 0, 1], [1, 127, 127]],
                         [[127, 1, 0], [127, 1, 0]]], dtype=np.uint8)
    np.testing.assert_array_equal(sonuc, beklenen)
    np.testing.assert_array_equal(resim, once)
    assert sonuc.shape == resim.shape and sonuc.dtype == np.uint8


def test_olcum_siniri_ve_ayni_orijinal(bolme, monkeypatch):
    resim = np.full((2, 3, 3), 255, dtype=np.uint8)
    sonuc = np.empty_like(resim)
    gercek_bol = np.floor_divide
    olaylar = []
    saat = iter([0, 10, 20, 40, 50, 80, 90, 130, 140, 190, 200, 260, 270, 340])

    def bol(girdi, sayi, out):
        assert girdi is resim and out is sonuc and sayi == 2
        olaylar.append("bol")
        return gercek_bol(girdi, sayi, out=out)

    def zaman():
        olaylar.append("saat")
        return next(saat)

    monkeypatch.setattr(bolme.np, "floor_divide", bol)
    monkeypatch.setattr(bolme, "perf_counter_ns", zaman)
    assert bolme.olc(resim, sonuc) == [10, 20, 30, 40, 50, 60, 70]
    assert olaylar == ["bol"] + ["saat", "bol", "saat"] * 7
    assert np.all(resim == 255) and np.all(sonuc == 127)


def test_blok_dogrulama_son_satir_ve_son_kanal(bolme):
    resim = np.arange(63, dtype=np.uint8).reshape(7, 3, 3)
    sonuc = resim // 2
    bolme.dogrula(resim, sonuc, blok_satir=2)
    sonuc[-1, -1, -1] += 1
    with pytest.raises(ValueError, match="doğrulama"):
        bolme.dogrula(resim, sonuc, blok_satir=2)


def test_bolme_dosya_akisi_kucuk_boyutta(bolme, tmp_path, monkeypatch):
    monkeypatch.setattr(bolme, "KLASOR", tmp_path)
    monkeypatch.setattr(bolme, "GENISLIK", 11)
    monkeypatch.setattr(bolme, "YUKSEKLIK", 7)
    saat = iter(range(0, 14000, 1000))
    monkeypatch.setattr(bolme, "perf_counter_ns", lambda: next(saat))
    bolme.calistir()
    resim = cv2.imread(str(tmp_path / "girdi/8k.bmp"), cv2.IMREAD_UNCHANGED)
    sonuc = cv2.imread(str(tmp_path / "cikti/8k_yari.bmp"), cv2.IMREAD_UNCHANGED)
    assert resim.shape == sonuc.shape == (7, 11, 3)
    np.testing.assert_array_equal(sonuc, resim // 2)
    rapor = (tmp_path / "cikti/sure.txt").read_text(encoding="utf-8")
    for metin in ("11 x 7", "ölçüm sayısı: 7", "Medyan: 1000 ns = 0.001000 ms",
                  "Minimum: 1000 ns", "Maksimum: 1000 ns", "Python:", "OpenCV:", "NumPy:"):
        assert metin in rapor
    assert cv2.imread(str(tmp_path / "cikti/yan_yana.png")).shape == (540, 1920, 3)


def test_gercek_1000_bmp_akisi(gri, tmp_path, monkeypatch):
    okunan = []
    gercek_oku = cv2.imread

    def oku(yol, bayrak=cv2.IMREAD_COLOR):
        resim = gercek_oku(yol, bayrak)
        okunan.append((Path(yol), None if resim is None else resim.shape))
        return resim

    def gui_yasak(*args):
        pytest.fail("--gosterme sırasında GUI çağrıldı")

    monkeypatch.setattr(gri.cv2, "imread", oku)
    for ad in ("imshow", "waitKey", "destroyAllWindows"):
        monkeypatch.setattr(gri.cv2, ad, gui_yasak)
    gri.calistir(goster=False, kaynak=KLASOR / "girdi/renkli.bmp", klasor=tmp_path)
    renkli_yol = tmp_path / "girdi/renkli_1000.bmp"
    gri_yol = tmp_path / "cikti/gri.bmp"
    assert (renkli_yol, (1000, 1000, 3)) in okunan
    renkli = gercek_oku(str(renkli_yol), cv2.IMREAD_UNCHANGED)
    sonuc = gercek_oku(str(gri_yol), cv2.IMREAD_UNCHANGED)
    assert renkli.shape == (1000, 1000, 3)
    assert sonuc.shape == (1000, 1000) and sonuc.dtype == np.uint8
    np.testing.assert_array_equal(sonuc, cv2.cvtColor(renkli, cv2.COLOR_BGR2GRAY))
    assert renkli_yol.stat().st_size == 3000054
    assert gri_yol.stat().st_size == 1001078
    assert (tmp_path / "cikti/boyutlar.txt").is_file()
    assert gercek_oku(str(tmp_path / "cikti/yan_yana.png")).shape == (1000, 2000, 3)


@pytest.mark.parametrize("kanal,genislik", [(1, 1), (1, 3), (1, 1000), (3, 1), (3, 3), (3, 1000)])
def test_bmp_baslik_palet_hizalama(gri, tmp_path, kanal, genislik):
    yukseklik = 5
    sekil = (yukseklik, genislik) if kanal == 1 else (yukseklik, genislik, 3)
    yol = tmp_path / "ornek.bmp"
    assert cv2.imwrite(str(yol), np.zeros(sekil, dtype=np.uint8))
    veri = yol.read_bytes()
    assert veri[:2] == b"BM"
    baslik = 14 + struct.unpack_from("<I", veri, 14)[0]
    palet = 1024 if kanal == 1 else 0
    satir = ((genislik * kanal + 3) // 4) * 4
    assert struct.unpack_from("<H", veri, 28)[0] == kanal * 8
    assert struct.unpack_from("<I", veri, 10)[0] == baslik + palet
    assert struct.unpack_from("<I", veri, 2)[0] == len(veri)
    assert len(veri) == baslik + palet + satir * yukseklik
    if kanal == 1:
        assert veri[baslik:baslik + palet] == bytes(v for i in range(256) for v in (i, i, i, 0))
    bilgi = gri.bmp_bilgisi(yol)
    assert bilgi["baslik"] == baslik == 54
    assert bilgi["palet"] == palet
    assert bilgi["satir"] == satir
    assert bilgi["boyut"] == len(veri)
    assert bilgi["hesaplanan"] == len(veri)


@pytest.mark.parametrize("modul", ["gri", "bolme"])
def test_dosya_hatalari(request, modul, tmp_path, monkeypatch):
    kod = request.getfixturevalue(modul)
    with pytest.raises(OSError):
        kod.resim_oku(tmp_path / "yok.bmp")
    monkeypatch.setattr(kod.cv2, "imwrite", lambda *args: False)
    with pytest.raises(OSError):
        kod.kaydet(tmp_path / "yazilamaz.bmp", np.zeros((1, 1, 3), np.uint8))
