"""Hafta 2 dönüşümleri ve thread sonuçları için küçük sentetik testler."""
import numpy as np

try:
    from .histogram_thread import histogramlari_birlestir, parca_histogramlari, tum_histogram
    from .nicemle_6bit import nicemle_6bit
    from .parca_thread import parcalari_isle
    from .parlaklik_donusum import parlaklik_donusumleri
except ImportError:
    from histogram_thread import histogramlari_birlestir, parca_histogramlari, tum_histogram
    from nicemle_6bit import nicemle_6bit
    from parca_thread import parcalari_isle
    from parlaklik_donusum import parlaklik_donusumleri


def test_6_bit_nicemleme_en_fazla_64_seviye_ve_dortun_kati():
    gri = np.arange(256, dtype=np.uint8).reshape(16, 16)
    sonuc = nicemle_6bit(gri)
    assert np.unique(sonuc).size <= 64
    assert np.all(sonuc % 4 == 0)


def test_parca_sirali_ve_thread_sonuclari_esit():
    gri = np.arange(35, dtype=np.uint8).reshape(5, 7) * 7
    assert np.array_equal(parcalari_isle(gri, False), parcalari_isle(gri, True))


def test_parca_histogramlari_toplami_tam_goruntuya_esit():
    gri = np.arange(35, dtype=np.uint8).reshape(5, 7) * 7
    parcalar = parca_histogramlari(gri, paralel=True)
    toplam = histogramlari_birlestir(parcalar)
    assert np.array_equal(toplam, tum_histogram(gri))
    assert int(toplam.sum()) == gri.size


def test_parlaklik_donusumu_degerleri_ve_kirpma():
    gri = np.array([[0, 250, 255]], dtype=np.uint8)
    carp, topla, birlikte = parlaklik_donusumleri(gri)
    assert birlikte[0, 0] == 20
    assert birlikte[0, 2] == 211
    assert topla[0, 1] == 255
    assert carp[0, 2] == 191
