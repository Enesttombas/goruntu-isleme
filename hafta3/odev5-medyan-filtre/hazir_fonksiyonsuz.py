from pathlib import Path

import cv2
import numpy as np

klasor = Path(__file__).parent
girdi = klasor.parent / "girdi"
BOYUT = 5
YARI = BOYUT // 2


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(255 ** 2 / mse)


def medyan_filtre(gri):
    h, w = gri.shape

    # Kenarlar için görüntüyü 2 piksel genişlet; dışarıdaki piksel en yakın kenar pikselini kopyalar
    satirlar = [min(max(i, 0), h - 1) for i in range(-YARI, h + YARI)]
    sutunlar = [min(max(j, 0), w - 1) for j in range(-YARI, w + YARI)]
    genis = gri[satirlar][:, sutunlar]

    # 5x5 pencerenin 25 elemanı: katman k, her pikselin penceresindeki k. değer
    katmanlar = []
    for dy in range(BOYUT):
        for dx in range(BOYUT):
            katmanlar.append(genis[dy:dy + h, dx:dx + w].copy())

    # Kabarcık sıralaması: komşu iki katmanı karşılaştır, küçüğü öne büyüğü arkaya koy.
    # Bu işlem bütün pikseller için aynı anda yapılır.
    n = len(katmanlar)
    for tur in range(n - 1):
        for k in range(n - 1 - tur):
            kucuk = np.minimum(katmanlar[k], katmanlar[k + 1])
            buyuk = np.maximum(katmanlar[k], katmanlar[k + 1])
            katmanlar[k], katmanlar[k + 1] = kucuk, buyuk

    # Sıralı 25 değerin ortadakisi (13.) medyandır
    return katmanlar[n // 2]


for ad in ["dag", "sis"]:
    temiz = cv2.imread(str(girdi / f"{ad}.jpg"), cv2.IMREAD_GRAYSCALE)
    gurultulu = cv2.imread(str(girdi / f"{ad}_tuz_biber.png"), cv2.IMREAD_GRAYSCALE)

    sonuc = medyan_filtre(gurultulu)

    cv2.imwrite(str(klasor / "cikti" / f"{ad}_hazir_fonksiyonsuz.png"), sonuc)
    print(f"{ad}: PSNR gürültülü {psnr(temiz, gurultulu):.2f} dB -> filtreli {psnr(temiz, sonuc):.2f} dB")
