from pathlib import Path

import cv2
import numpy as np

klasor = Path(__file__).parent
girdi = klasor.parent / "girdi"

# 3x3 ortalama filtresi: her eleman 1/9
cekirdek = [[1 / 9, 1 / 9, 1 / 9],
            [1 / 9, 1 / 9, 1 / 9],
            [1 / 9, 1 / 9, 1 / 9]]


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(255 ** 2 / mse)


def konvolusyon(gri, cekirdek):
    h, w = gri.shape

    # Kenar pikselleri için görüntüyü 1 piksel yansıtarak genişlet (cv2.blur ile aynı: 1 0 1)
    satirlar = [1] + list(range(h)) + [h - 2]
    sutunlar = [1] + list(range(w)) + [w - 2]
    genis = gri[satirlar][:, sutunlar].tolist()

    sonuc = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        ust, orta, alt = genis[y], genis[y + 1], genis[y + 2]
        satir = sonuc[y]
        for x in range(w):
            # Çekirdeği (y, x) pikselinin üstüne koy, 9 komşuyu ağırlıkla çarpıp topla
            toplam = (ust[x] * cekirdek[0][0] + ust[x + 1] * cekirdek[0][1] + ust[x + 2] * cekirdek[0][2]
                      + orta[x] * cekirdek[1][0] + orta[x + 1] * cekirdek[1][1] + orta[x + 2] * cekirdek[1][2]
                      + alt[x] * cekirdek[2][0] + alt[x + 1] * cekirdek[2][1] + alt[x + 2] * cekirdek[2][2])
            satir[x] = round(toplam)
    return sonuc


for ad in ["dag", "sis"]:
    temiz = cv2.imread(str(girdi / f"{ad}.jpg"), cv2.IMREAD_GRAYSCALE)
    gurultulu = cv2.imread(str(girdi / f"{ad}_gurultulu.png"), cv2.IMREAD_GRAYSCALE)

    sonuc = konvolusyon(gurultulu, cekirdek)

    cv2.imwrite(str(klasor / "cikti" / f"{ad}_hazir_fonksiyonsuz.png"), sonuc)
    print(f"{ad}: PSNR gürültülü {psnr(temiz, gurultulu):.2f} dB -> filtreli {psnr(temiz, sonuc):.2f} dB")
