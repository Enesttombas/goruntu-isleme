from pathlib import Path

import cv2
import numpy as np


def clahe(gri, clip_limit=2.0, karo=8):
    h, w = gri.shape

    # 1) Görüntü karo sayısına tam bölünmüyorsa alt ve sağ kenarı yansıtarak büyüt
    if h % karo or w % karo:
        satirlar = np.arange(h + karo - h % karo)
        sutunlar = np.arange(w + karo - w % karo)
        satirlar = np.where(satirlar < h, satirlar, 2 * (h - 1) - satirlar)
        sutunlar = np.where(sutunlar < w, sutunlar, 2 * (w - 1) - sutunlar)
        genis = gri[satirlar][:, sutunlar]
    else:
        genis = gri

    kh = genis.shape[0] // karo
    kw = genis.shape[1] // karo
    alan = kh * kw
    limit = max(int(clip_limit * alan / 256), 1)
    olcek = np.float32(255) / np.float32(alan)

    # 2) Her karo için: histogram -> kırp -> fazlayı dağıt -> CDF tablosu
    tablolar = np.zeros((karo, karo, 256), dtype=np.float32)
    for i in range(karo):
        for j in range(karo):
            parca = genis[i * kh:(i + 1) * kh, j * kw:(j + 1) * kw]

            hist = [0] * 256
            for deger in parca.ravel().tolist():
                hist[deger] += 1

            fazla = 0
            for k in range(256):
                if hist[k] > limit:
                    fazla += hist[k] - limit
                    hist[k] = limit

            pay, kalan = divmod(fazla, 256)
            for k in range(256):
                hist[k] += pay
            if kalan > 0:
                adim = max(256 // kalan, 1)
                k = 0
                while k < 256 and kalan > 0:
                    hist[k] += 1
                    k += adim
                    kalan -= 1

            toplam = 0
            for k in range(256):
                toplam += hist[k]
                tablolar[i, j, k] = min(255, round(float(np.float32(toplam) * olcek)))

    # 3) Her piksel çevresindeki 4 karonun tablosundan çift doğrusal ara değer alır
    y = np.arange(h, dtype=np.float32) * (np.float32(1) / np.float32(kh)) - np.float32(0.5)
    x = np.arange(w, dtype=np.float32) * (np.float32(1) / np.float32(kw)) - np.float32(0.5)
    y1 = np.floor(y).astype(int)
    x1 = np.floor(x).astype(int)
    ya = (y - y1)[:, None]
    xa = (x - x1)[None, :]
    y2 = np.minimum(y1 + 1, karo - 1)[:, None]
    x2 = np.minimum(x1 + 1, karo - 1)[None, :]
    y1 = np.maximum(y1, 0)[:, None]
    x1 = np.maximum(x1, 0)[None, :]

    ust = tablolar[y1, x1, gri] * (1 - xa) + tablolar[y1, x2, gri] * xa
    alt = tablolar[y2, x1, gri] * (1 - xa) + tablolar[y2, x2, gri] * xa
    sonuc = ust * (1 - ya) + alt * ya
    return np.clip(np.rint(sonuc), 0, 255).astype(np.uint8)


if __name__ == "__main__":
    klasor = Path(__file__).parent
    for ad in ["dag", "sis"]:
        gri = cv2.imread(str(klasor.parent / "girdi" / f"{ad}.jpg"), cv2.IMREAD_GRAYSCALE)

        sonuc = clahe(gri)

        cv2.imwrite(str(klasor / "cikti" / f"{ad}_hazir_fonksiyonsuz.png"), sonuc)
        print(f"{ad}: std {gri.std():.2f} -> {sonuc.std():.2f}")
