# 5.1 — Renk kanallarının histogramı

Renkli fotoğraf okunur, B/G/R kanallarının her biri için 256 seviyeli histogram çıkarılır. Her kanalın toplamı 512 × 512 = 262.144 pikseldir.

| Dosya | Yöntem | Çıktı |
|---|---|---|
| [opencv_ile.py](opencv_ile.py) | OpenCV `cv2.calcHist` | `cikti/histogram_*_opencv.png` |
| [kendi_fonksiyonum.py](kendi_fonksiyonum.py) | Kendi yazdığımız sayma döngüsü | `cikti/histogram_*_kendi.png` |

```powershell
python hafta5/5.1-renk-kanallari-histogram/opencv_ile.py
python hafta5/5.1-renk-kanallari-histogram/kendi_fonksiyonum.py
```

Girdi: [astronot.png](../girdi/astronot.png). OpenCV BGR sırasıyla okur: mavi 0, yeşil 1, kırmızı 2.

## Fotoğraf kaynağı

NASA'nın astronot Eileen Collins fotoğrafı, [scikit-image `data.astronaut` örnek görüntüsü](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut) olarak yayımlanmıştır; kamu malı, bilinen telif kısıtlaması yok. Dosya [scikit-image v0.19.3 deposundan](https://github.com/scikit-image/scikit-image/blob/v0.19.3/skimage/data/astronaut.png) 19 Ekim 2026'da alınmıştır.
