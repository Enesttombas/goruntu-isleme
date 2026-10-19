# 5.3 — RGB'den griye: OpenCV ve kendi fonksiyonum

Gri değer göz duyarlılığına göre ağırlıklandırılır: `Y = 0,299·R + 0,587·G + 0,114·B`. Hata, iki gri görüntüde eşleşen piksellerin mutlak farklarının toplamıdır.

| Dosya | Yöntem | Çıktı |
|---|---|---|
| [opencv_ile.py](opencv_ile.py) | OpenCV `cv2.cvtColor(..., COLOR_BGR2GRAY)` | `cikti/gri_opencv.png` |
| [kendi_fonksiyonum.py](kendi_fonksiyonum.py) | Kendi yazdığımız `kendi_gri` + hata hesabı | `cikti/gri_kendi.png`, `cikti/hata.txt` |

```powershell
python hafta5/5.3-gri-donusum-hata/opencv_ile.py
python hafta5/5.3-gri-donusum-hata/kendi_fonksiyonum.py
```

Sonuç: hata toplamı **0**. OpenCV ağırlıkları 15 bit tamsayıya çevirip yuvarlıyor; aynı işlemi yaptığımızda her piksel birebir eşleşiyor.
