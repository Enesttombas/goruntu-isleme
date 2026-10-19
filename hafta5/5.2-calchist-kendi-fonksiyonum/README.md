# 5.2 — calcHist fonksiyonunu kendimiz yazma

`kendi_calchist` 8-bit tek kanallı görüntüyü piksel piksel gezer, her yoğunluğun sayacını bir artırır; B, G, R kanallarında ayrı ayrı çalışır.

| Dosya | Yöntem | Çıktı |
|---|---|---|
| [kendi_fonksiyonum.py](kendi_fonksiyonum.py) | Kendi yazdığımız `kendi_calchist` | `cikti/histogram_*_kendi.png` |
| [opencv_ile.py](opencv_ile.py) | Karşılaştırma için OpenCV `cv2.calcHist` | `cikti/histogram_*_opencv.png` |

```powershell
python hafta5/5.2-calchist-kendi-fonksiyonum/kendi_fonksiyonum.py
python hafta5/5.2-calchist-kendi-fonksiyonum/opencv_ile.py
```
