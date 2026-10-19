# 5.4 — Renk uzayı dönüşümleri ve OpenCV ile hata karşılaştırması

BGR görüntü 6 renk uzayına çevrilir: gri, YCrCb, YUV, XYZ, HSV, HLS. Her dönüşümün kendi fonksiyonu `DONUSUMLER` sözlüğünde OpenCV kodu ile eşleşir ve döngüde çağrılır (Python'da fonksiyon pointer karşılığı). Kendi sonucumuz `cv2.cvtColor` çıktısıyla piksel piksel kıyaslanır.

| Dosya | Yöntem | Çıktı |
|---|---|---|
| [opencv_ile.py](opencv_ile.py) | OpenCV `cv2.cvtColor` | `cikti/*_opencv.png` |
| [kendi_fonksiyonum.py](kendi_fonksiyonum.py) | Kendi yazdığımız dönüşümler + hata hesabı | `cikti/*_kendi.png`, [hata.txt](cikti/hata.txt) |

```powershell
python hafta5/5.4-renk-uzayi-donusumu/opencv_ile.py
python hafta5/5.4-renk-uzayi-donusumu/kendi_fonksiyonum.py
```

Sonuç: altı dönüşümde de hata toplamı **0**. Eşleşme için OpenCV'nin 8-bit tamsayı ağırlıkları ve yuvarlaması birebir uygulandı (YCrCb/YUV 14 bit, gri 15 bit, XYZ ve HSV 12 bit; HLS kayan noktayla).
