# 5.2 — Kendi calcHist fonksiyonu

`kendi_calchist` 8-bit tek kanallı görüntüyü piksel piksel gezer, her yoğunluğun sayacını bir artırır. Fonksiyon B, G ve R kanalları için ayrı ayrı çalıştırılır ve sonuç `cv2.calcHist` ile karşılaştırılır; üç kanalda da birebir aynıdır.

```powershell
python hafta5/5.2-calchist-kendi-fonksiyonum/cozum.py
```

Çıktı: [kırmızı](cikti/histogram_kirmizi.png), [yeşil](cikti/histogram_yesil.png), [mavi](cikti/histogram_mavi.png).
