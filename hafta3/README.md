# Hafta 3

Girdi: `girdi/dag.jpg` ve `girdi/sis.jpg`, gri tonlamada okunur.
Her ödevde iki kod var: `hazir_fonksiyonsuz.py` ve `hazir_fonksiyonlu.py`. Çıktılar `cikti/` klasöründe.

| Ödev | Hazır fonksiyonlu | Sonuç |
|---|---|---|
| 1. Min-max doğrusal ölçekleme | `cv2.normalize` | dag 0-237 → 0-255, sis 92-249 → 0-255 |
| 2. Histogram, CDF, histogram eşitleme | `cv2.equalizeHist` | sis std 37.28 → 74.33 |
| 3. CLAHE (clip 2.0, 8x8 karo) | `cv2.createCLAHE` | sis std 37.28 → 45.95 |

Ödev 1 ve 2'de iki kodun çıktısı birebir aynı.
Ödev 3 testi: `cd odev3-clahe && python test_clahe.py`. OpenCV ile en büyük fark 1, farklı piksel %0.2'nin altında
(iki karonun tam ortasındaki piksellerde float yuvarlaması).
