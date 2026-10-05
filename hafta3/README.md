# Hafta 3

Girdi: `girdi/dag.jpg` ve `girdi/sis.jpg`, gri tonlamada okunur.
Her ödevde iki kod var: `hazir_fonksiyonsuz.py` ve `hazir_fonksiyonlu.py`. Çıktılar `cikti/` klasöründe.

| Ödev | Hazır fonksiyonlu | Sonuç |
|---|---|---|
| 1. Min-max doğrusal ölçekleme | `cv2.normalize` | dag 0-237 → 0-255, sis 92-249 → 0-255 |
| 2. Histogram, CDF, histogram eşitleme | `cv2.equalizeHist` | sis std 37.28 → 74.33 |
| 3. CLAHE (clip 2.0, 8x8 karo) | `cv2.createCLAHE` | sis std 37.28 → 45.95 |
| 4. 3x3 ortalama filtresi (konvolüsyon) | `cv2.blur` | PSNR dag 22.37 → 31.70 dB, sis 22.56 → 31.00 dB |
| 5. 5x5 medyan filtresi | `cv2.medianBlur` | PSNR dag 14.55 → 45.82 dB, sis 14.66 → 36.12 dB |

Ödev 1, 2, 4 ve 5'te iki kodun çıktısı birebir aynı.
Ödev 4 girdisi `girdi/*_gurultulu.png`: `odev4-ortalama-filtre/gurultu_ekle.py` ile eklenen Gauss gürültüsü (σ=20).
Ödev 5 girdisi `girdi/*_tuz_biber.png`: `odev5-medyan-filtre/gurultu_ekle.py` ile piksellerin %10'u siyah/beyaz yapıldı.
Ödev 3 testi: `cd odev3-clahe && python test_clahe.py`. OpenCV ile en büyük fark 1, farklı piksel %0.2'nin altında
(iki karonun tam ortasındaki piksellerde float yuvarlaması).
