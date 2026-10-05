# Hafta 3

Girdi: `dag.jpg` (gri, sisli dağlar) ve `sis.jpg` (renkli sisli yol, gri tonlamada okunur). Her betik çıktısını `cikti/` klasörüne yazar.
Her ödevin iki sürümü var: elle (hazır fonksiyonsuz) ve `_hazir` (OpenCV fonksiyonlu). İki sürümün çıktısı piksel piksel aynı.

| Ödev | Elle | Hazır | Sonuç |
|---|---|---|---|
| 1. Min-max doğrusal ölçekleme | `python kontrast_germe.py` | `python kontrast_germe_hazir.py` (`cv2.normalize`) | `dag` 0-237 → 0-255, `sis` 92-249 → 0-255 |
| 2. Histogram, CDF, histogram eşitleme | `python histogram_esitleme.py` | `python histogram_esitleme_hazir.py` (`cv2.calcHist`, `cv2.equalizeHist`) | `sis` std 37.28 → 74.33, `*_histogram_cdf.png` önce/sonra |
