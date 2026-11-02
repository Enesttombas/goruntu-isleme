# 6.1 — Canny + median, iki OpenCV Hough Line fonksiyonu

**Ödev:** Resmin kenarları Canny ile bulunur, kenar resmi median filtreden geçirilir, sonra iki farklı OpenCV Hough Line fonksiyonuyla doğrular bulunup resim üzerine farklı renkle çizilir.

| Fonksiyon | Ne döndürür | Renk | Çıktı |
|---|---|---|---|
| `cv2.HoughLines` | Sonsuz doğru (rho, theta) | kırmızı | `cikti/dogrular_houghlines.png` |
| `cv2.HoughLinesP` | Doğru parçası (x1, y1, x2, y2) | yeşil | `cikti/dogrular_houghlinesp.png` |

Ara çıktılar: `cikti/kenar_canny.png`, `cikti/kenar_median.png`.

```powershell
python hafta6/6.1-hough-line-opencv/opencv_ile.py
```

Sonuç: `HoughLines` 13 doğru, `HoughLinesP` 31 doğru parçası buldu.

Gözlem: Canny kenarı 1 piksel kalınlıktadır. 3x3 median ince kenarın çoğunu siler: 29537 kenar pikselinden 7129'u kaldı (%24). Bu yüzden eşikleri düşük tuttum (`HoughLines` 100 oy, `HoughLinesP` 50 oy). Gürültü azalıyor, ama zayıf doğrular da kayboluyor.
