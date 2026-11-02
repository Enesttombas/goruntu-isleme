# 6.3 — Canny + median, OpenCV HoughCircles

**Ödev:** Resmin kenarları Canny ile bulunur, kenar resmi median filtreden geçirilir, sonra `cv2.HoughCircles` ile çemberler bulunup resim üzerine farklı renkle çizilir.

| Dosya | Çıktı |
|---|---|
| [opencv_ile.py](opencv_ile.py) | `cikti/kenar_canny.png`, `cikti/kenar_median.png`, `cikti/cemberler_houghcircles.png` |

```powershell
python hafta6/6.3-hough-circles-opencv/opencv_ile.py
```

Parametreler: `HOUGH_GRADIENT`, `dp=2`, `minDist=30`, `param2=20`, yarıçap 20-40.

Sonuç: 12 çember bulundu, çoğu şekerlere oturuyor. Bazılarında yarıçap biraz büyük, birbirine değen şekerlerde merkez kayıyor.

Gözlem: median kenar piksellerini 3400'den 1145'e indirdi. `dp=1` ile oylar dağıldı, eşik çok düşmeden çember çıkmadı. `dp=2` (yarı çözünürlüklü oylama) oyları topladı. `param2=15` ile 18 çember çıktı, birkaçı yanlıştı; `param2=22` ile 9'a düştü.
