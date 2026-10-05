# Ödev 2: Histogram eşitleme

Düşük kontrastlı, tek kanallı görüntüyü aç; 256 değerli histogramını çıkar; histogramdan
kümülatif dağılımı (CDF) hesapla; CDF ile histogram eşitleme yap ve sonucu kaydet.

| Sürüm | Çalıştır | Kullanılan |
|---|---|---|
| Hazır fonksiyonsuz | `cd hazir_fonksiyonsuz && python histogram_esitleme.py` | histogram, CDF ve eşitleme tablosu döngüyle elle |
| Hazır fonksiyonlu | `cd hazir_fonksiyonlu && python histogram_esitleme.py` | `cv2.calcHist`, `np.cumsum`, `cv2.equalizeHist` |

Eşitleme: `yeni = round((CDF[x] - CDF_min) / (N - CDF_min) * 255)`, N = piksel sayısı.

| Girdi | Önce std | Sonra std |
|---|---|---|
| `dag.jpg` | 78.63 | 74.65 |
| `sis.jpg` | 37.28 | 74.33 |

Histogram ve CDF önce/sonra: `hazir_fonksiyonsuz/cikti/*_histogram_cdf.png`.
