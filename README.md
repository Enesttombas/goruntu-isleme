# Görüntü İşleme (EBLG351) — Ödevler

Python + OpenCV. Her haftanın kodu, girdisi ve çıktıları kendi klasöründe.

| Hafta | Ödev | Kod | Çıktı |
|---|---|---|---|
| 1 | Klasördeki resimleri 1024x768'e ölçekleme | [resize_klasor.py](hafta1/resize_klasor.py) | [cikti](hafta1/cikti/) |
| 2 | 8-bit → 6-bit nicemleme | [nicemle_6bit.py](hafta2/nicemle_6bit.py) | [6bit.png](hafta2/cikti/6bit.png) |
| 2 | 4 parça × 4 thread parlaklık, 1 vs 4 thread süre | [parca_thread.py](hafta2/parca_thread.py) | [parca_sonuc.png](hafta2/cikti/parca_sonuc.png) |
| 2 | Parça histogramları 4 thread ile, ana histogram | [histogram_thread.py](hafta2/histogram_thread.py) | [histogram.png](hafta2/cikti/histogram.png) |
| 2 | Histogram: 1 thread vs 4 thread süre | [histogram_sure.py](hafta2/histogram_sure.py) | [süreler](hafta2/cikti/histogram_sure_cikti.txt) |
| 2 | Parlaklık dönüşümü 0.75·x + 20 | [parlaklik_donusum.py](hafta2/parlaklik_donusum.py) | [karsilastirma.png](hafta2/cikti/karsilastirma.png) |
| 3 | Min-max doğrusal ölçekleme | [odev1-kontrast-germe](hafta3/odev1-kontrast-germe/) | [cikti](hafta3/odev1-kontrast-germe/cikti/) |
| 3 | Histogram eşitleme | [odev2-histogram-esitleme](hafta3/odev2-histogram-esitleme/) | [cikti](hafta3/odev2-histogram-esitleme/cikti/) |
| 3 | CLAHE | [odev3-clahe](hafta3/odev3-clahe/) | [cikti](hafta3/odev3-clahe/cikti/) |
| 3 | 3x3 ortalama filtresi | [odev4-ortalama-filtre](hafta3/odev4-ortalama-filtre/) | [cikti](hafta3/odev4-ortalama-filtre/cikti/) |
| 3 | 5x5 medyan filtresi (tuz-biber) | [odev5-medyan-filtre](hafta3/odev5-medyan-filtre/) | [cikti](hafta3/odev5-medyan-filtre/cikti/) |
| 4 | Prewitt X gradyan operatörü | [4.1-prewitt-x](hafta4/4.1-prewitt-x/) | [cikti](hafta4/4.1-prewitt-x/cikti/) |
| 4 | Prewitt Y gradyan operatörü | [4.2-prewitt-y](hafta4/4.2-prewitt-y/) | [cikti](hafta4/4.2-prewitt-y/cikti/) |
| 4 | 3x3 Sobel gradyan operatörü | [4.3-sobel](hafta4/4.3-sobel/) | [cikti](hafta4/4.3-sobel/cikti/) |
| 4 | Prewitt ve Sobel karşılaştırması | [4.4-prewitt-sobel-karsilastirma](hafta4/4.4-prewitt-sobel-karsilastirma/) | [cikti](hafta4/4.4-prewitt-sobel-karsilastirma/cikti/) |
| 4 | 3x3 Laplacian kenar tespiti | [4.5-laplacian](hafta4/4.5-laplacian/) | [cikti](hafta4/4.5-laplacian/cikti/) |
| 4 | 7x7 analitik LoG filtresi | [4.6-log](hafta4/4.6-log/) | [cikti](hafta4/4.6-log/cikti/) |
| 4 | OpenCV Canny ve filtre başarım ölçümleri | [4.7-canny](hafta4/4.7-canny/) | [cikti](hafta4/4.7-canny/cikti/) |
| 5 | RGB kanal histogramları (OpenCV ve kendi) | [5.1-renk-kanallari-histogram](hafta5/5.1-renk-kanallari-histogram/) | [cikti](hafta5/5.1-renk-kanallari-histogram/cikti/) |
| 5 | calcHist fonksiyonunu kendimiz yazma (OpenCV ile karşılaştırma) | [5.2-calchist-kendi-fonksiyonum](hafta5/5.2-calchist-kendi-fonksiyonum/) | [cikti](hafta5/5.2-calchist-kendi-fonksiyonum/cikti/) |
| 5 | RGB→gri (OpenCV vs kendi) ve mutlak fark hatası | [5.3-gri-donusum-hata](hafta5/5.3-gri-donusum-hata/) | [cikti](hafta5/5.3-gri-donusum-hata/cikti/) |
| 5 | Renk uzayı dönüşümleri (gri, YCrCb, YUV, XYZ, HSV, HLS) ve hata = 0 | [5.4-renk-uzayi-donusumu](hafta5/5.4-renk-uzayi-donusumu/) | [cikti](hafta5/5.4-renk-uzayi-donusumu/cikti/) |
| 5 | X-ray görüntülerini griye çevirip renklendirme, palet yorumu | [5.5-xray-renklendirme](hafta5/5.5-xray-renklendirme/) | [cikti](hafta5/5.5-xray-renklendirme/cikti/) |
| 6 | Canny + median, `HoughLines` ve `HoughLinesP` | [6.1-hough-line-opencv](hafta6/6.1-hough-line-opencv/) | [cikti](hafta6/6.1-hough-line-opencv/cikti/) |
| 6 | Doğru bulan Hough, kendi kodumla, oylama matrisi resmi | [6.2-hough-line-kendi](hafta6/6.2-hough-line-kendi/) | [cikti](hafta6/6.2-hough-line-kendi/cikti/) |
| 6 | Canny + median, `HoughCircles` | [6.3-hough-circles-opencv](hafta6/6.3-hough-circles-opencv/) | [cikti](hafta6/6.3-hough-circles-opencv/cikti/) |
| 6 | Çember bulan Hough, kendi kodumla, 3B oylama | [6.4-hough-circle-kendi](hafta6/6.4-hough-circle-kendi/) | [cikti](hafta6/6.4-hough-circle-kendi/cikti/) |
| 6 | Eş merkezli çember bulan Hough, gradyan yönünde oylama | [6.5-es-merkezli-cember-kendi](hafta6/6.5-es-merkezli-cember-kendi/) | [cikti](hafta6/6.5-es-merkezli-cember-kendi/cikti/) |
| 6 | Hough Line CUDA ile paralel, CPU ile birebir test | [6.6-hough-line-cuda](hafta6/6.6-hough-line-cuda/) | [cikti](hafta6/6.6-hough-line-cuda/cikti/) |
| 7 | Faz, genlik, frekans/periyot ile vektörü sinüsle doldurma | [7.1-sinus-vektor](hafta7/7.1-sinus-vektor/) | [cikti](hafta7/7.1-sinus-vektor/cikti/) |
| 7 | Genliğe ve orana göre vektöre gürültü ekleme | [7.2-gurultu-ekleme](hafta7/7.2-gurultu-ekleme/) | [cikti](hafta7/7.2-gurultu-ekleme/cikti/) |
| 7 | Gürültülü sinyali Gauss ve medyan filtreyle temizleme | [7.3-gauss-medyan-filtre](hafta7/7.3-gauss-medyan-filtre/) | [cikti](hafta7/7.3-gauss-medyan-filtre/cikti/) |
| 7 | DFT ve kendi FFT'miz, harmonik vektörü | [7.4-dft-harmonik](hafta7/7.4-dft-harmonik/) | [cikti](hafta7/7.4-dft-harmonik/cikti/) |
| 7 | 50 Hz/A=12 ve 30 Hz/A=5 sinyaller, toplamın DFT analizi | [7.5-iki-sinyal-toplam](hafta7/7.5-iki-sinyal-toplam/) | [cikti](hafta7/7.5-iki-sinyal-toplam/cikti/) |
| 7 | Eşik altı harmonikleri silen frekans filtresi | [7.6-frekans-filtresi](hafta7/7.6-frekans-filtresi/) | [cikti](hafta7/7.6-frekans-filtresi/cikti/) |
| 7 | 2 boyutlu DFT, harmonik matris | [7.7-2b-dft](hafta7/7.7-2b-dft/) | [cikti](hafta7/7.7-2b-dft/cikti/) |
| 7 | Gauss ve tuz-biber gürültü, harmonik matris karşılaştırması | [7.8-resim-gurultu](hafta7/7.8-resim-gurultu/) | [cikti](hafta7/7.8-resim-gurultu/cikti/) |
| 7 | Ters DFT ile geri dönme, ZNCC ve benzerlik metrikleri raporu | [7.9-ters-dft-zncc](hafta7/7.9-ters-dft-zncc/) | [cikti](hafta7/7.9-ters-dft-zncc/cikti/) |
| 7 | Alçak, yüksek, bant geçiren ve bant durduran filtreler | [7.10-frekans-filtreleri](hafta7/7.10-frekans-filtreleri/) | [cikti](hafta7/7.10-frekans-filtreleri/cikti/) |

Ortam: Python 3.14, OpenCV 4.15.0, NumPy, Matplotlib (7. hafta grafikleri).
