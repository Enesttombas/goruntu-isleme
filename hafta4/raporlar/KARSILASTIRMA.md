# EBLG351 Görüntü İşleme - Hafta 4 Kenar Tespit Raporu

**Tarih:** 2026-10-15
**Öğrenci / Depo:** EBLG351 Hafta 4 Kenar Algoritmaları
**Çalışma Ortamı:** Python 3.14.4, OpenCV 4.15.0-dev, NumPy 2.4.6, Tek İş Parçacığı (`cv2.setNumThreads(1)`), Görüntü Boyutları: 512×512 piksel
**Ölçüm Protokolü:** 3 ısınma + 20 tekrar `time.perf_counter()` medyanı
**Ölçülen Pipeline:** Ön filtre (varsa Gauss) + çekirdek/gradyan konvolüsyonu + ikili kenar haritası dönüşümü (Prewitt/Sobel için NumPy hypot ve tepe değerin %20'si eşikleme; Laplacian/LoG için 8-komşuluk sıfır geçişi; Canny için cv2.Canny dahili NMS ve çift eşikli histerezis). Dosya disk okuma/yazma (I/O) ve F1 skorlama süre ölçümüne dahil değildir.

---

## 1. Prewitt ve Sobel Karşılaştırması

### Kuramsal Temeller ve Çekirdekler
1. mertebeden yönlü türev operatörleri görüntüdeki ani yoğunluk değişimlerini hesaplar.
- **Prewitt Operatörü:**
  $$\mathbf{G_x} = \begin{bmatrix} -1 & 0 & 1 \\ -1 & 0 & 1 \\ -1 & 0 & 1 \end{bmatrix}, \quad \mathbf{G_y} = \begin{bmatrix} -1 & -1 & -1 \\ 0 & 0 & 0 \\ 1 & 1 & 1 \end{bmatrix}$$
- **Sobel Operatörü:**
  $$\mathbf{G_x} = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \quad \mathbf{G_y} = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$$
- **Gradyan Büyüklüğü:**
  $$M(x, y) = \sqrt{G_x(x,y)^2 + G_y(x,y)^2}$$

### Farklılıklar ve Nitel Analiz
- **Yumuşatma Etkisi ve Gürültü Davranışı:** Sobel çekirdeği merkez katsayısına $2$ ağırlığını vererek türev eksenine dik doğrultuda yerel bir üçgensel (1-2-1) yumuşatma gerçekleştirir. Prewitt'te ise tüm katsayılar eşit ($1$) ağırlıktadır. Teorik olarak Sobel yerel düzleştirme avantajı sunsa da, bu deneyde kullanılan test sahnelerinde Prewitt ve Sobel benzer kenar sürekliliği ve gürültü davranışı sergilemiştir; bu çalışma tek başına genel bir 'üstünlük' iddiası olarak yorumlanamaz.
- **Türev İşareti ve Görselleştirme:** Türev matrisleri işaretli (float32) üretilmiştir. Görselleştirmede negatif değerleri sıfıra kırpmak yerine doğrusal gerdirme (min-max normalization) veya mutlak değer alınarak tüm gradyan yönleri eksiksiz korunmuştur.
- **İkili Eşikleme Sınırlılığı:** İkili kenar haritası tepe gradyanın %20'si ($T = 0.2 \times \max(M)$) ile üretilmiştir. Bu dinamik ölçekleme sabit resimlerde doğru olarak 0 kenar üretir; ancak sahnede çok parlak tekil bir yansıma varsa diğer gerçek kenarlar eşik altında kalabilir.
- **Hesaplama Süreleri:** Kamera görüntüsünde ölçülen tam pipeline süreleri Prewitt için Gauss'suz 4.12 ms / Gauss'lu 4.34 ms; Sobel için Gauss'suz 4.13 ms / Gauss'lu 4.38 ms düzeyindedir. Her iki filtre de 3×3 konvolüsyon ve NumPy tabanlı ikili eşikleme adımlarına sahip olduğundan süreleri birbirine çok yakındır.

Görsel: [Prewitt - Sobel Karşılaştırma Paneli](../cikti/karsilastirma_prewitt_sobel.png)

---

## 2. Laplacian, LoG ve Gauss Ön Filtreleme Etkisi

### 2. Mertebe İzotropik Türev ve Sıfır Geçişleri
- **4-Komşuluk 3x3 Laplacian:**
  $$\nabla^2 = \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}$$
  Merkez katsayı negatiftir (-4). Çekirdek matrisi **ayrılabilir (separable) değildir**; doğrudan 2B konvolüsyon ile uygulanır.
- **7x7 Analitik LoG (Laplacian of Gaussian) Çekirdeği ($\sigma=1.0$):**
  $$LoG(x, y) = \frac{r^2 - 2\sigma^2}{\sigma^4} \exp\left(-\frac{r^2}{2\sigma^2}\right), \quad r^2 = x^2 + y^2$$
  Çekirdek ortalaması $0$'a çekilmiş ve mutlak toplamı $1$'e normalize edilmiştir. Merkez katsayısı negatiftir.
- **Sıfır Geçişi (Zero-Crossing):** İkinci türevin işareti kenar sınırında artıdan eksiye geçer. 8-komşulukta işaret değişimi olan ve mutlak farkı dinamik eşiği aşan pikseller kenar olarak işaretlenir. Sıfır geçişine yakın pikseller işaretlenir; kenarlar kalın veya kesintili olabilir, NMS veya kontur inceltme/kapatma uygulanmaz (1 piksel kalınlığında kapalı kontur garantisi yoktur).

### Gauss Ön Filtrelemenin Rolü ve Sınırları
2. mertebeden türev filtresi frekans düzleminde $\omega^2$ çarpanı sebebiyle yüksek frekanslı gürültüyü büyütür ve sahte sıfır geçişlerinin sayısını aşırı artırabilir. 5x5 Gauss ($\sigma=1.0$) uygulandığında veya dahili Gauss içeren LoG kullanıldığında bu sahte sıfır geçişleri bastırılır. Ancak Gauss ön filtrelemenin her veri ve eşik değerinde her metriği otomatik olarak iyileştireceği varsayılmamalıdır; nitekim sentetik gürültülü veride kenar yayılması nedeniyle Laplacian ve LoG doğruluk metriklerinde düşüş kaydedilmiştir.

Görsel: [Laplacian - LoG Karşılaştırma Paneli](../cikti/karsilastirma_laplacian_log.png)

---

## 3. Canny, Sayısal Ölçümler ve Yöntem Sınırlılıkları

### Canny Algoritması
Canny algoritması (OpenCV `cv2.Canny`, histerezis eşikleri 50 ve 100, `L2gradient=True`) yönlü gradyan hesabı, Non-Maximum Suppression (NMS ile 1 piksel inceltme) ve çift eşikli histerezis adımlarından oluşur.

### Deneysel Ölçüm Sonuçları (3 Isınma + 20 Ölçüm Medyanı, cv2.setNumThreads(1))

| Görüntü | Yöntem | Gauss Yumuşatma | Süre (ms) | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| kamera | Prewitt | Hayır | 4.119 | - | - | - |
| kamera | Prewitt | Evet | 4.345 | - | - | - |
| kamera | Sobel | Hayır | 4.128 | - | - | - |
| kamera | Sobel | Evet | 4.379 | - | - | - |
| kamera | Laplacian | Hayır | 14.318 | - | - | - |
| kamera | Laplacian | Evet | 15.251 | - | - | - |
| kamera | LoG | Evet | 17.567 | - | - | - |
| kamera | Canny | Hayır | 3.446 | - | - | - |
| kamera | Canny | Evet | 2.450 | - | - | - |
| sentetik | Prewitt | Hayır | 6.134 | 1.0000 | 1.0000 | 1.0000 |
| sentetik | Prewitt | Evet | 8.080 | 0.8707 | 1.0000 | 0.9309 |
| sentetik | Sobel | Hayır | 7.202 | 1.0000 | 1.0000 | 1.0000 |
| sentetik | Sobel | Evet | 7.766 | 0.8743 | 1.0000 | 0.9329 |
| sentetik | Laplacian | Hayır | 15.485 | 1.0000 | 1.0000 | 1.0000 |
| sentetik | Laplacian | Evet | 10.967 | 1.0000 | 1.0000 | 1.0000 |
| sentetik | LoG | Evet | 10.581 | 0.7415 | 1.0000 | 0.8515 |
| sentetik | Canny | Hayır | 1.483 | 1.0000 | 1.0000 | 1.0000 |
| sentetik | Canny | Evet | 1.589 | 1.0000 | 1.0000 | 1.0000 |
| sentetik_gurultulu | Prewitt | Hayır | 4.971 | 0.9995 | 1.0000 | 0.9998 |
| sentetik_gurultulu | Prewitt | Evet | 4.424 | 0.8759 | 1.0000 | 0.9338 |
| sentetik_gurultulu | Sobel | Hayır | 4.553 | 0.9998 | 1.0000 | 0.9999 |
| sentetik_gurultulu | Sobel | Evet | 4.553 | 0.8779 | 1.0000 | 0.9350 |
| sentetik_gurultulu | Laplacian | Hayır | 14.333 | 0.0270 | 1.0000 | 0.0526 |
| sentetik_gurultulu | Laplacian | Evet | 13.833 | 0.0209 | 0.9996 | 0.0410 |
| sentetik_gurultulu | LoG | Evet | 12.434 | 0.0196 | 1.0000 | 0.0385 |
| sentetik_gurultulu | Canny | Hayır | 2.741 | 0.0712 | 1.0000 | 0.1329 |
| sentetik_gurultulu | Canny | Evet | 1.066 | 1.0000 | 1.0000 | 1.0000 |

### Sonuçların Değerlendirilmesi ve Sınırlılıklar
1. **Çalışma Süreleri:** Aynı kamera görüntüsü ve Gauss=Evet varyantlarında (LoG dahili Gauss içerir), ölçülen pipeline sürelerine göre bu çalıştırmada en hızlı yöntem 2.45 ms ile **Canny** olmuştur. Canny'nin hız avantajı OpenCV'nin C++ düzeyindeki optimize kodundan kaynaklanmaktadır. Prewitt (4.34 ms) ve Sobel (4.38 ms) Python/NumPy ikili eşikleme adımlarıyla birbirine çok yakın seyrederken; piksel bazlı 8-komşuluk sıfır geçişi taraması yapan Laplacian (15.25 ms) ve LoG (17.57 ms) en yüksek işlem sürelerine sahiptir.
2. **Doğruluk Analizi ve Gürültü Tepkisi:** Sentetik gürültülü ($\sigma=10$) görüntüde Canny'ye Gauss ön filtreleme uygulandığında Precision 0.0712'den 1.0000'ye, F1 skoru 0.1329'den 1.0000'e yükselmiştir. Ancak Gaussian yumuşatma her yöntemi iyileştirmemiştir: Prewitt'te F1 skoru 0.9998'den 0.9338'ye, Sobel'de 0.9999'den 0.9350'ye, Laplacian'da ise 0.0526'den 0.0410'ye gerilemiştir. Bu düşüş, Gauss filtresinin keskin sentetik sınırları yayarak kenar bölgesini 1 piksel tolerans bandının ötesine taşıması ve eşikleme ile birlikte sınır kalınlaşması oluşturmasıyla ilişkilendirilebilir; nedenleri ayırmak için ek deney gerekir. Canny'de ise NMS adımı kenarları yönlü maksimumda 1 piksele incelttiği için bu yayılma avantaja dönüşmektedir.
3. **Sınırlılıklar:**
   - Precision, recall ve F1, bilinen sentetik referans kenar maskesine göre 1 piksel toleransla morfolojik genişletme kullanılarak ölçülmüştür; gerçek fotoğraflarda zemin gerçeği bulunmadığı için doğruluk uydurulmamıştır.
   - Elde edilen metrik ve süreler bu tek sahne, sentetik geometri ve belirlenen parametre setine (eşik oranları, $\sigma$) özgüdür; evrensel bir algoritma üstünlük sıralaması olarak genellenemez.

### Kaynaklar ve Dokümantasyon
- OpenCV GaussianBlur: [docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html)
- OpenCV Sobel & Laplacian: [docs.opencv.org/4.x/d2/d2c/tutorial_sobel_derivatives.html](https://docs.opencv.org/4.x/d2/d2c/tutorial_sobel_derivatives.html)
- OpenCV Canny: [docs.opencv.org/4.x/da/d22/tutorial_py_canny.html](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html)
- LoG Kuramı ve Laplacian of Gaussian: [scipy.ndimage.gaussian_laplace](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_laplace.html)
