# Ödev 4.4: Prewitt ve Sobel Karşılaştırma Raporu

**Tarih:** 2026-10-15<br/>
**Ödev:** EBLG351 Hafta 4 — Ödev 4.4 Prewitt ve Sobel Karşılaştırması<br/>
**Çalışma Ortamı:** Python 3.14.4, OpenCV 4.15.0-dev, NumPy 2.4.6

---

## 1. Kuramsal Temeller ve Çekirdek Matrisleri
1. mertebeden yönlü türev operatörleri yatay ve dikey yöndeki yoğunluk değişimlerini hesaplar:
- **Prewitt Operatörü:**
  $$G_x = \begin{bmatrix} -1 & 0 & 1 \\ -1 & 0 & 1 \\ -1 & 0 & 1 \end{bmatrix}, \quad G_y = \begin{bmatrix} -1 & -1 & -1 \\ 0 & 0 & 0 \\ 1 & 1 & 1 \end{bmatrix}$$
- **Sobel Operatörü:**
  $$G_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \quad G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$$
- **Gradyan Büyüklüğü:**
  $$M(x, y) = \sqrt{G_x(x,y)^2 + G_y(x,y)^2}$$

## 2. Temel Farklılıklar ve Nitel Analiz
- **Yumuşatma Etkisi:** Sobel çekirdeği merkez katsayısına 2 ağırlığını vererek türev eksenine dik doğrultuda yerel bir üçgensel (1-2-1) yumuşatma gerçekleştirir. Prewitt'te ise tüm katsayılar eşit (1) ağırlıktadır. Teorik olarak Sobel yerel düzleştirme avantajı sunsa da, bu deneyde kullanılan test sahnelerinde Prewitt ve Sobel benzer kenar sürekliliği sergilemiştir.
- **İşaretli Türev:** Türev matrisleri float32 işaretli korunmuştur; görselleştirmede negatifler kaybolmasın diye dinamik min-max gerdirme veya mutlak değer uygulanmıştır.
- **Eşikleme:** İkili harita tepe büyüklüğün %20'si ($T = 0.2 \times \max(M)$) ile elde edilmiştir.

### Sayısal Ölçüm Sonuçları (Doğrudan `olcumler.csv` Verilerinden)

| Görüntü | Yöntem | Gauss Yumuşatma | Süre (ms) | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| kamera | Prewitt | Hayır | 4.12 | - | - | - |
| kamera | Prewitt | Evet | 4.34 | - | - | - |
| kamera | Sobel | Hayır | 4.13 | - | - | - |
| kamera | Sobel | Evet | 4.38 | - | - | - |
| sentetik_gurultulu | Prewitt | Hayır | 4.97 | 0.9995 | 1.0000 | 0.9998 |
| sentetik_gurultulu | Prewitt | Evet | 4.42 | 0.8759 | 1.0000 | 0.9338 |
| sentetik_gurultulu | Sobel | Hayır | 4.55 | 0.9998 | 1.0000 | 0.9999 |
| sentetik_gurultulu | Sobel | Evet | 4.55 | 0.8779 | 1.0000 | 0.9350 |

## 3. Değerlendirme ve Sınırlılıklar
- Kamera görüntüsünde ölçülen tam pipeline süreleri her iki yöntemde de birbirine çok yakındır (4.34 ms vs 4.38 ms).
- Sentetik gürültülü veride Gauss yumuşatmanın F1 skorundaki hafif düşüşü, keskin sentetik sınırların yayılması sebebiyle 1 piksel tolerans bandının dışına taşması varsayımıyla açıklanabilir; kesin nedensellik için ek deney gereklidir.
- Görsel karşılaştırma paneli: [cikti/karsilastirma_prewitt_sobel.png](cikti/karsilastirma_prewitt_sobel.png)
