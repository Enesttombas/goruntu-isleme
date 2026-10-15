# Ödev 4.6: 7x7 LoG (Laplacian of Gaussian) Kenar Tespiti ve Karşılaştırma

**Tarih:** 2026-10-15<br/>
**Ödev:** EBLG351 Hafta 4 — Ödev 4.6 7x7 Analitik LoG<br/>
**Çalışma Ortamı:** Python 3.14.4, OpenCV 4.15.0-dev, NumPy 2.4.6

---

## 1. Gerçek 7x7 Analitik LoG Çekirdeği
LoG (Laplacian of Gaussian), Gauss yumuşatma ile Laplacian 2. türevini analitik olarak tek bir formülde birleştirir:
$$LoG(x, y) = \frac{r^2 - 2\sigma^2}{\sigma^4} \exp\left(-\frac{r^2}{2\sigma^2}\right), \quad r^2 = x^2 + y^2$$

$\sigma=1.0$ için $7 \times 7$ çekirdek üretilir ($x, y \in [-3, 3]$).
- Çekirdek ortalaması $0$'a çekilmiştir (düz alanlarda sıfır tepki).
- Mutlak toplamı $1.0$'a normalize edilmiştir.
- Merkez piksel değeri negatiftir.
- **Önemli Not:** LoG çekirdeği Gauss filtresini formülünde zaten içerdiğinden, görüntüye **ikinci bir harici Gauss yumuşatması uygulanmaz**.

## 2. Diğer Yöntemlerle (Prewitt, Sobel, Laplacian, Canny) Karşılaştırma
Tüm yöntem varyantlarının deneysel ölçüm sonuçları (3 ısınma + 20 tekrar medyanı):

| Görüntü | Yöntem | Gauss Yumuşatma | Süre (ms) | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| kamera | Prewitt | Hayır | 4.119 | - | - | - |
| kamera | Prewitt | Evet | 4.345 | - | - | - |
| kamera | Sobel | Hayır | 4.128 | - | - | - |
| kamera | Sobel | Evet | 4.379 | - | - | - |
| kamera | Laplacian | Hayır | 14.318 | - | - | - |
| kamera | Laplacian | Evet | 15.251 | - | - | - |
| kamera | LoG | Dahili (Evet) | 17.567 | - | - | - |
| kamera | Canny | Hayır | 3.446 | - | - | - |
| kamera | Canny | Evet | 2.450 | - | - | - |
| sentetik | Prewitt | Hayır | 6.134 | 1.0000 | 1.0000 | 1.0000 |
| sentetik | Prewitt | Evet | 8.080 | 0.8707 | 1.0000 | 0.9309 |
| sentetik | Sobel | Hayır | 7.202 | 1.0000 | 1.0000 | 1.0000 |
| sentetik | Sobel | Evet | 7.766 | 0.8743 | 1.0000 | 0.9329 |
| sentetik | Laplacian | Hayır | 15.485 | 1.0000 | 1.0000 | 1.0000 |
| sentetik | Laplacian | Evet | 10.967 | 1.0000 | 1.0000 | 1.0000 |
| sentetik | LoG | Dahili (Evet) | 10.581 | 0.7415 | 1.0000 | 0.8515 |
| sentetik | Canny | Hayır | 1.483 | 1.0000 | 1.0000 | 1.0000 |
| sentetik | Canny | Evet | 1.589 | 1.0000 | 1.0000 | 1.0000 |
| sentetik_gurultulu | Prewitt | Hayır | 4.971 | 0.9995 | 1.0000 | 0.9998 |
| sentetik_gurultulu | Prewitt | Evet | 4.424 | 0.8759 | 1.0000 | 0.9338 |
| sentetik_gurultulu | Sobel | Hayır | 4.553 | 0.9998 | 1.0000 | 0.9999 |
| sentetik_gurultulu | Sobel | Evet | 4.553 | 0.8779 | 1.0000 | 0.9350 |
| sentetik_gurultulu | Laplacian | Hayır | 14.333 | 0.0270 | 1.0000 | 0.0526 |
| sentetik_gurultulu | Laplacian | Evet | 13.833 | 0.0209 | 0.9996 | 0.0410 |
| sentetik_gurultulu | LoG | Dahili (Evet) | 12.434 | 0.0196 | 1.0000 | 0.0385 |
| sentetik_gurultulu | Canny | Hayır | 2.741 | 0.0712 | 1.0000 | 0.1329 |
| sentetik_gurultulu | Canny | Evet | 1.066 | 1.0000 | 1.0000 | 1.0000 |

## 3. Değerlendirme ve Sınırlılıklar
- **LoG vs Laplacian:** 7x7 LoG, ayrık Gauss + Laplacian ikilisine göre analitik olarak tek konvolüsyonda işlem yapar ve sıfır geçişleriyle kenar adaylarını belirler.
- **LoG vs Canny:** Canny NMS (yerel olmayan maksimumların bastırılması) ve çift eşikli histerezis sayesinde kenarları NMS ile inceltir ve gürültülü veride 1.0000 F1 puanına ulaşır. LoG sıfır geçişleri kalın veya kesintili olabilir; uygulama kontur kapatma ya da inceltme yapmaz.
- Görsel çıktılar: [cikti/kamera_log_ikili.png](cikti/kamera_log_ikili.png)
