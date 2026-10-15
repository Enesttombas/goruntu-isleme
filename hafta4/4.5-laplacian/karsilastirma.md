# Ödev 4.5: 3x3 Laplacian Kenar Tespiti ve Prewitt/Sobel Karşılaştırması

**Tarih:** 2026-10-15<br/>
**Ödev:** EBLG351 Hafta 4 — Ödev 4.5 3x3 Laplacian Kenar Tespiti<br/>
**Çalışma Ortamı:** Python 3.14.4, OpenCV 4.15.0-dev, NumPy 2.4.6

---

## 1. 3x3 Laplacian Operatörü ve Matematiksel Altyapı
Laplacian, yön bağımsız (izotropik) ikinci mertebeden türev operatörüdür:
$$\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}$$

Standart 4-komşuluklu 3×3 konvolüsyon çekirdeği:
$$K_{\text{Lap}} = \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}$$

Çekirdek ayrılabilir (separable) değildir; doğrudan 2B konvolüsyonla uygulanır. Merkez katsayısı negatiftir (-4). Kenarlar, ikinci türev haritasında işaretin artıdan eksiye geçtiği **sıfır geçişi (zero-crossing)** noktaları taranarak belirlenir.

## 2. 1. Mertebe Yöntemlerle (Prewitt ve Sobel) Karşılaştırma
- **Yön Bağımlılığı (Yönlü vs İzotropik):** Prewitt ve Sobel yönlü türevlerle ($G_x, G_y$) dikey ve yatay kenarları ayrı ayrı hesaplar; gradyan büyüklüğüyle kenar şiddeti bulunur. Laplacian ise yönden bağımsızdır (izotropik), tek adımda tüm yönlerdeki kenar geçişlerini saptar.
- **Kenar Tespiti Mantığı (Eşikleme vs Sıfır Geçişi):** Prewitt ve Sobel tepe büyüklüğün %20'si gibi bir dinamik eşikle ikili kenar üretir (kenarlar yer yer kalın kalabilir). Laplacian ise ikinci türevin işaret değişimini (sıfır geçişi) arayarak kenar merkez çizgilerini saptar.
- **Gürültü Duyarlılığı:** 2. türev operatörü frekans düzleminde $\omega^2$ faktörüyle yüksek frekansları kuvvetlendirir. Bu sebeple Gauss ön yumuşatması olmaksızın gürültülü görüntülerde çok sayıda sahte kenar üretir. 1. türev operatörleri (Prewitt ve Sobel) gürültüye karşı çok daha toleranslıdır.

### Sayısal Ölçüm Sonuçları (Doğrudan `olcumler.csv` Verilerinden)

| Görüntü | Yöntem | Gauss Yumuşatma | Süre (ms) | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| kamera | Prewitt | Evet | 4.34 | - | - | - |
| kamera | Sobel | Evet | 4.38 | - | - | - |
| kamera | Laplacian | Hayır | 14.32 | - | - | - |
| kamera | Laplacian | Evet | 15.25 | - | - | - |
| sentetik_gurultulu | Prewitt | Evet | 4.42 | 0.8759 | 1.0000 | 0.9338 |
| sentetik_gurultulu | Sobel | Evet | 4.55 | 0.8779 | 1.0000 | 0.9350 |
| sentetik_gurultulu | Laplacian | Hayır | 14.33 | 0.0270 | 1.0000 | 0.0526 |
| sentetik_gurultulu | Laplacian | Evet | 13.83 | 0.0209 | 0.9996 | 0.0410 |

## 3. Bulgular ve Sınırlılıklar
- Kamera görüntüsünde 8-komşuluklu sıfır geçişi taraması sebebiyle Laplacian pipeline süresi (15.25 ms), Prewitt (4.34 ms) ve Sobel'den (4.38 ms) daha uzundur.
- Gürültülü sentetik veride 2. türev gürültüyü büyüttüğü için Laplacian F1 skoru (0.0410) 1. türev yöntemlerine (0.9350) göre belirgin şekilde düşüktür.
- Görsel çıktılar: [cikti/kamera_laplacian_gauss_ikili.png](cikti/kamera_laplacian_gauss_ikili.png)
