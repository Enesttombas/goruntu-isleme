# Ödev 4.1: Prewitt ve Sobel Kenar Tespiti

Bu ödevde kamera görüntüsü üzerinde 1. mertebeden yönlü gradyan filtreleri (Prewitt ve Sobel) incelenmiştir.

## Yöntem ve Çekirdekler

1. **Ön İşleme:** Yüksek frekanslı sensör gürültüsünü bastırmak için 5x5 Gauss filtresi ($\sigma=1.0$) uygulanmıştır.
2. **Prewitt Çekirdekleri:**
   - $G_x = \begin{bmatrix} -1 & 0 & 1 \\ -1 & 0 & 1 \\ -1 & 0 & 1 \end{bmatrix}$, $G_y = \begin{bmatrix} -1 & -1 & -1 \\ 0 & 0 & 0 \\ 1 & 1 & 1 \end{bmatrix}$
3. **Sobel Çekirdekleri:**
   - $G_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}$, $G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$
4. **Büyüklük:** $M = \sqrt{G_x^2 + G_y^2}$
5. **İkili Eşikleme:** $T = 0.2 \times \max(M)$ eşiğiyle ikili kenar haritası oluşturulmuştur.

## Prewitt ve Sobel Karşılaştırması

- **Yumuşatma Etkisi:** Sobel çekirdeği merkez katsayıya 2 ağırlığını vererek türev yönüne dik eksende yerel Gauss yumuşatma gerçekleştirir. Bu nedenle Sobel, gürültülü kenarlarda Prewitt'e kıyasla daha kararlı ve pürüzsüz yanıt verir.
- **Tepki Genliği:** Sobel katsayılarının toplam genliği Prewitt'e göre daha büyüktür ($4 \times$ basamak faktörü), gradyan tepesi daha belirgindir.
- **Hesaplama:** İki yöntem de 3x3 ayrılabilir çekirdeklerle çok hızlıdır; aralarındaki süre farkı mikrosaniye mertebesindedir.

## Çıktılar

- `kamera_gauss.png`: Ön işleme uygulanmış kamera görüntüsü.
- `kamera_prewitt_gx.png`, `gy.png`, `hypot.png`, `ikili.png`: Prewitt bileşenleri.
- `kamera_sobel_gx.png`, `gy.png`, `hypot.png`, `ikili.png`: Sobel bileşenleri.
- `karsilastirma_prewitt_sobel.png`: Yan yana karşılaştırma paneli.
