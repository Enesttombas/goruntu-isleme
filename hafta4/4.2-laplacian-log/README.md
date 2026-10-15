# Ödev 4.2: Laplacian ve LoG (Laplacian of Gaussian)

Bu ödevde 2. mertebeden yön bağımsız türev filtreleri ve sıfır geçişleri (zero-crossing) incelenmiştir.

## Çekirdekler ve Matematiksel Altyapı

1. **4-Komşuluk 3x3 Laplacian:**
   - $\nabla^2 = \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}$
   - Merkez katsayı negatif (-4), doğrudan komşular pozitif (+1).
2. **7x7 Analitik LoG (Marr-Hildreth):**
   - $LoG(x, y) = \frac{r^2 - 2\sigma^2}{\sigma^4} \exp\left(-\frac{r^2}{2\sigma^2}\right)$, $\sigma=1.0$, $r^2=x^2+y^2$.
   - Çekirdek ortalaması sıfır yapılmış ve mutlak toplamı 1'e normalize edilmiştir.
3. **Sıfır Geçişi (Zero-Crossing):**
   - 2. türev haritasında bir piksel pozitifken 8 komşusundan biri negatifse ve aradaki işaret farkı dinamik eşik değerini aşıyorsa kenar pikselleri işaretlenir.

## Gauss Ön Filtrelemenin Önemi

- **Gauss'suz Laplacian:** İkinci türev, yüksek frekanslı gürültüyü katlanarak yükseltir. Ön filtreleme yapılmadığında homojen zeminlerde dahi binlerce sahte kenar üretilir.
- **Gauss'lu Laplacian vs LoG:** Gauss ardından Laplacian uygulamak (ayrık iki aşama) ile analitik LoG çekirdeği uygulamak kuramsal olarak eşdeğerdir. LoG tek adımda hem yumuşatır hem 2. türev alır.

## Çıktılar

- `kamera_laplacian_nogauss_ikili.png`: Gauss filtresiz gürültülü sıfır geçişleri.
- `kamera_laplacian_gauss_ikili.png`: Gauss ön filtrelemeli temiz sıfır geçişleri.
- `kamera_log_ikili.png`: 7x7 analitik LoG sıfır geçiş haritası.
- `karsilastirma_laplacian_log.png`: Karşılaştırma paneli.
