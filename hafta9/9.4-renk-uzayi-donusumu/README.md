# 9.4 — Renk uzayı dönüşümleri

**Ödev:** Renkli BMP'yi HSV, HSI, YUV ve CIE Lab uzaylarına dönüştürüp BMP olarak kaydetme; kanalları karşılaştırma.

Çalışma klasöründen:

```powershell
python hafta9/9.4-renk-uzayi-donusumu/renk_uzaylari.py
```

Girdi betiğin konumuna göre `../girdi/renkli.bmp` yolundan okunur. Çalışma dizini değişse de aynı girdi kullanılır. NumPy ve OpenCV yeterlidir; pencere açılmaz. Okunamayan girdi ve başarısız görüntü yazma işlemi hata verir. Import sırasında dosya okunmaz veya yazılmaz.

OpenCV görüntüyü **BGR** sırasıyla okur. `donustur` içinde `COLOR_BGR2HSV`, `COLOR_BGR2YUV` ve `COLOR_BGR2Lab` kullanılır. Üç kanallı, boş olmayan uint8 görüntü beklenir.

HSI, HLS'den farklıdır; `hsi_donustur` doğrudan şu hesabı yapar. R, G ve B önce float64'e çevrilir:

- `I = (R+G+B)/3`, burada girişler ve I 0..255 aralığındadır.
- `H = atan2(sqrt(3)*(G-B), 2R-G-B)`, dereceye çevrilip mod 360 ile [0,360) aralığına alınır.
- I>0 için `S = 1-min(R,G,B)/I`; siyah için S=0. R=G=B ise H=0, S=0.
- Bölme yalnız I>0 olan yerlerde yapılır; siyah piksel sıfıra bölünmez.

BMP için H/2, S*255 ve I saklanır. `np.rint` en yakın tam sayıya yuvarlar (tam yarımda çift sayıya); H 0..179'a, S ve I 0..255'e kırpılıp uint8'e çevrilir. 360 dereceye çok yakın H'nin yuvarlanarak 180 olması 179'a kırpılır. Bu niceleme hassasiyet kaybettirir. Örneğin saf kırmızı HSI=(0°,1,85), BMP üçlüsü (0,255,85); RGB(120,60,60) için I=80, S=0,25 ve BMP üçlüsü (0,64,80).

| Dosya | Kanal sırası / anlamı |
|---|---|
| `cikti/hsv.bmp` | H:0..179 (iki derece/birim), S:0..255, V:0..255 |
| `cikti/hsi.bmp` | H/2, S*255, I |
| `cikti/yuv.bmp` | Y, U, V; OpenCV'nin üç kanallı uint8 YUV dönüşümü |
| `cikti/lab.bmp` | L*=L*255/100, a*=a+128, b*=b+128 biçiminde kodlanmış kanallar |
| `cikti/karsilastirma.png` | Satırlar HSV, HSI, YUV, Lab; her satırda orijinal ve üç etiketli gri kanal |

Buradaki YUV tam çözünürlüklü üç kanaldır; alt örneklemeli video biçimi veya YCrCb değildir. Karşılaştırmada kanal değerleri 0..255 gri olarak doğrudan gösterilir, ayrı kontrast normalleştirmesi yapılmaz. Bu nedenle 0..179 H düzlemi tam beyaza ulaşmaz.

**Ham dönüşüm BMP'leri sıradan RGB görüntüler değildir.** Resim görüntüleyici kanalları renk bileşenleri sanarak yanlış renk gösterir. OpenCV ile ham kanal değerlerini şöyle geri okuyun:

```python
from pathlib import Path
import cv2

yol = Path("hafta9/9.4-renk-uzayi-donusumu/cikti/hsi.bmp")
ham = cv2.imread(str(yol), cv2.IMREAD_UNCHANGED)
h_kodu, s_kodu, i_kodu = ham[0, 0]
```

OpenCV ile yazılan dizi aynı yöntemle okununca kanal değerleri ve sırası birebir korunur; bu, dönüşümün veya uint8 nicelemesinin tersinir olduğu anlamına gelmez. Ters HSI dönüşümü yapılmaz.

Kaynaklar: [OpenCV renk dönüşümleri](https://docs.opencv.org/4.x/de/d25/imgproc_color_conversions.html) (HSV, Lab ve BGR dönüşüm kodları), [Pascal Getreuer - Colorspace Transforms](https://gmao.gsfc.nasa.gov/media/gmaoftp/jkolassa/Matlab_scripts/colorspace.html) (HSI). HSI, yukarıda açıklanan ton açısı, doygunluk ve ortalama yoğunluk formülleriyle hesaplanır.
