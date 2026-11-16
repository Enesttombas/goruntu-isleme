# 9.5 — Ten rengi maskeleri

**Ödev:** RGB, YCbCr ve HSV eşikleriyle üç ayrı ten maskesi üretme; seçilen pikselleri koruyup diğerlerini saf kırmızı yapma.

Çalışma klasöründen:

```powershell
python hafta9/9.5-ten-rengi/ten_rengi.py
```

Girdi betiğin konumuna göre `../girdi/renkli.bmp` yolundadır; 9.4 ile aynı dosya kullanılır. NumPy ve OpenCV yeterlidir. Betik pencere açmaz, import sırasında dosya işlemi yapmaz. Eksik/okunamayan girdi ve başarısız görüntü yazma işlemi hata verir.

`rgb_maskesi`, `ycbcr_maskesi` ve `hsv_maskesi` BGR uint8 resimden bool maske döndürür. **True ten olarak seçilen pikseldir.** `kirmizi_yap` orijinali kopyalar; True olanların üç kanalını birebir korur, False olanlara BGR **(0,0,255)** atar. Morfoloji veya ek maske düzeltmesi uygulanmaz.

Eşikler:

| Yöntem | Ten seçme koşulu |
|---|---|
| RGB | R>95, G>40, B>20; max(R,G,B)-min(R,G,B)>15; abs(R-G)>15; R>G ve R>B |
| YCbCr | 133≤Cr≤173 ve 77≤Cb≤127; Y için ek eşik yok |
| HSV | (0≤H≤20 veya 170≤H≤179), S≥40, V≥60 |

RGB'de kanallar fark alınmadan önce **int16** yapılır; uint8 altında negatif farklar sarılarak yanlış karar üretemez. Örneğin RGB(120,130,60) için R-G=-10'dur.

YCbCr yöntemi için OpenCV `COLOR_BGR2YCrCb` kullanılır. Çıktı sırası **Y,Cr,Cb** olduğundan Cr kanal 1'den, Cb kanal 2'den alınır. İsimdeki YCbCr sırası doğrudan dizi sırası olarak kullanılmaz.

HSV `COLOR_BGR2HSV` ile hesaplanır. uint8 H 0..179, S ve V 0..255'tir. H daireseldir: 179'dan sonra 0 gelir. Kırmızı çevresini kapsamak için iki aralık OR ile birleştirilir; bunlar yaklaşık 0..40° ve 340..358°'dir. S alt eşiği griye yakın pikselleri, V alt eşiği çok karanlık pikselleri eler. Aralıklar örnek tercihlerdir.

| Çıktı | Açıklama |
|---|---|
| `cikti/rgb_maske.bmp`, `ycbcr_maske.bmp`, `hsv_maske.bmp` | Tek kanallı uint8: seçilen ten 255, diğerleri 0 |
| `cikti/rgb.bmp`, `ycbcr.bmp`, `hsv.bmp` | Ten korunmuş, diğer pikseller saf kırmızı BGR |
| `cikti/karsilastirma.png` | Soldan sağa orijinal, RGB, YCbCr, HSV; başlıklarda sayım |
| `cikti/sayim.txt` | Seçilen piksel sayısı ve tüm görüntüye yüzdesi |

Verilen 512×512 girdi için:

| Yöntem | Seçilen / toplam piksel | Yüzde |
|---|---:|---:|
| RGB | 72432 / 262144 | %27,63 |
| YCbCr | 84709 / 262144 | %32,31 |
| HSV | 90944 / 262144 | %34,69 |

Bu sayılar maskenin seçtiği alanı ölçer; doğruluk oranı değildir. Maskeler bir eşikleme örneğidir. Işık ve cilt tonu değişimleri kaçırılan ten piksellerine; turuncu/kahverengi arka planlar yanlış seçime yol açabilir. Evrensel başarı veya ölçülmemiş doğruluk iddiası yoktur.

Kaynak: [OpenCV renk dönüşümleri](https://docs.opencv.org/4.x/de/d25/imgproc_color_conversions.html) (YCrCb kanal sırası ve HSV kodlaması).
