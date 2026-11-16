# 9.3 — 8K görüntünün bütün piksel değerlerini ikiye bölme

```powershell
python hafta9/9.3-8k-piksel-bolme/piksel_bolme.py
```

Betik 7680×4320 boyutunda uint8, üç kanallı, tekrarlanabilir renk geçişleri
üretir. `girdi/8k.bmp` dosyasını kaydedip OpenCV ile diskten açar. Her B, G ve R
elemanına tamsayı bölme uygulanır: `0, 1, 2, 3, 254, 255` değerleri sırasıyla
`0, 0, 1, 1, 127, 127` olur. Orijinal dizi değişmez.

NumPy çıktı tamponu ölçümden önce ayrılır ve doldurulur. Bir ısınmadan sonra
aynı orijinal üzerinden yedi ölçüm yapılır. `perf_counter_ns` yalnız bölme
çağrısını çevreler; görüntü üretimi, disk işlemleri, tampon hazırlığı,
doğrulama ve önizleme süreye dahil edilmez. Sonuçlar çalışma anında ölçülür;
`cikti/sure.txt` her ölçümü, gerçek medyan/minimum/maksimum değerlerini ns ve ms
olarak, çözünürlüğü, Python/OpenCV/NumPy sürümlerini ve ölçüm sayısını içerir.

Tam `x // 2` doğrulaması, son blok dahil bütün satırları ve üç kanalı 64 satırlık
bloklarla karşılaştırır. Tam boyutta bir beklenen sonuç dizisi oluşturulmaz.
Kaydedilen sonuç BMP'si de diskten okunup aynı şekilde doğrulanır.

| Dosya | İçerik |
|---|---|
| `girdi/8k.bmp` | Üretilen orijinal 8K BMP |
| `cikti/8k_yari.bmp` | Bütün BGR değerleri ikiye bölünmüş BMP |
| `cikti/sure.txt` | Gerçek çalışma süreleri ve ortam sürümleri |
| `cikti/yan_yana.png` | Solda orijinal, sağda sonuç; her biri 960×540 |

İki 8K BMP her çalıştırmada yeniden üretilir ve Git dışında tutulmalıdır;
depoya eklenmez. Her biri yaklaşık 99,5 MB'dir (99.532.854 bayt). Büyük BMP
yolları yalnız yukarıdaki iki yoldur. Gerçek 8K giriş ve sonuç oluşturulup diskten
yeniden okunarak bütün kanal değerleri doğrulandı. [Süre raporu](cikti/sure.txt)
ve [önizleme](cikti/yan_yana.png) depodadır.

Bu çalıştırmada yedi ölçümün medyanı **9,1075 ms**, minimumu **8,6420 ms**,
maksimumu **9,7183 ms** çıktı. Ortam: Python 3.14.4, OpenCV 4.14.0, NumPy 2.4.6.
Süre, bilgisayara ve o andaki sistem yüküne bağlıdır. Küçük örnekli otomatik
testlere ek olarak tam 8K dosyalarının bütün pikselleri ayrıca doğrulandı.
