# 9. hafta - Renk uzayları ve BMP

| Ödev | İçerik | Sonuç |
|---|---|---|
| [9.1 - Renk uzayları](9.1-renk-uzaylari/) | Avantajlar, sınırlamalar ve kullanım alanları; kaynaklı ve görselli özet | [3 sayfalık ders notu](9.1-renk-uzaylari/cikti/renk_uzaylari_ders_notu.pdf) |
| [9.2 - BMP ve gri dönüşüm](9.2-bmp-gri/) | 1000 × 1000 renkli BMP, gri dönüşüm, disk boyutu karşılaştırması | [Boyut raporu](9.2-bmp-gri/cikti/boyutlar.txt), [karşılaştırma](9.2-bmp-gri/cikti/yan_yana.png) |
| [9.3 - 8K piksel bölme](9.3-8k-piksel-bolme/) | 7680 × 4320 × 3 BMP'de tüm değerleri ikiye bölme; yalnız işlem süresi | [Süre raporu](9.3-8k-piksel-bolme/cikti/sure.txt), [önizleme](9.3-8k-piksel-bolme/cikti/yan_yana.png) |
| [9.4 - Renk uzayı dönüşümleri](9.4-renk-uzayi-donusumu/) | HSV, HSI, YUV ve CIE Lab; BMP olarak ham kanallar | [Çıktılar](9.4-renk-uzayi-donusumu/cikti/) |
| [9.5 - Ten rengi](9.5-ten-rengi/) | RGB, YCbCr ve HSV maskeleri; ten dışını tam kırmızı gösterme | [Çıktılar](9.5-ten-rengi/cikti/) |

Ortak girdi: [renkli.bmp](girdi/renkli.bmp), [kaynağı](girdi/KAYNAK.md). 9.2 ayrıca 1000 × 1000 BMP oluşturup diskten tekrar okur. 9.3 kendi üç kanallı 8K görüntüsünü üretir.

Depo kökünden:

```powershell
python hafta9/9.2-bmp-gri/bmp_gri.py
python hafta9/9.3-8k-piksel-bolme/piksel_bolme.py
python hafta9/9.4-renk-uzayi-donusumu/renk_uzaylari.py
python hafta9/9.5-ten-rengi/ten_rengi.py
```

9.2 renkli ve gri görüntüyü ekranda gösterir; kapatmak için bir tuşa basılır. Pencere açmadan çalıştırmak için `--gosterme` eklenir. İki 8K BMP'nin her biri yaklaşık 95 MiB olduğundan Git'e eklenmez; 9.3 çalıştırıldığında yerelde kaydedilir. Gerçek ölçüm raporu ve küçültülmüş önizleme depodadır.

Kontrol:

```powershell
python -m pytest -q
```

OpenCV ve NumPy mevcut ortamdan kullanılır. Süreler çalıştırılan bilgisayara ve sistem yüküne bağlıdır. Ten rengi eşikleri bir örnektir; tüm ışık koşulları ve ten tonlarında aynı doğruluk beklenmez.
