# 7.5 — İki gürültülü sinyal ve toplamlarının DFT analizi

**Ödev:** Frekansı 50 ve genliği 12 olan, frekansı 30 ve genliği 5 olan, gürültü eklenmiş iki sinyal önce ayrı ayrı DFT ile incelenir. Sonra toplanır; toplam sinyal ve DFT sonucu elde edilen harmonik vektörü görselleştirilir.

- N = 1024 örnek, fs = 1024 Hz (1 saniye). Frekans çözünürlüğü fs/N = 1 Hz, 30 ve 50 Hz tam birer harmoniğe düşüyor.
- Sinyaller 7.1'deki `sinus_doldur` ile üretiliyor, 7.2'deki `gurultu_ekle` ile 0.2 oranında gürültü ekleniyor (s1 için ±2.4, s2 için ±1.0). DFT 7.4'teki kendi `fft` fonksiyonumuz.

| Dosya | Çıktı |
|---|---|
| [iki_sinyal.py](iki_sinyal.py) | `cikti/iki_sinyal_ve_toplam.png` |

```powershell
python hafta7/7.5-iki-sinyal-toplam/iki_sinyal.py
```

Sonuç (tek taraflı genlik):

| sinyal | 30 Hz | 50 Hz | diğer harmonikler (ortalama / en büyük) |
|---|---|---|---|
| s1 (50 Hz, A=12) | 0.134 | 12.074 | 0.076 / 0.251 |
| s2 (30 Hz, A=5) | 5.008 | 0.007 | 0.032 / 0.099 |
| s1 + s2 | 5.025 | 12.067 | 0.082 / 0.281 |

`max|DFT(s1 + s2) - (DFT(s1) + DFT(s2))| = 9.3e-13`

Yorum:

- Zamanda toplam sinyal düzensiz bir dalga gibi görünüyor, iki sinüsü gözle ayırmak zor. Frekans uzayında ise iki ayrı çizgi: 30 Hz'te 5, 50 Hz'te 12. Genlikler doğrudan tepe yüksekliğinden okunuyor.
- DFT doğrusal: toplamın harmonik vektörü, ayrı ayrı harmonik vektörlerin toplamına sayısal hata seviyesinde eşit.
- Gürültü genliğe oranlı eklendiği için s1'in gürültü tabanı s2'ninkinden yüksek. Toplamda iki sinyalin gürültüsü birleşiyor ama taban (en büyük 0.28) yine 5'lik tepenin çok altında.
