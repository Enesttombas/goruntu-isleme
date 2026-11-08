# 7.7 — 2 boyutlu DFT

**Ödev:** Girdi olarak resim alan ve harmonik matrisini döndüren 2 boyutlu DFT fonksiyonu.

`H[u, v] = Σx Σy f[x, y]·e^(-j2π(ux/M + vy/N))`

- 2B DFT ayrılabilir: üstel ifade `e^(-j2πux/M)·e^(-j2πvy/N)` diye ikiye ayrılıyor. Bu yüzden önce her satırın 1B DFT'si, sonra çıkan matrisin her sütununun 1B DFT'si alınıyor.
- 1B dönüşüm 7.4'teki kendi `fft` fonksiyonumuz; satırların hepsi tek çağrıda dönüştürülüyor. Resim boyutları 2'nin kuvveti olmalı.
- `idft2`: aynı sırayla `ifft`.
- `spektrum_resmi`: `log(1 + |H|)`, DC ortaya alınarak (`merkeze_al`) 0-255'e ölçekleniyor. Log olmadan DC dışındaki her şey siyah görünüyor.

| Dosya | Çıktı |
|---|---|
| [dft2.py](dft2.py) | `cikti/harmonik_matris.png`, `cikti/resim_ve_harmonik_matris.png` |

```powershell
python hafta7/7.7-2b-dft/dft2.py
```

Sonuç (`girdi/kamera.png`, 512×512):

- Harmonik matris 512×512 karmaşık sayı. `np.fft.fft2` ile bağıl fark 1.2e-16, `idft2(dft2(resim))` ile resim arasındaki en büyük fark 1.4e-13.
- `H[0, 0] = 33 832 495` = piksellerin toplamı. DC bileşeni, ortalama parlaklık × piksel sayısı.

Harmonik matrisini incelerken: enerji ortadaki düşük frekanslarda toplanıyor (resmin büyük kısmı gökyüzü ve çimen gibi yavaş değişen bölgeler). Tripod bacaklarına ve paltonun kenarına dik yönde parlak çapraz çizgiler var; keskin bir kenar, kendisine dik doğrultuda bütün frekanslara enerji verir. Yatay ve dikey eksen çizgileri resmin kenarlarından geliyor: DFT resmi periyodik sayar, sağ kenarla sol kenar ve üst ile alt arasındaki parlaklık sıçraması da kenar gibi davranır.
