# 7.9 — Ters DFT ile orijinale dönme, ZNCC ve benzerlik metrikleri

**Ödev:** Gürültü eklenmiş resimler üzerinde ters DFT ile orijinal görüntüye dönmeye çalışılır. Orijinal ile yeniden üretilen görüntü arasındaki fark Zero Mean Normalized Cross Correlation ile ölçülür. İki piksel değerinin yakınlığını ölçen başka metrikler ve birbirlerine göre avantaj/dezavantajları raporlanır.

`ZNCC(a, b) = Σ(a - ā)(b - b̄) / √(Σ(a - ā)² · Σ(b - b̄)²)`, sonuç [-1, 1], 1 = birebir benzer.

- Gürültülü resmin harmonik matrisini olduğu gibi ters çevirmek gürültülü resmi aynen verir (DFT kayıpsız). Orijinale yaklaşmak için ters dönüşümden önce harmonik matris Gauss alçak geçiren maskeyle `exp(-D²/2D0²)` çarpılıyor; gürültü enerjisinin çoğu yüksek frekansta (7.8).
- D0 için 10-200 arası denendi, ZNCC'yi en yüksek yapan seçildi.
- MSE, PSNR ve SSIM de kendi kodumla hesaplanıyor (SSIM: Wang vd. 2004, 11×11 Gauss pencere, σ=1.5). Kontrol: ZNCC `np.corrcoef` ile aynı, SSIM scikit-image ile ±0.002.

| Dosya | Çıktı |
|---|---|
| [ters_dft.py](ters_dft.py) | `cikti/gurultulu_dogrudan_filtreli.png` (soldan: gürültülü, doğrudan ters DFT, maskeli ters DFT), `cikti/zncc_d0.png`, `cikti/*_ters_dft_d0_*.png` |

```powershell
python hafta7/7.9-ters-dft-zncc/ters_dft.py
```

## Sonuç

| resim | yöntem | ZNCC | MSE | PSNR (dB) | SSIM |
|---|---|---|---|---|---|
| Gauss σ=25 | gürültülü resim | 0.9501 | 564.3 | 20.62 | 0.290 |
| | doğrudan ters DFT | 0.9501 | 564.3 | 20.62 | 0.290 |
| | alçak geçiren D0=80 + ters DFT | **0.9884** | 128.1 | 27.05 | 0.653 |
| tuz-biber %5 | gürültülü resim | 0.9066 | 1074.7 | 17.82 | 0.354 |
| | doğrudan ters DFT | 0.9066 | 1074.7 | 17.82 | 0.354 |
| | alçak geçiren D0=60 + ters DFT | **0.9845** | 182.1 | 25.53 | 0.634 |

- Doğrudan ters DFT ile gürültülü resim arasındaki en büyük fark 0: ters dönüşüm tek başına gürültüyü gidermez, sadece aynı resmi geri verir.
- Maskeyle ZNCC Gauss'ta 0.950 → 0.988, tuz-biberde 0.907 → 0.985 oldu. D0 küçükse (10) resim fazla bulanıklaşıyor, büyükse (200) gürültü geri geliyor; ZNCC arada tepe yapıyor (`zncc_d0.png`).
- Tuz-biberde en iyi D0 daha küçük çıktı, çünkü bu gürültünün enerjisi daha büyük. Yine de siyah-beyaz noktalar yok olmuyor, gri lekelere dönüşüyor; bu gürültü için uzamsal medyan filtre daha uygun.

## Piksel yakınlığını ölçen metrikler

| metrik | tanım | avantaj | dezavantaj |
|---|---|---|---|
| MSE / RMSE | farkların karesinin ortalaması (RMSE kökü) | basit, hızlı, türevi alınabilir, birimi piksel | algıyla ilişkisi zayıf; parlaklık kayması gibi göze önemsiz değişimi büyük hata sayar, büyük hatalar kare yüzünden baskın |
| MAE | mutlak farkların ortalaması | aykırı piksellere (tuz-biber) MSE'den daha az duyarlı | yine algısal değil, yapıyı bilmez |
| PSNR | `10·log10(255² / MSE)` dB | alanda standart, sıkıştırma ve gürültü giderme çalışmalarında karşılaştırması kolay | MSE'nin log hali, aynı zayıflıklar; farklı içerikli resimler arasında anlamlı değil |
| NCC | `Σab / √(Σa²·Σb²)` | kontrast (çarpım) değişimine duyarsız | parlaklık eklemeye (ofset) duyarlı |
| ZNCC | ortalaması çıkarılmış NCC | hem parlaklık hem kontrast değişimine duyarsız, [-1, 1] yorumu kolay; şablon eşleme ve stereo eşlemede standart | bu duyarsızlık bazen kusur: `0.5·resim + 20` için ZNCC 1.0000 ama MSE 3361, SSIM 0.823. Tek global sayı, yerel bozulmayı göstermez; düz bölgede std ≈ 0 olduğu için kararsız |
| SSIM | yerel pencerelerde parlaklık, kontrast ve yapı karşılaştırması | insan algısına en yakın; gürültüyü ve bulanıklığı yerel olarak yakalar | hesabı ağır, parametreli (pencere, C1, C2), değer doğrusal değil |
| Karşılıklı bilgi (MI) | iki resmin ortak histogramından bilgi miktarı | parlaklıklar arasında doğrusal ilişki gerektirmez, farklı kaynaklı (MR-BT) resim hizalamada kullanılır | histogram kutu sayısına bağlı, uzamsal yapıyı bilmez, yavaş |

Bu ölçümlerde metriklerin farkı görünüyor: Gauss gürültülü resimde ZNCC 0.95 iken SSIM 0.29. ZNCC resmin genel olarak aynı olduğunu söylüyor, SSIM ise her küçük pencerede yapının bozulduğunu yakalıyor.
