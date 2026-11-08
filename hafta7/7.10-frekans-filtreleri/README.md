# 7.10 — Alçak, yüksek, bant geçiren ve bant durduran filtreler

**Ödev:** Bir görüntü ve gürültülü halleri üzerinde low-pass, high-pass, band-pass ve band-reject filtreler geliştirilip kullanılır. Sonuçların gürültülere, kenar geçişlerine ve genel parlaklığa etkisi yorumlanır.

Filtreler frekans uzayında maske; harmonik matris maskeyle çarpılıp ters DFT alınıyor (7.7'deki `dft2` / `idft2`). D, harmoniğin DC'ye uzaklığı:

| filtre | maske | parametre |
|---|---|---|
| alçak geçiren | `exp(-D² / 2D0²)` | D0 = 40 |
| yüksek geçiren | `1 - alçak geçiren` | D0 = 40 |
| bant geçiren | `exp(-((D² - C0²) / (D·W))²)` | C0 = 60, W = 40 |
| bant durduran | `1 - bant geçiren` | C0 = 60, W = 40 |

Maskeler Gauss biçimli seçildi; ideal (keskin kesimli) maske ters DFT'de kenarların etrafında halka (ringing) yapar. Girdiler 7.8'deki orijinal, Gauss σ=25 ve tuz-biber %5 resimleri.

| Dosya | Çıktı |
|---|---|
| [frekans_filtreleri.py](frekans_filtreleri.py) | `cikti/maskeler.png`, `cikti/filtre_sonuclari.png` (satırlar: orijinal, Gauss, tuz-biber) |

```powershell
python hafta7/7.10-frekans-filtreleri/frekans_filtreleri.py
```

Yüksek ve bant geçiren sonuçlar 0 etrafında artı/eksi değerler; resimde kenarlar ve gürültü görünsün diye +128 kaydırılarak gösterildi.

## Sonuç

- **ort. parlaklık:** çıktının ortalaması.
- **kenar gücü %:** ortalama Sobel gradyan büyüklüğü, orijinal resmin değerine oranla.
- **gürültü RMSE:** aynı filtrenin gürültülü resme ve temiz resme uygulanmış sonuçları arasındaki fark, yani filtreden geçen gürültü.

| girdi | filtre | ort. parlaklık | kenar gücü % | gürültü RMSE |
|---|---|---|---|---|
| orijinal | filtresiz | 129.1 | 100 | 0 |
| | alçak geçiren | 129.1 | 43 | 0 |
| | yüksek geçiren | 0.0 | 86 | 0 |
| | bant geçiren | 0.0 | 55 | 0 |
| | bant durduran | 129.1 | 92 | 0 |
| Gauss σ=25 | filtresiz | 129.2 | 252 | 23.8 |
| | alçak geçiren | 129.2 | 52 | **3.8** |
| | yüksek geçiren | 0.0 | 234 | 23.0 |
| | bant geçiren | 0.0 | 80 | 4.6 |
| | bant durduran | 129.2 | 231 | 22.9 |
| tuz-biber %5 | filtresiz | 128.9 | 231 | 32.8 |
| | alçak geçiren | 128.9 | 56 | **5.7** |
| | yüksek geçiren | 0.0 | 222 | 31.6 |
| | bant geçiren | 0.0 | 90 | 6.3 |
| | bant durduran | 128.9 | 232 | 31.6 |

## Yorum

**Gürültü:** Gürültü enerjisinin çoğu yüksek frekansta. Alçak geçiren gürültüyü en çok azaltan filtre (Gauss'ta 23.8 → 3.8, tuz-biberde 32.8 → 5.7). Yüksek geçiren ve bant durduran yüksek frekansları geçirdiği için gürültüyü neredeyse olduğu gibi bırakıyor. Bant geçiren yalnız dar bir halkayı geçirdiği için az gürültü geçiriyor, ama resmin de çoğunu atıyor. Tuz-biber alçak geçirenden sonra yok olmuyor, gri lekelere dönüşüyor.

**Kenar geçişleri:** Kenarlar ani değişim, yani yüksek frekans. Alçak geçiren kenarları yumuşatıyor (kenar gücü %43, resim bulanık). Yüksek geçiren düz bölgeleri silip yalnız kenar ve dokuyu bırakıyor (%86). Bant geçiren orta frekansları tutuyor: kalın kenar konturları kalıyor, ince doku ve gürültü gidiyor (%55). Bant durduran orta bandı çıkardığı için resim orijinale çok yakın (%92), kenarların yanında hafif bir gölge oluşuyor. Gürültülü resimlerde yüksek geçiren ve bant durduranda kenar gücü %220'nin üstünde; ölçülen "kenar"ın çoğu gürültü.

**Genel parlaklık:** Ortalama parlaklık DC bileşeninde (`H[0, 0]`). Alçak geçiren ve bant durduranın maskesi DC'de 1, ortalama 129 aynen kalıyor. Yüksek ve bant geçirenin maskesi DC'de 0, ortalama 0'a iniyor: gökyüzü, palto gibi geniş düz bölgeler aynı griye iniyor, yalnız değişimler kalıyor. Parlaklığı korurken kenar vurgulamak için yüksek geçiren sonucu orijinale eklenebilir (keskinleştirme).
