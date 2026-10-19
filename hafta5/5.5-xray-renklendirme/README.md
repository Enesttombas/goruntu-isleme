# 5.5 — X-ray görüntülerini griye çevirip renklendirme

**Ödev:** HUMS X-ray veri setinin `ThreatsRGB` klasöründeki 5 görüntü önce gri tonlu yapılır, sonra güvenlik operatörünün daha rahat anlayacağı şekilde renklendirilir. Palet seçimi açıklanır, orijinalle fark yorumlanır.

| Dosya | Yöntem | Çıktı |
|---|---|---|
| [opencv_ile.py](opencv_ile.py) | OpenCV `cvtColor` + `applyColorMap` | `cikti/*_gri_opencv.png`, `*_renkli_opencv.png` |
| [kendi_fonksiyonum.py](kendi_fonksiyonum.py) | Kendi gri fonksiyonu + kendi palet eşlemesi, orijinalle ölçüm | `cikti/*_gri_kendi.png`, `*_renkli_kendi.png`, `*_karsilastirma.png`, [olcumler.txt](cikti/olcumler.txt) |

Palet ortak dosyada: [xray_palet.py](../xray_palet.py). İki yöntemin çıktıları piksel piksel aynıdır. Yan yana görseller `*_karsilastirma.png` (orijinal | gri | bizim renklendirme).

```powershell
python hafta5/5.5-xray-renklendirme/opencv_ile.py
python hafta5/5.5-xray-renklendirme/kendi_fonksiyonum.py
```

Girdi: `ThreatsRGB` klasöründen ada göre sıralı ilk 5 uygun görüntü (30 KB üstü): `image_0_1000, 1072, 1196, 1212, 1222`.

## Hangi paleti seçtim, neden

Gri görüntüde parlak = boş, koyu = yoğun. Her gri değeri bir renge eşlenir:

| Yoğunluk | Renk | Anlam |
|---|---|---|
| boş | beyaz | zemin dikkat dağıtmasın |
| düşük | krem → turuncu | organik madde (yiyecek, kumaş, plastik) |
| orta | zeytin → yeşil | karışık |
| yüksek | mavi | metal, inorganik |
| çok yüksek | koyu mavi → lacivert | çok yoğun, çok dikkat gerektirir |

Nedenleri:
1. **Sektör kuralı:** Bagaj X-ray cihazlarında turuncu organik, yeşil karışık, mavi metal gösterilir. Operatör bu alışkanlıkla okur, yeni bir renk dilini öğrenmez.
2. **Parlaklık yoğunlukla birlikte azalır.** Yoğun nesne hem mavi hem koyu olur; renk körü bir operatör de koyuluktan yoğunluğu okuyabilir. Gökkuşağı (JET) paleti bunu bozar.
3. **Boş zemin beyaz kalır,** nesneler öne çıkar.
4. **Palet ayarı:** İlk denememde turuncuyu çok yoğun, maviyi çok geç başlattım. Orijinal görüntülerde turuncu pikseller gri ≈ 190-210, mavi ≈ 40-145 aralığındaydı. Durakları buna göre kaydırdım; sınıf uyumu belirgin arttı (örn. `image_0_1000`: 0.13 → 0.35). Bu kalibrasyon yalnızca bu 5 görüntüden yapıldı.

## Orijinalle fark ve ölçüm

Orijinaller zaten sahte renklidir: cihaz iki enerjide (yüksek/düşük) çekim yapar, iki değerin oranına göre malzeme rengi verir. Ben ise tek bir gri kanaldan renklendirdim. Ölçümler ([olcumler.txt](cikti/olcumler.txt)):

| Görüntü | Ort. mutlak fark | Ort. ΔE (Lab) | Malzeme sınıfı uyumu | Turuncu | Yeşil | Mavi |
|---|---|---|---|---|---|---|
| image_0_1000 | 22.9 | 23.9 | 0.35 | 0.96 | 0.10 | 0.37 |
| image_0_1072 | 22.6 | 25.8 | 0.67 | 0.71 | 0.24 | 0.95 |
| image_0_1196 | 24.3 | 27.6 | 0.69 | 0.70 | 0.23 | 0.95 |
| image_0_1212 | 17.1 | 19.7 | 0.80 | 0.95 | 0.17 | 0.96 |
| image_0_1222 | 20.5 | 23.8 | 0.65 | 0.67 | 0.20 | 0.97 |

"Sınıf uyumu": orijinalde renk taşıyan nesne piksellerinde, orijinaldeki renk sınıfını (turuncu/yeşil/mavi) benim çıktımın da verme oranı. Ortalama genel uyum ≈ 0.63. ΔE 20-28 büyük bir fark demektir, yani renkler birebir aynı değil; amaç zaten kopya değil, okunaklılıktı.

## Ne kadar başarılı, nerede zayıfım

**İyi olanlar**
- Metal ve çok yoğun nesneler (bıçak, tabanca benzeri koyu bloklar, kilitler) mavi/lacivert olarak net ayrılıyor; mavi uyumu 4 görüntüde %95-97.
- Organik içerik turuncu kalıyor (turuncu uyumu %67-96).
- Boş zemin sade, nesne konturları korunuyor.

**Zayıf olanlar**
- **Yeşil (karışık) sınıfı neredeyse hiç tutmuyor (%10-24).** Orijinaldeki yeşil, gri yoğunluk bakımından hem turuncuyla hem maviyle örtüşüyor. Tek kanaldan malzeme ayırt edilemez; ben yalnızca yoğunluğa göre renk veriyorum.
- **Malzemeyi değil yoğunluğu boyuyorum.** Ince ama metal olan bir bıçak (`image_0_1000`) az yoğun göründüğü için mavi yerine yeşil/zeytin çıkıyor (mavi uyumu %37). Kalın bir organik madde de mavimsi çıkabilir. Bu, yanlış güven verebilir; ölçüm sonucu bu yüzden tek başına karar için kullanılmamalı.
- **Tarayıcı çizgileri ve gürültü** (orijinalde macenta şeritler) benim çıktımda turuncu şerit/benek olarak kalıyor; operatörü gereksiz uyarabilir.
- Palet 5 görüntüden ayarlandı, başka görüntülerde aynı başarıyı garanti etmem.

**Nasıl iyileşir:** Veri setinde yüksek ve düşük enerjili çekimler (`HighLow` klasörü) da var. İkisinin oranı malzemeyi gerçekten ayırır; renklendirme tek gri yerine bu orana dayansa yeşil ve ince metal sorunları büyük ölçüde çözülür. Bu ödevde gri kullanılması istendiği için yapılmadı.

## Veri seti

HUMS X-ray Dataset (Hacettepe Üniversitesi ve MS Spektral), [Kaggle kopyası](https://www.kaggle.com/datasets/orvile/hums-x-ray-dataset), CC BY 4.0. Görüntüler yazarların GitHub deposundan ([ElifErarslan1/HUMS-X-ray-Dataset](https://github.com/ElifErarslan1/HUMS-X-ray-Dataset)) alındı (Kaggle indirme için giriş istiyor; dosya sayısı aynı, 741). Atıf: E. Erarslan Altındağ, S. E. Yüksel, "Threat detection in x-ray baggage security imagery using convolutional neural networks," Proc. SPIE 12104, ADIX VII, 121040H (2022), https://doi.org/10.1117/12.2622373.
