# 5.5 — X-ray görüntülerini griye çevirip renklendirme

**Ödev:** HUMS X-ray veri setinin `ThreatsRGB` klasöründeki 5 görüntü önce gri tonlu yapılır, sonra güvenlik operatörünün daha rahat anlayacağı şekilde renklendirilir. Palet seçimi açıklanır, orijinalle fark yorumlanır.

Kullanılan 5 görüntü: `image_0_1000`, `image_0_1054`, `image_0_1071`, `image_0_1072`, `image_0_1212` ([girdi](../girdi/ThreatsRGB/)).

| Dosya | Yöntem | Çıktı |
|---|---|---|
| [opencv_ile.py](opencv_ile.py) | OpenCV `cvtColor` + `applyColorMap` | `cikti/opencv/*_gri.png`, `*_renkli.png` |
| [kendi_fonksiyonum.py](kendi_fonksiyonum.py) | Kendi gri fonksiyonu + kendi palet eşlemesi, orijinalle ölçüm | `cikti/kendi/*_gri.png`, `*_renkli.png`, `*_karsilastirma.png`, [olcumler.txt](cikti/olcumler.txt) |

Palet ortak dosyada: [xray_palet.py](../xray_palet.py). İki yöntemin çıktıları piksel piksel aynıdır. `*_karsilastirma.png` yan yana orijinal | gri | bizim renklendirme gösterir.

```powershell
python hafta5/5.5-xray-renklendirme/opencv_ile.py
python hafta5/5.5-xray-renklendirme/kendi_fonksiyonum.py
```

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
4. **Palet ayarı:** İlk denememde turuncuyu çok yoğun, maviyi çok geç başlattım. Orijinal görüntülerde turuncu pikseller gri ≈ 190-210, mavi ≈ 40-145 aralığındaydı. Durakları buna göre kaydırdım. Bu ayar için `image_0_1000, 1072, 1196, 1212, 1222` görüntülerine baktım; `1054` ve `1071` ayara hiç katılmadı.

## Orijinalle fark ve ölçüm

Orijinaller zaten sahte renklidir: cihaz iki enerjide (yüksek/düşük) çekim yapar, iki değerin oranına göre malzeme rengi verir. Ben ise tek bir gri kanaldan renklendirdim. Ölçümler ([olcumler.txt](cikti/olcumler.txt)):

| Görüntü | Boyut | Ort. mutlak fark | Ort. ΔE (Lab) | Malzeme sınıfı uyumu | Turuncu | Yeşil | Mavi |
|---|---|---|---|---|---|---|---|
| image_0_1000 | 789×478 | 22.9 | 23.9 | 0.35 | 0.96 | 0.10 | 0.37 |
| image_0_1054 | 151×429 | 10.2 | 13.3 | 0.86 | 1.00 | 0.10 | 0.86 |
| image_0_1071 | 149×266 | 8.8 | 12.0 | 0.91 | — | — | 0.91 |
| image_0_1072 | 581×478 | 22.6 | 25.8 | 0.67 | 0.71 | 0.24 | 0.95 |
| image_0_1212 | 793×584 | 17.1 | 19.7 | 0.80 | 0.95 | 0.17 | 0.96 |
| **Tümü (piksel toplamı)** | | 16.3 | 18.9 | **0.64** | 0.90 | 0.18 | 0.68 |

"Sınıf uyumu": orijinalde renk taşıyan nesne piksellerinde, orijinaldeki renk sınıfını (turuncu/yeşil/mavi) benim çıktımın da verme oranı. "—": orijinalde o renkte nesne piksel yok. ΔE 12-26 aralığında; renkler birebir aynı değil, ama amaç kopya değil okunaklılıktı.

## Ne kadar başarılı, nerede zayıfım

**İyi olanlar**
- **Tek metal nesne, boş zemin** (`1054`, `1071` bıçak kesitleri): uyum %86-91, ΔE en düşük. Bıçak mavi/lacivert olarak net ayrılıyor, kabzanın hafif organik kısmı turuncu kalıyor.
- **Karmaşık bagajlarda metal** (`1072`, `1212`): mavi uyumu %95-96. Kilitler, bıçaklar, koyu bloklar öne çıkıyor.
- **Organik içerik** turuncu kalıyor (turuncu uyumu %71-100).
- Boş zemin sade, nesne konturları korunuyor.

**Zayıf olanlar**
- **Yeşil (karışık) sınıfı neredeyse hiç tutmuyor (%10-24).** Orijinaldeki yeşil, gri yoğunluk bakımından hem turuncuyla hem maviyle örtüşüyor. Tek kanaldan malzeme ayırt edilemez; ben yalnızca yoğunluğa göre renk veriyorum.
- **Malzemeyi değil yoğunluğu boyuyorum.** `image_0_1000`'deki ince bıçaklar az yoğun göründüğü için mavi yerine yeşil/zeytin çıktı (mavi uyumu %37, en kötü sonuç). Kalın bir organik madde de mavimsi çıkabilir; bu, operatöre yanlış güven verebilir.
- **Tarayıcı çizgileri ve gürültü** (orijinalde macenta şeritler, `1054`'te yatay çizgiler, `1212`'de kalın şeritler) çıktımda turuncu şerit/benek olarak kalıyor; operatörü gereksiz uyarabilir.
- **Küçük örneklem:** 5 görüntü, ikisi basit bıçak kesiti. `1054` ve `1071`'in yüksek skoru kolay vakalar olmalarından da geliyor; gerçek bagajlardaki (1000, 1072, 1212) başarı daha düşük ve değişken. Palet de başka görüntülerde aynı başarıyı garanti etmez.

**Nasıl iyileşir:** Veri setinde yüksek ve düşük enerjili çekimler (`HighLow` klasörü) da var. İkisinin oranı malzemeyi gerçekten ayırır; renklendirme tek gri yerine bu orana dayansa yeşil ve ince metal sorunları büyük ölçüde çözülür. Bu ödevde gri kullanılması istendiği için yapılmadı.

## Veri seti

HUMS X-ray Dataset (Hacettepe Üniversitesi ve MS Spektral), [Kaggle kopyası](https://www.kaggle.com/datasets/orvile/hums-x-ray-dataset), CC BY 4.0. Görüntüler yazarların GitHub deposundan ([ElifErarslan1/HUMS-X-ray-Dataset](https://github.com/ElifErarslan1/HUMS-X-ray-Dataset)) alındı (Kaggle indirme için giriş istiyor; dosya sayısı aynı, 741). Atıf: E. Erarslan Altındağ, S. E. Yüksel, "Threat detection in x-ray baggage security imagery using convolutional neural networks," Proc. SPIE 12104, ADIX VII, 121040H (2022), https://doi.org/10.1117/12.2622373.
