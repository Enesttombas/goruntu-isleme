# 7.8 — Gauss ve tuz-biber gürültü, harmonik matris karşılaştırması

**Ödev:** Resme Gauss ve SaltAndPepper gürültü ekleyen iki ayrı fonksiyon. Fonksiyonlar gürültü kuvveti ve/veya oranını parametre alır. Gürültülü resimlerin harmonik matrisi görüntülenip orijinalinkiyle kıyaslanır.

- `gauss_gurultu_ekle(resim, sigma)`: her piksele `N(0, sigma²)` sayı eklenir, 0-255'e kırpılır. Burada σ = 25.
- `tuz_biber_ekle(resim, oran)`: piksellerin `oran` kadarı rastgele seçilir; yarısı 0 (biber), yarısı 255 (tuz) yapılır. Burada oran = 0.05.
- Harmonik matris 7.7'deki `dft2` ile. Tohumlar sabit, 7.9 ve 7.10 aynı gürültülü resimleri kullanıyor.

| Dosya | Çıktı |
|---|---|
| [resim_gurultu.py](resim_gurultu.py) | `cikti/karsilastirma.png` (üst: resimler, alt: harmonik matrisler), ayrıca her biri ayrı `orijinal`, `gauss`, `tuz_biber` ve `*_harmonik.png` |

```powershell
python hafta7/7.8-resim-gurultu/resim_gurultu.py
```

Sonuç:

| resim | yüksek frekans enerji payı (DC hariç, merkezden > N/4) | ortalama log(1+\|H\|) |
|---|---|---|
| orijinal | %1.19 | 8.25 |
| Gauss σ=25 | %8.93 | 9.27 |
| tuz-biber %5 | %15.30 | 9.53 |

Gürültünün kendi spektrumu `|H_gürültülü - H_orijinal|` iki gürültüde de bütün frekanslarda aynı seviyede: std/ortalama = 0.53, beyaz gürültünün genliği için beklenen Rayleigh değeri 0.52.

Yorum:

- Orijinalde enerji merkezde (düşük frekans), dışa doğru hızla azalıyor; tripod ve palto kenarlarından gelen çapraz çizgiler net görünüyor.
- Gürültü belli bir yapıya sahip olmadığı için bütün frekanslara eşit yayılıyor. Harmonik matrisin dış kısmı düz griye dönüyor, yüksek frekans payı 7.5-13 kat artıyor. Merkezdeki güçlü düşük frekanslar neredeyse değişmiyor, ince çapraz çizgiler ise gürültü tabanına gömülüyor.
- Tuz-biber pikselleri yalnız %5 ama her biri 0 ya da 255'e sıçradığı için toplam gürültü enerjisi Gauss σ=25'ten büyük (resimle MSE 1075'e karşı 564). Harmonik matriste de daha yüksek bir taban bırakıyor.
