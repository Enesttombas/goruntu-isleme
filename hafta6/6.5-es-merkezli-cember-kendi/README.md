# 6.5 — Eş merkezli çemberleri bulan Hough dönüşümü, kendi kodumla

**Ödev:** Eş merkezli çemberleri bulan Hough tamamen kendimiz yazılır. Sin/cos baştan hesaplanır, 3B oylama matrisi kurulur, oylama açıkça gösterilir. Hocanın notu: oylama buna göre yapılmalı, post processing değil.

## Oylama nasıl eş merkezliliğe göre yapıldı

6.4'te her kenar pikseli 360 açının hepsine oy veriyordu; oylar merkez çevresinde halka şeklinde dağılır. Burada başka bir özelliği kullandım: **eş merkezli çemberlerin hepsinde kenar normali (gradyan yönü) aynı merkezden geçer.**

1. `sin_cos_tablosu`: 360 açı için cos ve sin bir kez hesaplanır.
2. `gradyan_aci_indeksi`: Sobel ile her pikselin gradyan açısı bulunur, tablodaki indekse çevrilir.
3. `es_merkezli_oylama`: her kenar pikseli **yalnız kendi gradyan doğrultusundaki** merkezlere oy verir: `a = x ∓ r·cos(φ)`, `b = y ∓ r·sin(φ)`. Açı hatasına ±2° tolerans var. İki yön de (∓) oylanır, çünkü halka zeminden açık da koyu da olabilir.

Böylece aynı merkezli bütün halkaların oyları 3B matriste **aynı (a, b) sütununa** düşer. Tek çember o sütunda tek tepe, eş merkezli küme birden çok tepe bırakır. Önce çemberleri ayrı ayrı bulup sonra merkezlerini karşılaştırma adımı yok; küme, oylama sonucunda doğrudan matristen okunur:

4. `merkezleri_bul`: merkez skoru = sütunda `oy / (2·pi·r) ≥ 0,3` olan oranların toplamı. Merkezin halkaları, sütundaki en az 6 px aralıklı tepelerdir.
5. En az 2 halkası olan merkez eş merkezli kümedir. Tek halkalı merkezler ince gri çizgiyle gösterilir, sayılmaz.

| Dosya | Çıktı |
|---|---|
| [kendi_fonksiyonum.py](kendi_fonksiyonum.py) | `cikti/kenar_canny.png`, `cikti/oylama_merkez_skoru.png`, `cikti/cemberler_es_merkezli.png` |

```powershell
python hafta6/6.5-es-merkezli-cember-kendi/kendi_fonksiyonum.py
```

## Sonuç

Test görüntüsü sentetik, doğru cevap biliniyor ([KAYNAK.md](../girdi/KAYNAK.md)). Matris 400 × 520 × 101 (yarıçap 10-110).

| Beklenen | Bulunan |
|---|---|
| Küme: (150, 190), r = 35, 65, 95 | (150, 190), r = 36, 66, 96 |
| Küme: (390, 150), r = 20, 45, 70 | (390, 150), r = 21, 46, 71 |
| Tek çember (380, 315), r = 50 | tek halka, eş merkezli sayılmadı |
| Tek çember (270, 330), r = 30 | tek halka, eş merkezli sayılmadı |

Merkezler tam doğru. Yarıçaplar 1 px büyük, çünkü Canny kenarı dolu dairenin dış sınırındaki piksele koyuyor. Halka eşiği 0,2-0,3 arasında değiştirildiğinde sonuç aynı kaldı. `oylama_merkez_skoru.png`'de eş merkezli merkezler en parlak iki nokta; doğru parçası ve tek çemberler sönük kalıyor.
