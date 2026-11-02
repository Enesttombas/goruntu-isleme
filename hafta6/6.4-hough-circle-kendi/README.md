# 6.4 — Çember bulan Hough dönüşümü, kendi kodumla

**Ödev:** Çember için Hough tamamen kendimiz yazılır. Sin ve cos değerleri baştan hesaplanıp bellekte tutulur. 3 boyutlu oylama matrisi oluşturulur, oylama açıkça gösterilir. Eşiği geçen çemberler orijinal resme çizilir.

Çember denklemi: `a = x − r·cos(theta)`, `b = y − r·sin(theta)`.

1. `sin_cos_tablosu`: 360 açı için cos ve sin bir kez hesaplanır. `merkez_kaydirmalari` her (r, theta) için `r·cos` ve `r·sin` değerlerini de bir kez hesaplar.
2. `hough_cember_oylama`: 3B matris `oylama[b, a, r]`. Her kenar pikseli, her yarıçap ve her açı için olası merkezi bulur ve o hücreye 1 oy verir. Aynı piksel aynı hücreye bir kez oy verir.
3. `cemberleri_sec`: büyük çember daha çok piksel taşır, bu yüzden eşik oranladır: `oy / (2·pi·r) ≥ 0,20`. Güçlüden zayıfa sıralanır, 20 pikselden yakın merkezli tekrarlar atılır.

| Dosya | Çıktı |
|---|---|
| [kendi_fonksiyonum.py](kendi_fonksiyonum.py) | `cikti/oylama_matrisi_merkezler.png`, `cikti/cemberler_kendi.png` |

```powershell
python hafta6/6.4-hough-circle-kendi/kendi_fonksiyonum.py
```

Sonuç: matris 356 × 413 × 21 (yarıçap 20-40). 12 çember bulundu, hepsi şekerlere oturuyor; yarıçaplar 25-27. Kenar resmi 6.3 ile aynı (Canny + median). `oylama_matrisi_merkezler.png` her (a, b) merkezinin en güçlü yarıçaptaki oyunu gösterir.

Eşik denemesi: 0,15 ile 14 çember çıktı (2'si yanlış), 0,25 ile 5'e düştü.
