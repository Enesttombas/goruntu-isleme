# 6.2 — Doğru bulan Hough dönüşümü, kendi kodumla

**Ödev:** Hough dönüşümü tamamen kendimiz yazılır. Sin ve cos değerleri baştan hesaplanıp bellekte tutulur. Oylama matrisi oluşturulur, oylama açıkça gösterilir, matris resim olarak kaydedilir. Eşiği geçen doğrular orijinal resme çizilir.

Doğru denklemi `rho = x·cos(theta) + y·sin(theta)`.

1. `sin_cos_tablosu`: 180 theta (1° adım) için cos ve sin bir kez hesaplanır.
2. `hough_oylama`: her kenar pikseli, her theta için rho'yu bulur ve `oylama[rho + köşegen, theta]` hücresine 1 oy ekler. Negatif rho için satırlar köşegen kadar kaydırılır.
3. `tepeleri_bul`: oyu eşiği (100) geçen ve 3x3 komşuluğunun en büyüğü olan hücreler doğrudur.

| Dosya | Çıktı |
|---|---|
| [kendi_fonksiyonum.py](kendi_fonksiyonum.py) | `cikti/oylama_matrisi.png`, `cikti/dogrular_kendi.png` |

```powershell
python hafta6/6.2-hough-line-kendi/kendi_fonksiyonum.py
```

Sonuç: oylama matrisi 1587 (rho) × 180 (theta), en yüksek oy 209. Eşik 100 ile 14 doğru bulundu; OpenCV `HoughLines` aynı kenar resminde 13 buldu.

Oylama matrisini incelerken: her kenar pikseli matriste bir sinüs eğrisi çizer. Aynı doğrudaki piksellerin eğrileri tek noktada kesişir; parlak noktalar bulunan doğrulardır. Yatay doğrular theta = 90°, dikey doğrular theta ≈ 0° ve 175-178° sütunlarında toplanıyor. Resim, az oylar da görünsün diye karekök ölçekle kaydedildi.
