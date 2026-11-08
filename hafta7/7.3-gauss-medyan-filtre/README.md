# 7.3 — Gürültülü sinyali Gauss ve medyan filtreyle temizleme

**Ödev:** 7.2'deki gürültülü veri sırasıyla Gauss ve medyan filtreyle temizlenir, sonuçlar gösterilip yorumlanır.

- `gauss_filtre(x, sigma)`: çekirdek `exp(-k²/2σ²)`, yarıçap `3σ`, toplamı 1 olacak şekilde normalize. Her eleman komşularının ağırlıklı ortalaması olur.
- `medyan_filtre(x, pencere)`: her eleman, pencere içindeki değerlerin ortanca değeri olur.
- İkisinde de kenarlar aynalanarak (`reflect`) uzatılıyor, çıktı boyu değişmiyor.

| Dosya | Çıktı |
|---|---|
| [filtreler.py](filtreler.py) | `cikti/gauss_medyan.png` |

```powershell
python hafta7/7.3-gauss-medyan-filtre/filtreler.py
```

Sonuç (oran 0.3, temiz sinyalle MSE):

| | MSE |
|---|---|
| gürültülü | 0.0290 |
| Gauss σ = 1 / 2 / **3** / 5 / 8 | 0.0078 / 0.0039 / **0.0028** / 0.0032 / 0.0090 |
| medyan 3 / 5 / 9 / **15** / 25 | 0.0169 / 0.0120 / 0.0069 / **0.0059** / 0.0094 |

Yorum:

- Gauss filtre hatayı yaklaşık 10 kat, medyan en iyi pencerede 5 kat azalttı. Bu gürültü düzgün dağılımlı, sıfır ortalamalı ve her örnekte küçük; ortalama almak böyle gürültüyü iyi götürüyor.
- Medyan filtre dürtü (tek tek sıçrayan) gürültüde güçlüdür. Burada ortanca değer de gürültülü örneklerden biri olduğu için çıktı basamaklı ve pürüzlü kalıyor, tepe 1.0 yerine 1.133'e çıktı (Gauss'ta 1.058).
- Pencere büyüdükçe gürültü azalıyor ama sinüsün kendisi de düzleşiyor: σ = 8 ve 25'lik medyanda tepeler basılıyor, hata yeniden artıyor. Pencere boyu sinyal periyoduna (burada 102 örnek) göre seçilmeli.
- Başta ve sonda küçük sapma var (t = 0'da Gauss 0.22): aynalama sinüsün devamını tam vermiyor.
