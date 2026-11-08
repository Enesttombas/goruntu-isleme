# 7.1 — Vektörü sinüs değerleriyle doldurma

**Ödev:** Faz, genlik ve periyot ya da frekans alan, parametre olarak verilen vektörü sinüs değerleriyle dolduran bir fonksiyon. Vektör boyu 2'nin kuvveti. Oluşan vektör ve sinüzoidal dalga görselleştirilir.

`x[n] = A·sin(2π·f·n/fs + faz)`

- `sinus_doldur(vektor, genlik, faz, frekans=None, periyot=None, ornekleme_frekansi=None)` vektörü yerinde doldurur.
- Periyot verilirse `f = 1 / periyot`. İkisinden yalnız biri verilmeli.
- Örnekleme frekansı verilmezse `fs = N` alınır, yani vektör 1 saniyelik sinyal olur.
- Boy 2'nin kuvveti değilse `ValueError` verir (kontrol `n & (n - 1) == 0`).

| Dosya | Çıktı |
|---|---|
| [sinus.py](sinus.py) | `cikti/sinus.png` |

```powershell
python hafta7/7.1-sinus-vektor/sinus.py
```

Sonuç: grafikte turuncu noktalar vektörün elemanları, mavi çizgi aynı parametrelerle çizilmiş sürekli sinüs.

- N=64, A=2, f=3 Hz, faz=45°: ilk eleman `2·sin(45°) = 1.4142`, en büyük değer 2.0000, 1 saniyede 3 periyot.
- N=128, A=1, periyot 0.25 s: f = 4 Hz, 1 saniyede 4 periyot, faz 0 olduğu için ilk eleman 0.
- N=100 ile çağrınca: `Vektör boyu 2'nin kuvveti olmalı, gelen: 100`.
