# 7.6 — Eşik altındaki harmonikleri silen frekans filtresi

**Ödev:** Harmonik vektörde belli bir eşiğin altında kalan tüm harmonikleri filtreleyip orijinal sinyali yeniden üreten frekans filtresi. 5. görevdeki sinyaller ve toplamları için test edilir.

`frekans_filtresi(x, esik)`:

1. Harmonik vektör bulunur (7.4'teki `fft`).
2. Her harmoniğin genliği `|X[k]|·2/N` ile sinüs genliği birimine çevrilir (DC ve Nyquist için `/N`).
3. Genliği eşikten küçük harmonikler 0 yapılır.
4. Ters DFT (`ifft`) ile sinyal geri üretilir.

Gerçek sinyalde `X[k]` ile `X[N-k]` aynı genliktedir, ikisi birlikte silinir ya da kalır; sonuç gerçek sayı olur.

| Dosya | Çıktı |
|---|---|
| [frekans_filtresi.py](frekans_filtresi.py) | `cikti/frekans_filtresi.png`, `cikti/esik_secimi.png` |

```powershell
python hafta7/7.6-frekans-filtresi/frekans_filtresi.py
```

Sonuç (eşik 1.0, RMSE gürültüsüz sinyale göre):

| sinyal | kalan harmonik | RMSE gürültülü | RMSE filtreli |
|---|---|---|---|
| s1 (50 Hz, A=12) | 2 | 1.389 | 0.069 |
| s2 (30 Hz, A=5) | 2 | 0.574 | 0.007 |
| s1 + s2 | 4 | 1.492 | 0.120 |

Toplam sinyalde eşik denemesi:

| eşik | 0.05 | 0.1 | 0.3 – 4.0 | 6.0 | 13.0 |
|---|---|---|---|---|---|
| kalan harmonik | 756 | 320 | 4 | 2 | 0 |
| RMSE | 1.465 | 1.240 | 0.120 | 3.536 | 9.192 |

Yorum:

- Filtre gürültüyü büyük ölçüde siliyor; filtreli sinyal grafikte orijinalin üstüne oturuyor. Kalan küçük hata, tutulan harmoniklerin içindeki gürültü payı (o frekanstaki gürültü harmonikten ayrılamıyor).
- Eşik, gürültü tabanından (en büyük 0.28) büyük ve tutulmak istenen en küçük harmonikten (5) küçük olmalı. 0.3 ile 4 arasındaki her eşik aynı sonucu veriyor.
- Eşik çok düşükse gürültü harmonikleri kalıyor (0.1'de 320 tane). Eşik 6 olunca 5 genlikli 30 Hz bileşeni de siliniyor: hata 5/√2 = 3.54, yani tam o sinüsün RMS değeri. Eşik 13'te hiçbir şey kalmıyor, hata toplam sinyalin RMS'i √(12²/2 + 5²/2) = 9.19.
