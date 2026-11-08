# 7.4 — DFT ile harmonik vektörü

**Ödev:** 2'nin kuvveti uzunluktaki vektörü DFT ile frekans uzayına çeviren ve harmonik vektörünü döndüren fonksiyon. Gürültüsüz ve gürültülü vektörler ayrı ayrı gönderilip harmonik vektörü yorumlanır.

`X[k] = Σ x[n]·e^(-j2πkn/N)`

- `dft(x)`: tanımın birebir hali, N×N matrisle çarpım, O(N²).
- `fft(x)`: aynı sonucu veren Cooley-Tukey radix-2. Vektör çift ve tek indisli elemanlara bölünür, iki yarım DFT `X[k] = Ç[k] ± W^k·T[k]` ile birleştirilir, O(N log N). Uzunluğun 2'nin kuvveti olması bu yüzden gerekiyor.
- `ifft(X)`: ters dönüşüm, `conj(fft(conj(X))) / N`.
- `harmonik_vektoru(x)`: ödevin istediği fonksiyon, `fft` sonucunu döndürür.
- `genlik_spektrumu`: `|X[k]|·2/N`; f Hz'teki A genlikli sinüs, f'de A yüksekliğinde tepe verir.

| Dosya | Çıktı |
|---|---|
| [dft.py](dft.py) | `cikti/harmonik_vektor.png` |

```powershell
python hafta7/7.4-dft-harmonik/dft.py
```

Doğrulama: `max|dft - fft| = 1.8e-11`, `max|fft - np.fft.fft| = 1.0e-13`, `max|ifft(fft(x)) - x| = 7.1e-16`. numpy yalnız kontrol için kullanıldı.

Sonuç (N=512, fs=512 Hz, A=1, f=5 Hz, gürültü oranı 0.3):

| | tepe | tepe genliği | diğer harmonikler (ortalama / en büyük) |
|---|---|---|---|
| gürültüsüz | 5 Hz | 1.0000 | 3.4e-17 / ~0 |
| gürültülü | 5 Hz | 0.9943 | 0.0131 / 0.0410 |

Yorum:

- Gürültüsüz sinüs tek bir frekanstır: harmonik vektöründe yalnız 5 Hz'te 1.0 yüksekliğinde tepe var, geri kalanı sayısal olarak sıfır. f tam sayı ve fs = N olduğu için tepe tam bir harmoniğe düşüyor, sızıntı (leakage) yok.
- Gürültü belli bir frekansa ait değil, bütün harmoniklere yayılıp düz bir taban oluşturuyor. Sinyalin harmoniği neredeyse aynı kalıyor (0.994).
- Sinyal tepesi tabanın en büyük değerinden 24 kat yüksek. Zamanda iç içe olan sinyal ve gürültü frekans uzayında kolayca ayrılıyor; 7.6'daki filtre bu fikre dayanıyor.
