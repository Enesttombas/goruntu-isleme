# 7.2 — Vektöre gürültü ekleme

**Ödev:** Girdi olarak vektör ve [0, 1] aralığında gürültü oranı alan fonksiyon. Vektördeki genliğe ve orana göre her elemana rastgele gürültü eklenir, sonuç grafikle gösterilir.

- `genlik = max|x|`
- Her elemana `[-oran·genlik, +oran·genlik]` aralığından düzgün dağılımlı rastgele sayı eklenir.
- Girdi vektörü değişmez, gürültülü yeni vektör döner. Oran aralık dışındaysa `ValueError`.
- Tohum sabit (`TOHUM = 7`), aynı gürültü 7.3 ve 7.4'te de kullanılıyor.

| Dosya | Çıktı |
|---|---|
| [gurultu.py](gurultu.py) | `cikti/gurultulu_sinyal.png` |

```powershell
python hafta7/7.2-gurultu-ekleme/gurultu.py
```

Sonuç (sinyal N=512, A=1, f=5 Hz):

| oran | en büyük \|gürültü\| | gürültü RMS | beklenen RMS (oran/√3) |
|---|---|---|---|
| 0.1 | 0.0998 | 0.0567 | 0.0577 |
| 0.3 | 0.2994 | 0.1702 | 0.1732 |
| 0.6 | 0.5989 | 0.3405 | 0.3464 |

Gürültü her oranda sınırın içinde kalıyor ve RMS değeri düzgün dağılımın teorik değerine yakın. Oran 0.6'da sinüs hâlâ seçiliyor ama tepeler artık tek bir değer gibi görünmüyor.
