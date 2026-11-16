# 9.2 — Renkli BMP ve gri BMP boyutları

Ortak `../girdi/renkli.bmp` OpenCV ile okunur, 1000×1000 boyutuna getirilir ve
`girdi/renkli_1000.bmp` olarak kaydedilir. Bu BMP diskten tekrar okunup griye
çevrilir. Normal çalıştırma iki pencere açar; bir tuşa basılınca pencereler kapanır.

Çalışma klasöründen:

```powershell
python hafta9/9.2-bmp-gri/bmp_gri.py
python hafta9/9.2-bmp-gri/bmp_gri.py --gosterme
```

`--gosterme` pencere açmadan aynı çıktıları üretir. Yollar betiğin konumuna göre
hesaplanır; komutun çalıştırıldığı klasöre bağlı değildir.

| Dosya | İçerik |
|---|---|
| `girdi/renkli_1000.bmp` | Diskten tekrar okunan 1000×1000, üç kanallı BGR girdi |
| `cikti/gri.bmp` | 1000×1000, tek kanallı gri görüntü |
| `cikti/yan_yana.png` | Solda renkli, sağda gri |
| `cikti/boyutlar.txt` | Gerçek disk boyutları ve BMP hesabı |

Bir milyon piksel için uint8 renkli veri 3.000.000, gri veri 1.000.000 bayttır.
BMP'de 14 bayt dosya başlığı ve 40 bayt görüntü başlığı bulunur. Gri BMP ayrıca
256 girişli, giriş başına 4 baytlık palet taşır: 1024 bayt. Tek kanallı gri veri
bu paletteki gri tonlarının indekslerini tutar.

Satır uzunluğu `((genişlik * kanal + 3) // 4) * 4` bayttır. 1000 genişlikte
3000 ve 1000 zaten 4'ün katı olduğundan satır dolgusu gerekmez. Gerçek dosyalar
renkli için `54 + 3000 * 1000 = 3.000.054`, gri için
`54 + 1024 + 1000 * 1000 = 1.001.078` bayttır. Disk oranı yaklaşık 2,996823424;
başlık ve palet nedeniyle tam 3 değildir. Başka genişliklerde satır dolgusu da
hesaba katılır. Betik bu hesabı gerçek BMP başlıkları ve disk boyutlarıyla denetler.
