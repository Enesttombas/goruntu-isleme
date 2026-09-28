# Hafta 2

Girdi: `dama.jpg` (gri tonlamada okunur). Her betik çıktısını `cikti/` klasörüne yazar.

| Ödev | Çalıştır | Sonuç |
|---|---|---|
| 1. 8-bit → 6-bit nicemleme | `python nicemle_6bit.py` | 256 → 64 seviye |
| 2. 4 parça × 4 thread parlaklık | `python parca_thread.py` | 1 thread 0.0078 sn, 4 thread 0.0054 sn (20 tekrar ort.) |
| 3. Parça histogramları, ana histogram | `python histogram_thread.py` | Parça toplamı = tüm görüntü histogramı |
| 4. Histogram süresi | `python histogram_sure.py` | 1 thread 0.0060 sn, 4 thread 0.0038 sn (20 tekrar ort.) |
| 5. Parlaklık dönüşümü `0.75*x + 20` | `python parlaklik_donusum.py` | `0.75*x`, `x+20` ve ikisi ayrı; yan yana `karsilastirma.png` |
