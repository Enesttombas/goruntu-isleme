# Hafta 2

Girdi: `dama.jpg` (gri okunur); betiklerin varsayılan çıktısı `cikti/` klasörüdür.

1. **6-bit nicemleme (`nicemle_6bit.py`):** Alt iki bit atılır, nicemli görüntü ve ölçekli fark kaydedilir; gerçek çalışmada 256 → 64 seviye oldu, süre ölçülmedi.
   `python hafta2/nicemle_6bit.py` — isteğe bağlı: `[girdi] [cikti]`.
2. **Parça işlemleri (`parca_thread.py`):** Dört kadrana parlaklık, karartma, negatif ve gamma işlemi uygulanır.
   `python hafta2/parca_thread.py` — 20 tekrar ortalaması: 1 thread 0.007802 sn, 4 thread 0.005381 sn.
3. **Parça histogramı (`histogram_thread.py`):** Dört histogram thread ile hesaplanıp birleştirilir; tam görüntü histogramıyla eşitliği doğrulanır ve grafik çizilir; süre ölçülmedi.
   `python hafta2/histogram_thread.py` — isteğe bağlı: `[girdi] [cikti]`.
4. **Histogram süresi (`histogram_sure.py`):** Tek histogram ile dört parçalı histogramın sonuçları aynı çıkar.
   `python hafta2/histogram_sure.py` — 20 tekrar ortalaması: 1 thread 0.006001 sn, 4 thread 0.003788 sn.
5. **Parlaklık dönüşümü (`parlaklik_donusum.py`):** `0.75*x`, `x+20` ve `0.75*x+20` ayrı kaydedilir, etiketli karşılaştırma oluşturulur; süre ölçülmedi.
   `python hafta2/parlaklik_donusum.py` — isteğe bağlı: `[girdi] [cikti]`.
