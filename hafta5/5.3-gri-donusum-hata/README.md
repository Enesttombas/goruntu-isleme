# 5.3 — RGB'den griye: OpenCV ve kendi fonksiyonum

Gri değer göz duyarlılığına göre ağırlıklandırılır: `Y = 0,299·R + 0,587·G + 0,114·B`. Düz ortalama (R+G+B)/3 yeşili az, maviyi çok sayar. `cv2.cvtColor(..., COLOR_BGR2GRAY)` ile `kendi_gri` aynı ağırlıkları kullanır; hata, eşleşen piksellerin mutlak farklarının toplamıdır.

```powershell
python hafta5/5.3-gri-donusum-hata/cozum.py
```

Çıktı: [OpenCV gri](cikti/gri_hazir_fonksiyon.png), [kendi gri](cikti/gri_kendi_fonksiyon.png), [hata](cikti/hata.txt).

Sonuç: 512 × 512 piksel için hata toplamı **111** (piksel başına ≈ 0,0004). Fark, yalnızca yuvarlama ayrıntısından gelir.
