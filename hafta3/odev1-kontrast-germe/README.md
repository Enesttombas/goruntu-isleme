# Ödev 1: Min-max doğrusal ölçekleme

Düşük kontrastlı, tek kanallı görüntüyü aç; minimum ve maksimum değeri bul; parlaklığı
`y = (x - min) * 255 / (max - min)` ile [0, 255] aralığına doğrusal ölçekle.

| Sürüm | Çalıştır | Kullanılan |
|---|---|---|
| Hazır fonksiyonsuz | `cd hazir_fonksiyonsuz && python kontrast_germe.py` | formül elle, NumPy dizi aritmetiği |
| Hazır fonksiyonlu | `cd hazir_fonksiyonlu && python kontrast_germe.py` | `cv2.minMaxLoc`, `cv2.normalize(NORM_MINMAX)` |

| Girdi | Önce min/max | Sonra min/max |
|---|---|---|
| `dag.jpg` | 0 / 237 | 0 / 255 |
| `sis.jpg` | 92 / 249 | 0 / 255 |

Önce/sonra yan yana: `hazir_fonksiyonsuz/cikti/*_karsilastirma.png`.
