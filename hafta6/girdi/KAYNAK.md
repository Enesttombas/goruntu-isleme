# Girdi görüntüleri

| Dosya | Kullanıldığı ödev | Kaynak |
|---|---|---|
| `sudoku.png` | 6.1, 6.2, 6.6 (doğru) | OpenCV örnek verisi, [opencv/samples/data/sudoku.png](https://github.com/opencv/opencv/blob/4.x/samples/data/sudoku.png) (Apache 2.0), 558×563 |
| `smarties.png` | 6.3, 6.4 (çember) | OpenCV örnek verisi, [opencv/samples/data/smarties.png](https://github.com/opencv/opencv/blob/4.x/samples/data/smarties.png) (Apache 2.0), 413×356 |
| `es_merkezli.png` | 6.5 (eş merkezli çember) | Kendi çizdiğimiz sentetik test görüntüsü, 520×400 |

`es_merkezli.png` içeriği (doğru cevabı bildiğimiz için seçildi):

- Eş merkezli küme 1: merkez (150, 190), yarıçap 35, 65, 95
- Eş merkezli küme 2: merkez (390, 150), yarıçap 20, 45, 70
- Tek çember (eş merkezli değil): merkez (380, 315) r=50 ve (270, 330) r=30
- Bir doğru parçası ve σ=8 Gauss gürültüsü

Halkalar dolu daire olarak büyükten küçüğe `cv2.circle(..., -1)` ile çizildi; her halka sınırı tek kenar verir.
