# Ödev 4.3: OpenCV Canny ve Tüm Yöntemlerin Karşılaştırılması

Bu ödevde Canny kenar bulma algoritması incelenmiş ve geliştirilen 9 filtreleme varyantının çalışma süreleri ile doğruluk metrikleri (Precision, Recall, F1) karşılaştırılmıştır.

## Canny Algoritması Aşamaları

1. **Gauss Filtresi:** Yüksek frekanslı gürültüyü giderme.
2. **Gradyan Hesaplama:** Sobel $G_x$ ve $G_y$ ile büyüklük ve yön açısı eldesi.
3. **Non-Maximum Suppression (NMS):** Gradyan yönünde yerel tepe noktası olmayan piksellerin elenmesi (1 piksel kalınlığa inceltme).
4. **Histerezis Eşikleme (Çift Eşik):**
   - Alt eşik: 50, Üst eşik: 100.
   - 100 üzerindeki pikseller kesin kenar, 50-100 arası pikseller kesin kenara bağlıysa kenar kabul edilir.

## Başarım ve Süre Karşılaştırması

- **Ölçüm Protokolü:** 512×512 görüntüler üzerinde tek iş parçacığı (`cv2.setNumThreads(1)`), 3 ısınma + 20 ölçümün medyanı alınmıştır.
- **Doğruluk Değerlendirmesi:** Yalnızca bilinen zemin gerçekliğe (ground truth) sahip sentetik görüntülerde yapılmıştır. Kamera fotoğrafı için doğruluk uydurulmamıştır.
- **Detaylı Tablo:** Tüm sayısal ölçümler `hafta4/cikti/olcumler.csv` dosyasında yer almaktadır.

## Çıktılar

- `kamera_canny_gauss.png`: Gauss ön filtrelemeli Canny kenar haritası.
- `kamera_canny_nogauss.png`: Gauss ön filtresiz Canny kenar haritası.
- `karsilastirma_tum_yontemler.png`: Tüm yöntemlerin yan yana karşılaştırma paneli.
