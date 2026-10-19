# 5.1 — Renk kanallarının histogramı ve siyah beyaz görüntü

Tek bir renkli fotoğrafın kırmızı, yeşil ve mavi kanalları için ayrı 256 seviyeli histogram oluşturulur. Aynı fotoğraf OpenCV'nin `BGR2GRAY` dönüşümüyle tek kanallı gri tonlamaya çevrilir. Buradaki “siyah beyaz”, yalnızca 0 ve 255 değerlerinden oluşan ikili görüntü değil, 0–255 gri tonlamadır.

## Çalıştırma

Depo kökünden:

```powershell
python hafta5/5.1-renk-kanallari-histogram/cozum.py
```

Girdi: [astronot.png](../girdi/astronot.png). Çıktılar: [kırmızı](cikti/histogram_kirmizi.png), [yeşil](cikti/histogram_yesil.png), [mavi](cikti/histogram_mavi.png) histogramları ve [gri fotoğraf](cikti/astronot_gri.png).

OpenCV görüntüyü BGR sırasıyla okur; kırmızı için kanal 2, yeşil için kanal 1, mavi için kanal 0 kullanılır. Histogramın yatay ekseni 0–255 yoğunluk, dikey ekseni o yoğunluktaki piksel sayısıdır. Üç kanalın her birinin toplamı fotoğrafın 512 × 512 = 262.144 pikseline eşittir.

## Fotoğraf kaynağı

NASA'nın astronot Eileen Collins fotoğrafı, [scikit-image `data.astronaut` örnek görüntüsü](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut) olarak yayımlanmıştır. Dokümantasyon görüntüyü kamu malı ve bilinen telif kısıtlaması bulunmayan bir fotoğraf olarak tanımlar. Dosya [scikit-image v0.19.3 deposundan](https://github.com/scikit-image/scikit-image/blob/v0.19.3/skimage/data/astronaut.png) 19 Ekim 2026'da alınmıştır.
