# Hafta 4 — Kenar Algoritmaları (EBLG351)

Python + OpenCV + NumPy ile 1. ve 2. mertebeden kenar tespit algoritmaları, analitik filtre çekirdekleri, sıfır geçişleri, Canny kenar tespiti ve başarım karşılaştırmaları.

---

## 1. Yedi Madde ve Ödev Eşlemesi

Hafta 4 kapsamında istenen 7 kuramsal ve uygulamalı maddenin ödev klasörlerine dağılımı:

| No | Konu / Madde | İlgili Ödev | Açıklama |
|---|---|---|---|
| 1 | **Hafif Gauss Yumuşatma** (5×5, $\sigma=1.0$) | Ortak / [4.1](4.1-prewitt-sobel/), [4.2](4.2-laplacian-log/), [4.3](4.3-canny/) | Yüksek frekanslı gürültüyü bastıran temel ön filtreleme |
| 2 | **Prewitt X / Y Gradyanları ve Büyüklük** | [4.1 Prewitt-Sobel](4.1-prewitt-sobel/) | 1. mertebeden yönlü türevler ve $M=\sqrt{G_x^2+G_y^2}$ büyüklüğü |
| 3 | **Sobel X / Y Gradyanları ve Büyüklük** | [4.1 Prewitt-Sobel](4.1-prewitt-sobel/) | Yerel Gauss ağırlıklı gradyan filtresi ve Prewitt ile 1 sayfalık kıyas |
| 4 | **3×3 Laplacian 2. Türev ve Sıfır Geçişleri** | [4.2 Laplacian-LoG](4.2-laplacian-log/) | 4-komşuluk izotropik 2. türev; Gauss'lu ve Gauss'suz gürültü farkı |
| 5 | **7×7 Analitik LoG Çekirdeği** | [4.2 Laplacian-LoG](4.2-laplacian-log/) | $\sigma=1.0$ analitik Marr-Hildreth çekirdeği ve sıfır geçiş tespiti |
| 6 | **OpenCV Canny Kenar Algoritması** | [4.3 Canny](4.3-canny/) | Gradyan, Non-Maximum Suppression (1 px) ve 50/100 histerezis eşikleme |
| 7 | **Süre ve Doğruluk (F1) Karşılaştırması** | [4.3 Canny](4.3-canny/) | 9 varyantın kamera, sentetik ve gürültülü verideki süre ve F1 skorları |

---

## 2. Ortak Fonksiyonlar Sözleşmesi (`kenarlar.py`)

| Fonksiyon | Girdi | Çıktı | Açıklama ve Neden Yapıldığı |
|---|---|---|---|
| `gauss_yumusat(gri)` | `gri` (uint8/float32) | `np.ndarray` | 5×5 Gauss ($\sigma=1.0$, `BORDER_REFLECT_101`). Türev öncesi gürültüyü bastırır. |
| `prewitt(gri, yumusat=True)` | `gri`, `yumusat` | `(gx, gy, hypot)` | Prewitt $X/Y$ gradyanları ve büyüklüğü (işaretli float32). |
| `sobel(gri, yumusat=True)` | `gri`, `yumusat` | `(gx, gy, hypot)` | Sobel $X/Y$ gradyanları ve büyüklüğü (`CV_32F`, `scale=1`). |
| `laplacian(gri, yumusat=True)` | `gri`, `yumusat` | `lap` (float32) | 4-komşuluk `[[0,1,0],[1,-4,1],[0,1,0]]` 2. türev filtresi. |
| `log_cekirdegi()` | — | `np.ndarray` (7×7) | $\sigma=1.0$ analitik LoG çekirdeği. Ortalaması 0, mutlak toplamı 1, merkezi negatif. |
| `log_filtresi(gri)` | `gri` | `log_res` (float32) | 7×7 LoG yanıtı. Dahili Gauss taşıdığı için harici 2. Gauss uygulanmaz. |
| `canny(gri, yumusat=True)` | `gri`, `yumusat` | `np.ndarray` (uint8) | OpenCV `Canny` (eşikler 50, 100, `apertureSize=3`, `L2gradient=True`). |
| `gradyan_ikili(hypot, esik_orani=0.2)` | `hypot`, `esik_orani` | `np.ndarray` (uint8) | Tepe değerin %20'si ile ikili kenar haritası. Sabit resimde 0 kenar üretir. |
| `sifir_gecis(turev2, esik_orani=0.05)` | `turev2`, `esik_orani` | `np.ndarray` (uint8) | 2. türevde 8-komşuluklu işaret değişimi takibiyle kenar çıkarır. |
| `isaretli_goruntu_uint8(dizi)` | `dizi` (float32) | `np.ndarray` (uint8) | Negatif türevleri kırpmadan 0-255 aralığına gererek görselleştirir. |
| `mutlak_uint8(dizi)` | `dizi` (float32) | `np.ndarray` (uint8) | Mutlak değeri normalize ederek uint8 yapar. |
| `kenar_skoru(tahmin, referans, tolerans=1)` | `tahmin`, `referans` | `(precision, recall, f1)` | 1 piksel toleranslı (dilate) doğruluk metrikleri. Sentetik referans için ölçülür. |
| `sentetik_veri_uret()` | — | `(temiz, referans, gurultulu)` | Daire, dikdörtgen ve çizgili 512×512 sentetik test verisi (RNG=42, $\sigma=10$). |

---

## 3. Çalıştırma Talimatları

### Tüm Çalışmayı Yeniden Üretme
Repo kök dizininden:
```bash
python hafta4/calistir.py
```
Bu komut 4.1, 4.2 ve 4.3 ödevlerini sırasıyla çalıştırır, tüm filtre çıktılarını diske yazar, `hafta4/cikti/olcumler.csv` dosyasını oluşturur ve `hafta4/raporlar/KARSILASTIRMA.pdf` (tam 3 sayfa) ile `KARSILASTIRMA.md` raporlarını derler.

### Birim Testleri Çalıştırma
```bash
pytest hafta4/test_kenarlar.py -v
```

### Alt Ödevleri Ayrı Ayrı Çalıştırma
Her ödev doğrudan kendi dizininden veya repo kökünden çalıştırılabilir:
```bash
# Repo kökünden:
python hafta4/4.1-prewitt-sobel/cozum.py
python hafta4/4.2-laplacian-log/cozum.py
python hafta4/4.3-canny/cozum.py

# Ya da ilgili klasörün içinden:
cd hafta4/4.1-prewitt-sobel && python cozum.py
```

---

## 4. Rapor ve Çıktı Dosyaları

- `hafta4/cikti/olcumler.csv`: 9 varyantın 3 test görüntüsündeki süre ve başarım metrikleri (27 satır).
- `hafta4/cikti/calisma_raporu.txt`: Çalıştırma özeti ve doğrulama kayıtları.
- `hafta4/cikti/*.png`: Tekil filtre çıktıları (Prewitt, Sobel, Laplacian, LoG, Canny) ve karşılaştırma panoları (toplam 15+ görsel).
- `hafta4/raporlar/KARSILASTIRMA.md`: Detaylı kuramsal açıklama, matematiksel formüller, görsel bağlantıları ve sonuç değerlendirmesi.
- `hafta4/raporlar/KARSILASTIRMA.pdf`: Tam 3 sayfa olarak derlenmiş resmi rapor:
  - **Sayfa 1:** Prewitt ve Sobel Karşılaştırması (1 sayfalık kuramsal ve deneysel kıyas)
  - **Sayfa 2:** Laplacian, 7×7 LoG Çekirdeği ve Gauss Ön Filtreleme Etkisi
  - **Sayfa 3:** Canny Algoritması, 9 Varyantlık Sayısal Ölçüm Tablosu, Sınırlılıklar ve Kaynaklar
