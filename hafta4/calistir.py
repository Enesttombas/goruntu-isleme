"""EBLG351 Hafta 4 - Ana Çalıştırıcı Betik (calistir.py).

Bu betik:
1. Hafta 4'ün 7 ödevini sırasıyla çalıştırır (4.1 - 4.7).
2. Tüm filtre görsel çıktılarını (>= 12 PNG) ve panoları üretir.
3. Çalışma süresi ve doğruluk ölçümlerini hafta4/cikti/olcumler.csv dosyasına kaydeder.
4. En son ölçüm verisiyle tum_raporlari_guncelle() fonksiyonunu çağırır.
5. Bir özet çalıştırma raporunu hafta4/cikti/calisma_raporu.txt olarak kaydeder.
"""

from datetime import datetime
from pathlib import Path
import runpy
import sys
import time
import pymupdf

dosya_yolu = Path(__file__).resolve()
hafta4_dizin = dosya_yolu.parent
repo_kok = hafta4_dizin.parent
for d in [str(repo_kok), str(hafta4_dizin)]:
    if d not in sys.path:
        sys.path.insert(0, d)

from hafta4.rapor_uret import tum_raporlari_guncelle


def calistir():
    t_basla = time.time()
    log_satirlari = []

    def log(msg):
        print(msg)
        log_satirlari.append(msg)

    log("=" * 65)
    log(f"EBLG351 Hafta 4 Kenar Algoritmaları — Başlatıldı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 65)

    gorevler = [
        ("4.1-prewitt-x", "Ödev 4.1: Prewitt X Gradyan"),
        ("4.2-prewitt-y", "Ödev 4.2: Prewitt Y Gradyan"),
        ("4.3-sobel", "Ödev 4.3: 3x3 Sobel Gradyan"),
        ("4.4-prewitt-sobel-karsilastirma", "Ödev 4.4: Prewitt-Sobel Karşılaştırması"),
        ("4.5-laplacian", "Ödev 4.5: 3x3 Laplacian"),
        ("4.6-log", "Ödev 4.6: 7x7 Analitik LoG"),
        ("4.7-canny", "Ödev 4.7: Canny ve Ölçümler"),
    ]

    for idx, (klasor, aciklama) in enumerate(gorevler, 1):
        log(f"\n[{idx}/7] {aciklama} çalıştırılıyor...")
        t0 = time.time()
        p = hafta4_dizin / klasor / "cozum.py"
        runpy.run_path(str(p), run_name="__main__")
        log(f"   -> {klasor} tamamlandı ({time.time() - t0:.2f} sn)")

    log("\n[Raporlar] Tüm raporlar en son CSV ölçümleriyle güncelleniyor...")
    t0 = time.time()
    tum_raporlari_guncelle()
    log(f"   -> Raporlar güncellendi ({time.time() - t0:.2f} sn)")

    # Doğrulama kontrolleri
    genel_cikti = hafta4_dizin / "cikti"
    png_dosyalari = list(genel_cikti.glob("*.png"))
    csv_dosyasi = genel_cikti / "olcumler.csv"
    pdf_dosyasi = hafta4_dizin / "raporlar" / "KARSILASTIRMA.pdf"
    md_dosyasi = hafta4_dizin / "raporlar" / "KARSILASTIRMA.md"

    doc_pdf = pymupdf.open(str(pdf_dosyasi))
    pdf_sayfa = len(doc_pdf)

    log("\n" + "=" * 65)
    log("KONTROL VE DOĞRULAMA:")
    log(f"• Üretilen PNG görsel sayısı (hafta4/cikti/): {len(png_dosyalari)} (Hedef: >= 12)")
    log(f"• Ölçüm CSV dosyası mevcut mu: {csv_dosyasi.exists()} ({csv_dosyasi.stat().st_size} bayt)")
    log(f"• Markdown raporu mevcut mu: {md_dosyasi.exists()}")
    log(f"• PDF raporu mevcut mu: {pdf_dosyasi.exists()} | Sayfa sayısı: {pdf_sayfa} (Hedef: 3)")
    log(f"• Toplam çalışma süresi: {time.time() - t_basla:.2f} sn")
    log("=" * 65)

    if len(png_dosyalari) < 12:
        raise ValueError(f"Hata: En az 12 PNG görsel üretilmelidir, üretilen: {len(png_dosyalari)}")
    if not csv_dosyasi.exists():
        raise FileNotFoundError("Hata: olcumler.csv bulunamadı!")
    if pdf_sayfa != 3:
        raise ValueError(f"Hata: PDF tam 3 sayfa olmalıdır, mevcut: {pdf_sayfa}")

    # Log dosyasını kaydet
    log_yol = genel_cikti / "calisma_raporu.txt"
    log_yol.write_text("\n".join(log_satirlari), encoding="utf-8")
    print(f"\nÇalışma özeti kaydedildi: {log_yol}")


if __name__ == "__main__":
    calistir()
