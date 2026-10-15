"""EBLG351 Hafta 4 - Ana Çalıştırıcı Betik (calistir.py).

Bu betik:
1. Hafta 4'ün tüm alt ödevlerini (4.1, 4.2, 4.3) sırasıyla çalıştırır.
2. Tüm filtre görsel çıktılarını (>= 12 PNG) ve karşılaştırma panolarını üretir.
3. Çalışma süresi ve doğruluk ölçümlerini hafta4/cikti/olcumler.csv dosyasına kaydeder.
4. hafta4/rapor_uret.py modülünü çağırarak KARSILASTIRMA.md ve tam 3 sayfalık KARSILASTIRMA.pdf belgelerini üretir.
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


def calistir():
    t_basla = time.time()
    log_satirlari = []

    def log(msg):
        print(msg)
        log_satirlari.append(msg)

    log("=" * 65)
    log(f"EBLG351 Hafta 4 Kenar Algoritmaları — Başlatıldı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 65)

    # 1. Ödev 4.1: Prewitt ve Sobel
    log("\n[1/4] Ödev 4.1 (Prewitt & Sobel) çalıştırılıyor...")
    t0 = time.time()
    p41 = hafta4_dizin / "4.1-prewitt-sobel" / "cozum.py"
    runpy.run_path(str(p41), run_name="__main__")
    log(f"   -> 4.1 tamamlandı ({time.time() - t0:.2f} sn)")

    # 2. Ödev 4.2: Laplacian ve LoG
    log("\n[2/4] Ödev 4.2 (Laplacian & LoG) çalıştırılıyor...")
    t0 = time.time()
    p42 = hafta4_dizin / "4.2-laplacian-log" / "cozum.py"
    runpy.run_path(str(p42), run_name="__main__")
    log(f"   -> 4.2 tamamlandı ({time.time() - t0:.2f} sn)")

    # 3. Ödev 4.3: Canny ve Tüm Ölçümler
    log("\n[3/4] Ödev 4.3 (Canny & Ölçümler) çalıştırılıyor...")
    t0 = time.time()
    p43 = hafta4_dizin / "4.3-canny" / "cozum.py"
    runpy.run_path(str(p43), run_name="__main__")
    log(f"   -> 4.3 tamamlandı ({time.time() - t0:.2f} sn)")

    # 4. Rapor Üretimi (Markdown ve 3 Sayfa PDF)
    log("\n[4/4] Raporlar üretiliyor (KARSILASTIRMA.md ve KARSILASTIRMA.pdf)...")
    t0 = time.time()
    rapor_script = hafta4_dizin / "rapor_uret.py"
    runpy.run_path(str(rapor_script), run_name="__main__")
    log(f"   -> Raporlar üretildi ({time.time() - t0:.2f} sn)")

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
