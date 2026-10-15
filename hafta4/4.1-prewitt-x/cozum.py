"""EBLG351 Hafta 4 - Ödev 4.1: Prewitt X Gradyan Operatörü.

Bu betik:
1. Girdi kamera görüntüsüne 5x5 Gauss yumuşatma (sigma=1.0) uygular.
2. Yalnızca [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]] Prewitt X çekirdeğini konvolüsyonla uygular.
   Gx, x yönündeki (yatay) yoğunluk değişimini ölçer; bu nedenle dikey kenarlara duyarlıdır.
3. İşaretli float32 türevi negatifleri kaybetmeden uint8 olarak görselleştirir.
4. Girdi, yumuşatılmış ve sonuç panosunu oluşturup kaydeder.
"""

from pathlib import Path
import sys
import cv2
import numpy as np

# Repo ve hafta4 dizinlerini içe aktarma yoluna ekle
dosya_yolu = Path(__file__).resolve()
hafta4_dizin = dosya_yolu.parent.parent
repo_kok = hafta4_dizin.parent
for d in [str(repo_kok), str(hafta4_dizin)]:
    if d not in sys.path:
        sys.path.insert(0, d)

from hafta4.kenarlar import gauss_yumusat, isaretli_goruntu_uint8, panel_olustur


def prewitt_x(gri: np.ndarray) -> np.ndarray:
    """Prewitt X çekirdeği ile dikey kenarları tespit eder (işaretli float32 döner)."""
    kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    girdi_f = gri.astype(np.float32, copy=False)
    return cv2.filter2D(girdi_f, cv2.CV_32F, kx, borderType=cv2.BORDER_REFLECT_101)


def calistir():
    """Ödev 4.1 Prewitt X adımlarını yürütür ve çıktıları kaydeder."""
    cikti_klasor = dosya_yolu.parent / "cikti"
    cikti_klasor.mkdir(parents=True, exist_ok=True)
    genel_cikti = hafta4_dizin / "cikti"
    genel_cikti.mkdir(parents=True, exist_ok=True)

    girdi_yol = hafta4_dizin / "girdi" / "kamera.png"
    if not girdi_yol.exists():
        raise FileNotFoundError(f"Girdi dosyası bulunamadı: {girdi_yol}")

    gri = cv2.imread(str(girdi_yol), cv2.IMREAD_GRAYSCALE)
    if gri is None:
        raise ValueError("kamera.png okunamadı!")

    # 1. 5x5 Gauss Yumuşatma (sigma=1.0)
    yumusak = gauss_yumusat(gri)

    # 2. Prewitt X Türevi (Gx - yatay türev, dikey kenarlara duyarlı)
    gx = prewitt_x(yumusak)
    gx_vis = isaretli_goruntu_uint8(gx)

    # 3. Görsel Panel: Girdi / Yumuşatılmış / Prewitt X Sonuç
    panel = panel_olustur(
        [gri, yumusak, gx_vis],
        ["1. Girdi (Orijinal Gri)", "2. 5x5 Gauss Yumusatma", "3. Prewitt X (Dikey Kenar Sonucu)"],
        (1, 3),
    )

    # Çıktıları kaydet
    cv2.imwrite(str(cikti_klasor / "kamera_gauss.png"), yumusak)
    cv2.imwrite(str(cikti_klasor / "kamera_prewitt_gx.png"), gx_vis)
    cv2.imwrite(str(cikti_klasor / "panel_prewitt_x.png"), panel)

    # Genel çıktı dizini için de güncelle
    cv2.imwrite(str(genel_cikti / "kamera_gauss.png"), yumusak)
    cv2.imwrite(str(genel_cikti / "kamera_prewitt_gx.png"), gx_vis)

    print("Ödev 4.1 (Prewitt X) tamamlandı.")


if __name__ == "__main__":
    calistir()
