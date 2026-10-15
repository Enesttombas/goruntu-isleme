"""EBLG351 Hafta 4 - Ödev 4.2: Prewitt Y Gradyan Operatörü.

Bu betik:
1. Girdi kamera görüntüsüne 5x5 Gauss yumuşatma (sigma=1.0) uygular.
2. Yalnızca [[-1, -1, -1], [0, 0, 0], [1, 1, 1]] Prewitt Y çekirdeğini uygular.
   Gy, y yönündeki (dikey) yoğunluk değişimini ölçer; bu nedenle yatay kenarlara duyarlıdır.
3. İşaretli float32 türevi negatifleri kaybetmeden uint8 olarak görselleştirir.
4. Girdi, yumuşatılmış ve sonuç panosunu oluşturup kaydeder.
"""

from pathlib import Path
import sys
import cv2
import numpy as np

dosya_yolu = Path(__file__).resolve()
hafta4_dizin = dosya_yolu.parent.parent
repo_kok = hafta4_dizin.parent
for d in [str(repo_kok), str(hafta4_dizin)]:
    if d not in sys.path:
        sys.path.insert(0, d)

from hafta4.kenarlar import gauss_yumusat, isaretli_goruntu_uint8, panel_olustur


def prewitt_y(gri: np.ndarray) -> np.ndarray:
    """Prewitt Y çekirdeği ile yatay kenarları tespit eder (işaretli float32 döner)."""
    ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    girdi_f = gri.astype(np.float32, copy=False)
    return cv2.filter2D(girdi_f, cv2.CV_32F, ky, borderType=cv2.BORDER_REFLECT_101)


def calistir():
    """Ödev 4.2 Prewitt Y adımlarını yürütür ve çıktıları kaydeder."""
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

    # 2. Prewitt Y Türevi (Gy - dikey türev, yatay kenarlara duyarlı)
    gy = prewitt_y(yumusak)
    gy_vis = isaretli_goruntu_uint8(gy)

    # 3. Görsel Panel: Girdi / Yumuşatılmış / Prewitt Y Sonuç
    panel = panel_olustur(
        [gri, yumusak, gy_vis],
        ["1. Girdi (Orijinal Gri)", "2. 5x5 Gauss Yumusatma", "3. Prewitt Y (Yatay Kenar Sonucu)"],
        (1, 3),
    )

    # Çıktıları kaydet
    cv2.imwrite(str(cikti_klasor / "kamera_gauss.png"), yumusak)
    cv2.imwrite(str(cikti_klasor / "kamera_prewitt_gy.png"), gy_vis)
    cv2.imwrite(str(cikti_klasor / "panel_prewitt_y.png"), panel)

    # Genel çıktı dizini için de güncelle
    cv2.imwrite(str(genel_cikti / "kamera_prewitt_gy.png"), gy_vis)

    print("Ödev 4.2 (Prewitt Y) tamamlandı.")


if __name__ == "__main__":
    calistir()
