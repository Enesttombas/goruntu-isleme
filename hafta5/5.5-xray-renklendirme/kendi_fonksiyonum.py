"""Hafta 5 / 5.5: gri + renklendirme KENDİ fonksiyonlarımızla; orijinal örnekle sayısal karşılaştırma."""

import sys
from pathlib import Path

import cv2
import numpy as np

KLASOR = Path(__file__).resolve().parent
sys.path.insert(0, str(KLASOR.parent))
from xray_palet import palet_olustur  # noqa: E402

GIRDI = KLASOR.parent / "girdi" / "ThreatsRGB"
CIKTI = KLASOR / "cikti"


def kendi_gri(bgr: np.ndarray) -> np.ndarray:
    """Y = 0,299 R + 0,587 G + 0,114 B, 15 bit tamsayı ağırlıklarla (5.3 ile aynı)."""
    mavi, yesil, kirmizi = (bgr[:, :, i].astype(np.int32) for i in range(3))
    return ((9798 * kirmizi + 19235 * yesil + 3735 * mavi + (1 << 14)) >> 15).astype(np.uint8)


def kendi_renklendir(gri: np.ndarray, palet: np.ndarray) -> np.ndarray:
    """Her gri değeri palet tablosundan karşılık gelen BGR renge çevirir."""
    return palet[gri]


def renk_sinifi(bgr: np.ndarray) -> np.ndarray:
    """Renk tonuna göre sınıf: 1 turuncu/sarı, 2 yeşil, 3 mavi, 0 diğer."""
    ton = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 0].astype(np.int32) * 2  # derece 0-358
    sinif = np.zeros(ton.shape, dtype=np.uint8)
    sinif[(ton >= 10) & (ton < 55)] = 1
    sinif[(ton >= 55) & (ton < 170)] = 2
    sinif[(ton >= 170) & (ton < 270)] = 3
    return sinif


def olc(orijinal: np.ndarray, boyali: np.ndarray, gri: np.ndarray) -> dict:
    """Orijinal sahte renkli görüntü ile bizim renklendirmemizi karşılaştırır."""
    lab_o = cv2.cvtColor(orijinal.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)
    lab_b = cv2.cvtColor(boyali.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)
    delta_e = np.sqrt(((lab_o - lab_b) ** 2).sum(axis=2))

    nesne = gri < 230  # boşluğu (parlak zemin) dışarıda bırak
    doygun = cv2.cvtColor(orijinal, cv2.COLOR_BGR2HSV)[:, :, 1] > 60
    maske = nesne & doygun  # orijinalde malzeme rengi taşıyan pikseller
    s_o, s_b = renk_sinifi(orijinal)[maske], renk_sinifi(boyali)[maske]
    uyum = float((s_o == s_b).mean()) if maske.any() else float("nan")
    # Sınıf bazında: orijinalde X olan piksellerin ne kadarında biz de X dedik.
    sinif_uyum = {
        ad: (float((s_b[s_o == k] == k).mean()) if (s_o == k).any() else float("nan"))
        for k, ad in ((1, "turuncu"), (2, "yesil"), (3, "mavi"))
    }
    return {
        "ortalama_mutlak_fark": float(np.abs(orijinal.astype(np.int32) - boyali.astype(np.int32)).mean()),
        "ortalama_delta_e": float(delta_e.mean()),
        "nesne_orani": float(nesne.mean()),
        "malzeme_pikseli": int(maske.sum()),
        "genel_sinif_uyumu": uyum,
        "sinif_uyumu": sinif_uyum,
    }


def yan_yana(orijinal, gri, boyali) -> np.ndarray:
    gri3 = cv2.cvtColor(gri, cv2.COLOR_GRAY2BGR)
    parcalar = []
    for resim, yazi in ((orijinal, "Orijinal"), (gri3, "Gri"), (boyali, "Bizim renklendirme")):
        kopya = resim.copy()
        cv2.rectangle(kopya, (0, 0), (kopya.shape[1], 30), (255, 255, 255), -1)
        cv2.putText(kopya, yazi, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        parcalar.append(kopya)
    return np.hstack(parcalar)


def calistir() -> None:
    CIKTI.mkdir(exist_ok=True)
    palet = palet_olustur()
    dosyalar = sorted(GIRDI.glob("*.jpg"))
    if not dosyalar:
        raise FileNotFoundError(f"Görüntü yok: {GIRDI}")
    satirlar = []
    for dosya in dosyalar:
        orijinal = cv2.imread(str(dosya), cv2.IMREAD_COLOR)
        if orijinal is None:
            raise FileNotFoundError(f"Okunamadı: {dosya}")
        gri = kendi_gri(orijinal)
        boyali = kendi_renklendir(gri, palet)
        cv2.imwrite(str(CIKTI / f"{dosya.stem}_gri_kendi.png"), gri)
        cv2.imwrite(str(CIKTI / f"{dosya.stem}_renkli_kendi.png"), boyali)
        cv2.imwrite(str(CIKTI / f"{dosya.stem}_karsilastirma.png"), yan_yana(orijinal, gri, boyali))
        o = olc(orijinal, boyali, gri)
        su = o["sinif_uyumu"]
        satirlar.append(
            f"{dosya.name}: ort.mutlak fark={o['ortalama_mutlak_fark']:.1f}, ort.deltaE={o['ortalama_delta_e']:.1f}, "
            f"nesne orani={o['nesne_orani']:.2f}, malzeme pikseli={o['malzeme_pikseli']}, "
            f"sinif uyumu={o['genel_sinif_uyumu']:.2f} (turuncu {su['turuncu']:.2f}, yesil {su['yesil']:.2f}, mavi {su['mavi']:.2f})"
        )
    (CIKTI / "olcumler.txt").write_text("\n".join(satirlar) + "\n", encoding="utf-8")
    print("\n".join(satirlar))


if __name__ == "__main__":
    calistir()
