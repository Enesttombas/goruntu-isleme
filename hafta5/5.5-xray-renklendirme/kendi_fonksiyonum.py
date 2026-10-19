"""Hafta 5 / 5.5: gri + renklendirme KENDİ fonksiyonlarımızla; orijinal örnekle sayısal karşılaştırma."""

import sys
from pathlib import Path

import cv2
import numpy as np

KLASOR = Path(__file__).resolve().parent
sys.path.insert(0, str(KLASOR.parent))
from xray_palet import palet_olustur  # noqa: E402

GIRDI = KLASOR.parent / "girdi" / "ThreatsRGB"
CIKTI = KLASOR / "cikti" / "kendi"
SINIFLAR = ("diger", "turuncu", "yesil", "mavi")


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
    maske = nesne & (cv2.cvtColor(orijinal, cv2.COLOR_BGR2HSV)[:, :, 1] > 60)  # orijinalde malzeme rengi olan pikseller
    s_o, s_b = renk_sinifi(orijinal)[maske], renk_sinifi(boyali)[maske]
    karisiklik = np.zeros((4, 4), dtype=np.int64)  # satır: orijinal sınıfı, sütun: bizim sınıfımız
    np.add.at(karisiklik, (s_o, s_b), 1)
    return {
        "mutlak": float(np.abs(orijinal.astype(np.int32) - boyali.astype(np.int32)).mean()),
        "delta_e": float(delta_e.mean()),
        "nesne": float(nesne.mean()),
        "karisiklik": karisiklik,
    }


def uyum_oranlari(karisiklik: np.ndarray) -> tuple[float, dict]:
    toplam = karisiklik.sum()
    genel = float(np.trace(karisiklik) / toplam) if toplam else float("nan")
    sinif = {
        SINIFLAR[k]: (float(karisiklik[k, k] / karisiklik[k].sum()) if karisiklik[k].sum() else float("nan"))
        for k in (1, 2, 3)
    }
    return genel, sinif


def yan_yana(orijinal, gri, boyali) -> np.ndarray:
    gri3 = cv2.cvtColor(gri, cv2.COLOR_GRAY2BGR)
    parcalar = []
    for resim, yazi in ((orijinal, "Orijinal"), (gri3, "Gri"), (boyali, "Bizim renklendirme")):
        kopya = resim.copy()
        cv2.rectangle(kopya, (0, 0), (kopya.shape[1], 30), (255, 255, 255), -1)
        cv2.putText(kopya, yazi, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        parcalar.append(kopya)
    return np.hstack(parcalar)


def ozet_satiri(ad: str, sonuclar: list) -> str:
    karisiklik = sum(s["karisiklik"] for s in sonuclar)
    genel, sinif = uyum_oranlari(karisiklik)
    return (
        f"{ad} ({len(sonuclar)} görüntü): ort.mutlak fark={np.mean([s['mutlak'] for s in sonuclar]):.1f}, "
        f"ort.deltaE={np.mean([s['delta_e'] for s in sonuclar]):.1f}, malzeme pikseli={int(karisiklik.sum())}, "
        f"sınıf uyumu={genel:.2f} (turuncu {sinif['turuncu']:.2f}, yeşil {sinif['yesil']:.2f}, mavi {sinif['mavi']:.2f})"
    )


def calistir() -> None:
    CIKTI.mkdir(parents=True, exist_ok=True)
    palet = palet_olustur()
    dosyalar = sorted(GIRDI.glob("*.jpg"))
    if not dosyalar:
        raise FileNotFoundError(f"Görüntü yok: {GIRDI}")
    satirlar, sonuclar = [], {}
    for dosya in dosyalar:
        orijinal = cv2.imread(str(dosya), cv2.IMREAD_COLOR)
        if orijinal is None:
            raise FileNotFoundError(f"Okunamadı: {dosya}")
        gri = kendi_gri(orijinal)
        boyali = kendi_renklendir(gri, palet)
        cv2.imwrite(str(CIKTI / f"{dosya.stem}_gri.png"), gri)
        cv2.imwrite(str(CIKTI / f"{dosya.stem}_renkli.png"), boyali)
        cv2.imwrite(str(CIKTI / f"{dosya.stem}_karsilastirma.png"), yan_yana(orijinal, gri, boyali))
        o = olc(orijinal, boyali, gri)
        sonuclar[dosya.stem] = o
        genel, sinif = uyum_oranlari(o["karisiklik"])
        satirlar.append(
            f"{dosya.name} {orijinal.shape[1]}x{orijinal.shape[0]}: mutlak={o['mutlak']:.1f}, deltaE={o['delta_e']:.1f}, "
            f"nesne={o['nesne']:.2f}, sinif uyumu={genel:.2f} (turuncu {sinif['turuncu']:.2f}, yesil {sinif['yesil']:.2f}, mavi {sinif['mavi']:.2f})"
        )
    ozet = [ozet_satiri("TUMU", list(sonuclar.values()))]
    (CIKTI.parent / "olcumler.txt").write_text("\n".join(ozet + [""] + satirlar) + "\n", encoding="utf-8")
    print("\n".join(ozet))


if __name__ == "__main__":
    calistir()
