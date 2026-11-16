"""9.2: Renkli BMP'yi griye çevirme ve dosya boyutunu inceleme."""

import argparse
import struct
from pathlib import Path

import cv2
import numpy as np

KLASOR = Path(__file__).resolve().parent


def resim_oku(yol):
    resim = cv2.imread(str(yol), cv2.IMREAD_COLOR)
    if resim is None:
        raise OSError(f"Resim okunamadı: {yol}")
    return resim


def kaydet(yol, resim):
    if not cv2.imwrite(str(yol), resim):
        raise OSError(f"Resim kaydedilemedi: {yol}")


def bmp_bilgisi(yol):
    """Bu ödevdeki sıkıştırılmamış 8 ve 24 bit BMP başlıklarını okur."""
    yol = Path(yol)
    with yol.open("rb") as dosya:
        veri = dosya.read(54)
    if len(veri) != 54 or veri[:2] != b"BM":
        raise ValueError(f"Geçersiz BMP: {yol}")
    dib = struct.unpack_from("<I", veri, 14)[0]
    genislik, yukseklik = struct.unpack_from("<ii", veri, 18)
    bit = struct.unpack_from("<H", veri, 28)[0]
    sikistirma = struct.unpack_from("<I", veri, 30)[0]
    if dib != 40 or bit not in (8, 24) or sikistirma != 0 or genislik <= 0:
        raise ValueError(f"Desteklenmeyen BMP biçimi: {yol}")
    renk_sayisi = struct.unpack_from("<I", veri, 46)[0]
    palet = (renk_sayisi or 256) * 4 if bit == 8 else 0
    baslik = 14 + dib
    satir = ((genislik * bit + 31) // 32) * 4
    boyut = yol.stat().st_size
    hesaplanan = baslik + palet + satir * abs(yukseklik)
    if (struct.unpack_from("<I", veri, 10)[0] != baslik + palet
            or struct.unpack_from("<I", veri, 2)[0] != boyut
            or boyut != hesaplanan):
        raise ValueError(f"BMP boyutu başlık hesabıyla uyuşmuyor: {yol}")
    return dict(baslik=baslik, palet=palet, satir=satir, boyut=boyut,
                hesaplanan=hesaplanan)


def calistir(goster=True, kaynak=None, klasor=None):
    klasor = KLASOR if klasor is None else Path(klasor)
    kaynak = KLASOR.parent / "girdi/renkli.bmp" if kaynak is None else Path(kaynak)
    girdi = klasor / "girdi"
    cikti = klasor / "cikti"
    girdi.mkdir(parents=True, exist_ok=True)
    cikti.mkdir(parents=True, exist_ok=True)

    renkli_yol = girdi / "renkli_1000.bmp"
    gri_yol = cikti / "gri.bmp"
    renkli = cv2.resize(resim_oku(kaynak), (1000, 1000), interpolation=cv2.INTER_LINEAR)
    kaydet(renkli_yol, renkli)
    renkli = resim_oku(renkli_yol)
    gri = cv2.cvtColor(renkli, cv2.COLOR_BGR2GRAY)
    kaydet(gri_yol, gri)
    kaydet(cikti / "yan_yana.png", np.hstack([renkli, cv2.cvtColor(gri, cv2.COLOR_GRAY2BGR)]))

    renkli_bilgi = bmp_bilgisi(renkli_yol)
    gri_bilgi = bmp_bilgisi(gri_yol)
    satirlar = [
        "Çözünürlük: 1000 x 1000 = 1.000.000 piksel.",
        "uint8: kanal başına 1 bayt; renkli BGR 3 kanal, gri 1 kanal.",
        "Ham piksel verisi: renkli 3.000.000 bayt, gri 1.000.000 bayt.",
        "BMP satırları 4 baytın katına tamamlanır: ((genişlik * kanal + 3) // 4) * 4.",
        "Bu genişlikte renkli 3000, gri 1000 bayt/satır; ek satır dolgusu 0 bayt.",
        "Gri BMP paleti: 256 giriş x 4 bayt = 1024 bayt (B, G, R, ayrılmış bayt).",
    ]
    for ad, bilgi in [("Renkli", renkli_bilgi), ("Gri", gri_bilgi)]:
        satirlar.append(
            f"{ad}: başlık {bilgi['baslik']} + palet {bilgi['palet']} + "
            f"satır {bilgi['satir']} x 1000 = {bilgi['hesaplanan']} bayt; "
            f"diskte {bilgi['boyut']} bayt.")
    oran = renkli_bilgi["boyut"] / gri_bilgi["boyut"]
    satirlar.append(f"Gerçek renkli/gri disk boyutu oranı: {oran:.9f}. "
                    "Ham piksel oranı 3 olsa da başlık ve palet nedeniyle disk oranı tam 3 değildir.")
    (cikti / "boyutlar.txt").write_text("\n".join(satirlar) + "\n", encoding="utf-8")
    if goster:
        try:
            cv2.imshow("Renkli BMP", renkli)
            cv2.imshow("Gri BMP", gri)
            cv2.waitKey(0)
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    komut = argparse.ArgumentParser(description=__doc__)
    komut.add_argument("--gosterme", action="store_true", help="Pencere açmadan çıktı üret.")
    calistir(goster=not komut.parse_args().gosterme)
