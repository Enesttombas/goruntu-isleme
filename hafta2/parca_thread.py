"""Ödev 2: Dört görüntü parçasını farklı işlemlerle paralel ve sıralı işle.

Kullanım: python parca_thread.py [girdi] [cikti] [--tekrar 20]
Varsayılan: dama.jpg ve cikti klasörü.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

KLASOR = Path(__file__).resolve().parent


def _bol(gri):
    h, w = gri.shape
    y, x = h // 2, w // 2
    return [(gri[:y, :x], (slice(0, y), slice(0, x))),
            (gri[:y, x:], (slice(0, y), slice(x, w))),
            (gri[y:, :x], (slice(y, h), slice(0, x))),
            (gri[y:, x:], (slice(y, h), slice(x, w)))]


def _isle(girdi):
    sira, parca = girdi
    x = parca.astype(np.float32)
    if sira == 0:
        x += 50
    elif sira == 1:
        x -= 50
    elif sira == 2:
        x = 255 - x
    else:
        x = np.sqrt(x / 255) * 255  # Gamma 0.5
    return np.clip(x, 0, 255).astype(np.uint8)


def parcalari_isle(gri, paralel=True):
    bolumler = _bol(gri)
    girdiler = [(i, parca) for i, (parca, _) in enumerate(bolumler)]
    if paralel:
        with ThreadPoolExecutor(max_workers=4) as havuz:
            sonuclar = list(havuz.map(_isle, girdiler))
    else:
        sonuclar = [_isle(girdi) for girdi in girdiler]
    sonuc = np.empty_like(gri)
    for (_, konum), parca in zip(bolumler, sonuclar):
        sonuc[konum] = parca
    return sonuc


def _sure_olc(gri, paralel, tekrar):
    baslangic = perf_counter()
    for _ in range(tekrar):
        sonuc = parcalari_isle(gri, paralel)
    return (perf_counter() - baslangic) / tekrar, sonuc


def main():
    ap = argparse.ArgumentParser(description="Dört parçayı parlaklık işlemleriyle karşılaştır.")
    ap.add_argument("girdi", nargs="?", type=Path, default=KLASOR / "dama.jpg")
    ap.add_argument("cikti", nargs="?", type=Path, default=KLASOR / "cikti")
    ap.add_argument("--tekrar", type=int, default=20)
    args = ap.parse_args()
    if args.tekrar < 1:
        raise SystemExit("--tekrar en az 1 olmalı.")
    gri = cv2.imread(str(args.girdi), cv2.IMREAD_GRAYSCALE)
    if gri is None:
        raise SystemExit(f"Görüntü okunamadı: {args.girdi}")
    tek, _ = _sure_olc(gri, False, args.tekrar)
    dort, sonuc = _sure_olc(gri, True, args.tekrar)
    args.cikti.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.cikti / "parca_sonuc.png"), sonuc):
        raise SystemExit("parca_sonuc.png yazılamadı.")
    print(f"Ortalama süre ({args.tekrar} tekrar): 1 thread {tek:.6f} sn, 4 thread {dort:.6f} sn")
    print("NumPy işlemleri GIL'i bırakır; küçük görüntüde thread açma maliyeti kazancı yiyebilir, sonucu ölçümle yorumlayın.")


if __name__ == "__main__":
    main()
