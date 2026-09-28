"""Hafta 1 / Ödev 2: Bir klasördeki tüm resimleri 1024x768 boyutuna ölçekle.

Kullanım:
    python resize_klasor.py girdi cikti
    python resize_klasor.py girdi cikti --gpu      # CUDA'lı OpenCV ile
"""
import argparse
import time
from pathlib import Path

import cv2

HEDEF_GENISLIK = 1024
HEDEF_YUKSEKLIK = 768
RESIM_UZANTILARI = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def olcekle_cpu(resim):
    # cv2.resize boyutu (GENİŞLİK, YÜKSEKLİK) sırasıyla ister; resim.shape ise (yükseklik, genişlik, kanal) verir.
    return cv2.resize(resim, (HEDEF_GENISLIK, HEDEF_YUKSEKLIK), interpolation=cv2.INTER_AREA)


def olcekle_gpu(resim):
    gpu_resim = cv2.cuda_GpuMat()
    gpu_resim.upload(resim)                      # RAM -> ekran kartı belleği
    gpu_sonuc = cv2.cuda.resize(gpu_resim, (HEDEF_GENISLIK, HEDEF_YUKSEKLIK), interpolation=cv2.INTER_AREA)
    return gpu_sonuc.download()                  # ekran kartı -> RAM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("girdi", type=Path)
    ap.add_argument("cikti", type=Path)
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()

    if args.gpu and cv2.cuda.getCudaEnabledDeviceCount() == 0:
        raise SystemExit("CUDA destekli OpenCV veya GPU bulunamadı; --gpu olmadan çalıştır.")
    olcekle = olcekle_gpu if args.gpu else olcekle_cpu

    args.cikti.mkdir(parents=True, exist_ok=True)
    dosyalar = sorted(p for p in args.girdi.iterdir() if p.suffix.lower() in RESIM_UZANTILARI)
    if not dosyalar:
        raise SystemExit(f"{args.girdi} içinde resim yok.")

    baslangic = time.perf_counter()
    for yol in dosyalar:
        resim = cv2.imread(str(yol))
        if resim is None:                        # bozuk/okunamayan dosya: atla, çökme
            print(f"ATLANDI (okunamadı): {yol.name}")
            continue
        yeni = olcekle(resim)
        cv2.imwrite(str(args.cikti / yol.name), yeni)
        print(f"{yol.name}: {resim.shape[1]}x{resim.shape[0]} -> {yeni.shape[1]}x{yeni.shape[0]}")
    print(f"{len(dosyalar)} dosya, {time.perf_counter() - baslangic:.3f} sn ({'GPU' if args.gpu else 'CPU'})")


if __name__ == "__main__":
    main()
