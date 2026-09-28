# Görüntü İşleme (EBLG351) — Ödevler

Python + OpenCV. Her hafta kendi klasöründe: kod, girdi görselleri ve çıktılar.

| Hafta | Ödev | Dosya |
|---|---|---|
| 1 | CUDA destekli OpenCV kurulumu | [hafta1/1-cuda-kurulum.md](hafta1/1-cuda-kurulum.md), [hafta1/cuda_kontrol.py](hafta1/cuda_kontrol.py) |
| 1 | Klasördeki resimleri 1024x768'e ölçekleme | [hafta1/resize_klasor.py](hafta1/resize_klasor.py) |
| 2 | Nicemleme, çoklu thread parlaklık ve histogram | [hafta2/](hafta2/) |

## Çalıştırma

```bash
pip install opencv-python numpy pytest
cd hafta1
python cuda_kontrol.py              # CUDA'lı OpenCV gerekir
python resize_klasor.py girdi cikti # --gpu ile CUDA sürümü
python -m pytest -q
```

Ortam: RTX 4080 Laptop GPU, OpenCV 4.15.0-dev (CUDA ile derlenmiş), Python 3.14.
