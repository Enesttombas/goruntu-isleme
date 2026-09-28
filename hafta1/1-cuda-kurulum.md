# Hafta 1 / Ödev 1 — CUDA destekli OpenCV kurulumu

Temel gerçek: `pip install opencv-python` **CPU** sürümüdür; `cv2.cuda` modülü var görünür
ama `getCudaEnabledDeviceCount()` 0 döner. CUDA için OpenCV'nin `WITH_CUDA=ON` ile
**kaynaktan derlenmesi** gerekir.

## Yol A — Yerel (VS Code, bu laptop)

Bu makinede zaten hazır (2026-09-28 ölçüldü): OpenCV `4.15.0-dev`, RTX 4080 Laptop GPU,
`getCudaEnabledDeviceCount() = 1`. Doğrulama:

```bash
python cuda_kontrol.py
```

Sıfırdan kurmak gerekseydi: CUDA Toolkit + cuDNN + Visual Studio (MSVC) + CMake kur,
`opencv` ve `opencv_contrib` kaynaklarını indir, CMake'te
`WITH_CUDA=ON, OPENCV_DNN_CUDA=ON, CUDA_ARCH_BIN=8.9 (RTX 40), BUILD_opencv_python3=ON,
OPENCV_EXTRA_MODULES_PATH=opencv_contrib/modules` ile derle, `install` et.

## Yol B — Colab

1. `Çalışma zamanı > Çalışma zamanı türünü değiştir > T4 GPU`.
2. Hücre: `!nvidia-smi` — GPU görünmeli.
3. Derleme hücresi (~40-60 dk sürer; bir kez derle, Drive'a kaydet):

```bash
!git clone --depth 1 https://github.com/opencv/opencv.git
!git clone --depth 1 https://github.com/opencv/opencv_contrib.git
!mkdir -p opencv/build
%cd opencv/build
!cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_CUDA=ON -DCUDA_ARCH_BIN=7.5 \
  -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
  -DBUILD_opencv_python3=ON -DPYTHON3_EXECUTABLE=$(which python3) \
  -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF
!make -j$(nproc) && make install
```

`CUDA_ARCH_BIN=7.5` T4 içindir. Bitince `pip uninstall -y opencv-python opencv-contrib-python`
(pip'in CPU sürümü derlediğini gölgelemesin), çalışma zamanını yeniden başlat,
`cuda_kontrol.py` içeriğini hücrede çalıştır.

## Öğrenilecek fikir

GPU'da iş = `upload` (RAM→VRAM) + işlem + `download` (VRAM→RAM). Küçük resimde
kopyalama maliyeti işlemi geçer; GPU ancak büyük/çok işlemde kazandırır.
Ölçüm (3 resim resize): CPU 0,097 sn, GPU 0,248 sn — ilk CUDA çağrısı da ısınma bedeli öder.
