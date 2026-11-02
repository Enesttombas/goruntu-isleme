# 6.6 — Hough Line, CUDA ile paralel

**Ödev:** Hough Line fonksiyonu CUDA ile paralel gerçeklenir ve test edilir.

- [hough_line.cu](hough_line.cu): her GPU iş parçacığı bir (kenar pikseli, theta) çiftini işler. rho'yu hesaplar, oylama matrisindeki hücreye `atomicAdd` ile 1 oy ekler. Sin/cos tablosu CPU'da bir kez hesaplanıp GPU'ya bir kez kopyalanır.
- [cuda_ile.py](cuda_ile.py): `.cu` dosyasını DLL olarak derler (`nvcc -shared`), `ctypes` ile çağırır. Sonucu 6.2'deki kendi CPU kodumla karşılaştırır, doğruları çizer.

```powershell
python hafta6/6.6-hough-line-cuda/cuda_ile.py
```

Gereken: NVIDIA GPU, CUDA Toolkit (`nvcc`), Visual Studio Build Tools (`vcvars64.bat`). Test makinesi: RTX 4080 Laptop, CUDA 13.4.

## Test ([test.txt](cikti/test.txt))

| Test | Kenar pikseli | CPU/GPU matris farkı | CPU | GPU (kopyalama dahil) | Yalnız çekirdek |
|---|---|---|---|---|---|
| sudoku, Canny + median | 7129 | **0** | 121 ms | 3,3 ms | 0,06 ms |
| sudoku, yalnız Canny | 29537 | **0** | 518 ms | 8,3 ms | 0,20 ms |

- **Doğruluk:** GPU oylama matrisi CPU matrisiyle hücre hücre aynı (mutlak fark toplamı 0). Bunun için GPU da `double` ve `rint` (yarımda çifte yuvarlama) kullanıyor; NumPy'deki `np.rint` ile aynı.
- **Hız:** süreler 20 tekrarın ortancası. GPU 37-62 kat hızlı. GPU süresinin çoğu bellek ayırma ve kopyalama; oylama çekirdeği 1 ms'nin altında.
- GPU oylamasıyla eşik 100'de 14 doğru bulundu (`cikti/dogrular_cuda.png`), 6.2 ile aynı.
