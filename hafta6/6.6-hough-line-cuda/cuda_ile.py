"""Hafta 6 / 6.6: Hough Line oylaması CUDA ile paralel; CPU sürümüyle (6.2) birebir karşılaştırma.

Derleme (DLL yoksa betik kendisi derler):
    vcvars64.bat && nvcc -O2 -shared -o hough_line.dll hough_line.cu
"""

import ctypes
import importlib.util
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

KLASOR = Path(__file__).resolve().parent
GIRDI = KLASOR.parent / "girdi" / "sudoku.png"
CIKTI = KLASOR / "cikti"
CU = KLASOR / "hough_line.cu"
DLL = KLASOR / "hough_line.dll"
VCVARS = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat")

THETA_ADIM = 180
ESIK = 100
TEKRAR = 20  # süre ölçümü için tekrar sayısı


def cpu_modulu():
    """6.2'deki kendi CPU Hough kodumuz: karşılaştırma bununla yapılır."""
    yol = KLASOR.parent / "6.2-hough-line-kendi" / "kendi_fonksiyonum.py"
    spec = importlib.util.spec_from_file_location("hough_cpu", yol)
    if spec is None or spec.loader is None:
        raise ImportError(f"CPU modülü yüklenemedi: {yol}")
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)
    return modul


def dll_yukle() -> ctypes.CDLL:
    if not DLL.exists() or DLL.stat().st_mtime < CU.stat().st_mtime:
        komut = f'"{VCVARS}" && nvcc -O2 -shared -o "{DLL}" "{CU}"'
        subprocess.run(komut, shell=True, check=True, cwd=KLASOR, stdout=subprocess.DEVNULL)
    dll = ctypes.CDLL(str(DLL))
    ip = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
    dp = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
    dll.hough_line_cuda.argtypes = [ip, ip, ctypes.c_int, dp, dp, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int, ip, ctypes.POINTER(ctypes.c_float)]
    dll.hough_line_cuda.restype = ctypes.c_int
    return dll


def hough_cuda(dll: ctypes.CDLL, kenar: np.ndarray, cos_t: np.ndarray, sin_t: np.ndarray):
    """GPU'da oylama. Dönüş: (oylama matrisi, köşegen kaydırması, çekirdek süresi ms)."""
    yukseklik, genislik = kenar.shape
    kosegen = int(np.ceil(np.hypot(yukseklik, genislik)))
    ys, xs = np.nonzero(kenar)
    xs = np.ascontiguousarray(xs, dtype=np.int32)
    ys = np.ascontiguousarray(ys, dtype=np.int32)
    oylama = np.zeros((2 * kosegen + 1, len(cos_t)), dtype=np.int32)
    kernel_ms = ctypes.c_float()
    kod = dll.hough_line_cuda(xs, ys, len(xs), cos_t, sin_t, len(cos_t),
                              kosegen, oylama.shape[0], oylama, ctypes.byref(kernel_ms))
    if kod != 0:
        raise RuntimeError(f"CUDA hata kodu: {kod}")
    return oylama, kosegen, kernel_ms.value


def sure_olc(fonksiyon) -> float:
    """TEKRAR kez çalıştırıp ortanca süreyi milisaniye olarak döner."""
    sureler = []
    for _ in range(TEKRAR):
        t0 = time.perf_counter()
        fonksiyon()
        sureler.append((time.perf_counter() - t0) * 1000)
    return float(np.median(sureler))


def karsilastir(cpu, dll, kenar, cos_t, sin_t, ad: str) -> tuple[list[str], int]:
    """Aynı kenar resminde CPU ve GPU oylama matrislerini ve sürelerini karşılaştırır."""
    oylama_gpu, _, kernel_ms = hough_cuda(dll, kenar, cos_t, sin_t)
    oylama_cpu, _ = cpu.hough_oylama(kenar, cos_t, sin_t)
    fark = int(np.abs(oylama_gpu.astype(np.int64) - oylama_cpu).sum())
    cpu_ms = sure_olc(lambda: cpu.hough_oylama(kenar, cos_t, sin_t))
    gpu_ms = sure_olc(lambda: hough_cuda(dll, kenar, cos_t, sin_t))
    satirlar = [
        f"{ad}: {int((kenar > 0).sum())} kenar pikseli x {len(cos_t)} theta",
        f"  CPU ile GPU oylama matrisi mutlak fark toplamı: {fark}",
        f"  Süre (ortanca, {TEKRAR} tekrar): CPU {cpu_ms:.1f} ms, GPU {gpu_ms:.1f} ms "
        f"(yalnız çekirdek {kernel_ms:.3f} ms), hızlanma {cpu_ms / gpu_ms:.0f}x",
    ]
    return satirlar, fark


def calistir() -> None:
    renkli = cv2.imread(str(GIRDI), cv2.IMREAD_COLOR)
    if renkli is None:
        raise FileNotFoundError(f"Resim okunamadı: {GIRDI}")
    CIKTI.mkdir(exist_ok=True)

    cpu = cpu_modulu()
    dll = dll_yukle()

    gri = cv2.cvtColor(renkli, cv2.COLOR_BGR2GRAY)
    kenar = cv2.medianBlur(cv2.Canny(gri, 50, 150), 3)  # 6.1 / 6.2 ile aynı kenar resmi
    _, cos_t, sin_t = cpu.sin_cos_tablosu(THETA_ADIM)
    cos_t, sin_t = np.ascontiguousarray(cos_t), np.ascontiguousarray(sin_t)

    oylama, kosegen, _ = hough_cuda(dll, kenar, cos_t, sin_t)
    tepeler = cpu.tepeleri_bul(oylama, ESIK)
    cizim = renkli.copy()
    for rho_i, theta_i, _ in tepeler:
        cpu.dogru_ciz(cizim, rho_i - kosegen, cos_t[theta_i], sin_t[theta_i])
    if not cv2.imwrite(str(CIKTI / "dogrular_cuda.png"), cizim):
        raise OSError("Çizim kaydedilemedi")

    # Daha büyük yük: median'sız Canny kenarı (yaklaşık 4 kat kenar pikseli).
    buyuk = cv2.Canny(gri, 50, 150)
    satirlar1, fark1 = karsilastir(cpu, dll, kenar, cos_t, sin_t, "Test 1 - sudoku.png")
    satirlar2, fark2 = karsilastir(cpu, dll, buyuk, cos_t, sin_t, "Test 2 - sudoku.png, median'sız Canny")
    satirlar = satirlar1 + [f"  GPU oylamasıyla eşik {ESIK}: {len(tepeler)} doğru"] + satirlar2

    (CIKTI / "test.txt").write_text("\n".join(satirlar) + "\n", encoding="utf-8")
    print("\n".join(satirlar))
    if fark1 or fark2:
        raise SystemExit("HATA: CPU ve GPU oylama matrisleri farklı")


if __name__ == "__main__":
    calistir()
