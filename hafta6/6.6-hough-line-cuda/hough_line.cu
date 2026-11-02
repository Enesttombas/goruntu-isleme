// Hafta 6 / 6.6: Hough Line oylaması CUDA ile paralel.
// Her GPU iş parçacığı bir (kenar pikseli, theta) çiftini işler: rho'yu hesaplar ve
// oylama matrisindeki (rho, theta) hücresine atomicAdd ile 1 oy ekler.

#include <cuda_runtime.h>

__global__ void hough_oylama_kernel(const int* xs, const int* ys, int piksel_sayisi,
                                    const double* cos_t, const double* sin_t, int theta_sayisi,
                                    int kosegen, int* oylama)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)piksel_sayisi * theta_sayisi) return;

    int p = (int)(i / theta_sayisi);  // hangi kenar pikseli
    int t = (int)(i % theta_sayisi);  // hangi theta
    // CPU sürümündeki np.rint ile aynı yuvarlama (yarımda çifte yuvarla), aynı double hassasiyet.
    int rho = (int)rint(xs[p] * cos_t[t] + ys[p] * sin_t[t]);
    atomicAdd(&oylama[(rho + kosegen) * theta_sayisi + t], 1);
}

// Python'dan ctypes ile çağrılır. Dönüş: 0 = başarılı, diğerleri CUDA hata kodu.
extern "C" __declspec(dllexport)
int hough_line_cuda(const int* xs, const int* ys, int piksel_sayisi,
                    const double* cos_t, const double* sin_t, int theta_sayisi,
                    int kosegen, int rho_sayisi, int* oylama_cikti, float* kernel_ms)
{
    int *d_xs = 0, *d_ys = 0, *d_oylama = 0;
    double *d_cos = 0, *d_sin = 0;
    size_t oylama_boyut = (size_t)rho_sayisi * theta_sayisi * sizeof(int);
    cudaEvent_t basla, bitis;
    cudaError_t hata;

    cudaMalloc(&d_xs, piksel_sayisi * sizeof(int));
    cudaMalloc(&d_ys, piksel_sayisi * sizeof(int));
    cudaMalloc(&d_cos, theta_sayisi * sizeof(double));
    cudaMalloc(&d_sin, theta_sayisi * sizeof(double));
    cudaMalloc(&d_oylama, oylama_boyut);

    // Sin/cos tablosu CPU'da bir kez hesaplandı; GPU belleğine bir kez kopyalanır.
    cudaMemcpy(d_xs, xs, piksel_sayisi * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ys, ys, piksel_sayisi * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cos, cos_t, theta_sayisi * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sin, sin_t, theta_sayisi * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_oylama, 0, oylama_boyut);

    long long is_sayisi = (long long)piksel_sayisi * theta_sayisi;
    int blok = 256;
    int izgara = (int)((is_sayisi + blok - 1) / blok);

    cudaEventCreate(&basla);
    cudaEventCreate(&bitis);
    cudaEventRecord(basla);
    hough_oylama_kernel<<<izgara, blok>>>(d_xs, d_ys, piksel_sayisi, d_cos, d_sin,
                                          theta_sayisi, kosegen, d_oylama);
    cudaEventRecord(bitis);
    cudaEventSynchronize(bitis);
    cudaEventElapsedTime(kernel_ms, basla, bitis);

    cudaMemcpy(oylama_cikti, d_oylama, oylama_boyut, cudaMemcpyDeviceToHost);
    hata = cudaGetLastError();

    cudaEventDestroy(basla);
    cudaEventDestroy(bitis);
    cudaFree(d_xs); cudaFree(d_ys); cudaFree(d_cos); cudaFree(d_sin); cudaFree(d_oylama);
    return (int)hata;
}
