"""Hafta 7 / 7.9: Gürültülü resimlerden ters DFT ile orijinale dönmeye çalışma, ZNCC ile ölçme.

Gürültülü resmin harmonik matrisini olduğu gibi ters çevirmek gürültülü resmi aynen geri verir,
çünkü DFT kayıpsızdır. Orijinale yaklaşmak için ters dönüşümden önce harmonik matriste yüksek
frekanslar (gürültünün çoğu orada) Gauss alçak geçiren maske ile bastırılır.

ZNCC (Zero Mean Normalized Cross Correlation):
    ZNCC(a, b) = sum((a - ort_a) * (b - ort_b)) / (N * std_a * std_b),  sonuç [-1, 1], 1 = birebir benzer
"""

import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

KLASOR = Path(__file__).resolve().parent
CIKTI = KLASOR / "cikti"
for klasor in ["7.1-sinus-vektor", "7.2-gurultu-ekleme", "7.4-dft-harmonik", "7.7-2b-dft", "7.8-resim-gurultu"]:
    sys.path.insert(0, str(KLASOR.parent / klasor))

from dft2 import dft2, idft2  # noqa: E402
from resim_gurultu import gurultulu_resimler  # noqa: E402

D0_LISTESI = [10, 20, 30, 40, 50, 60, 80, 100, 150, 200]


def zncc(a, b):
    a = a.astype(float) - a.mean()
    b = b.astype(float) - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def mse(a, b):
    return float(np.mean((a.astype(float) - b.astype(float)) ** 2))


def psnr(a, b):
    return float(10 * np.log10(255 ** 2 / mse(a, b)))


def ssim(a, b):
    """Wang vd. (2004) SSIM, 11x11 Gauss pencere (sigma 1.5), haritanın ortalaması."""
    a, b = a.astype(float), b.astype(float)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    def bulanik(x):
        return cv2.GaussianBlur(x, (11, 11), 1.5)

    mu_a, mu_b = bulanik(a), bulanik(b)
    var_a = bulanik(a * a) - mu_a ** 2
    var_b = bulanik(b * b) - mu_b ** 2
    kov = bulanik(a * b) - mu_a * mu_b
    harita = ((2 * mu_a * mu_b + c1) * (2 * kov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))
    return float(harita.mean())


def uzaklik_matrisi(m, n):
    """Her harmoniğin DC'ye (frekans 0'a) uzaklığı. Matris ortalanmadan, köşeler düşük frekans."""
    v = np.minimum(np.arange(m), m - np.arange(m))
    u = np.minimum(np.arange(n), n - np.arange(n))
    return np.hypot(v[:, None], u[None, :])


def ters_dft_ile_geri_don(gurultulu, d0=None):
    harmonik = dft2(gurultulu)
    if d0 is not None:
        harmonik = harmonik * np.exp(-uzaklik_matrisi(*harmonik.shape) ** 2 / (2 * d0 ** 2))
    return np.clip(np.rint(np.real(idft2(harmonik))), 0, 255).astype(np.uint8)


def olcumler(orijinal, resim):
    return zncc(orijinal, resim), mse(orijinal, resim), psnr(orijinal, resim), ssim(orijinal, resim)


def calistir():
    CIKTI.mkdir(exist_ok=True)
    resimler = gurultulu_resimler()
    orijinal = resimler["orijinal"]

    sekil, eksen = plt.subplots(figsize=(9, 5))
    satirlar = []
    print(f"{'resim':10s} {'yöntem':30s} {'ZNCC':>7s} {'MSE':>8s} {'PSNR':>7s} {'SSIM':>6s}")
    for ad in ["gauss", "tuz_biber"]:
        gurultulu = resimler[ad]
        dogrudan = ters_dft_ile_geri_don(gurultulu)
        skorlar = [zncc(orijinal, ters_dft_ile_geri_don(gurultulu, d0)) for d0 in D0_LISTESI]
        en_iyi_d0 = D0_LISTESI[int(np.argmax(skorlar))]
        en_iyi = ters_dft_ile_geri_don(gurultulu, en_iyi_d0)
        eksen.plot(D0_LISTESI, skorlar, "o-", label=ad)
        eksen.axhline(zncc(orijinal, dogrudan), ls=":", color=eksen.lines[-1].get_color())

        print(f"{ad} D0 -> ZNCC: " + ", ".join(f"{d0}: {s:.4f}" for d0, s in zip(D0_LISTESI, skorlar)))
        for yontem, sonuc in [("gürültülü resim", gurultulu), ("doğrudan ters DFT", dogrudan), (f"Gauss AG D0={en_iyi_d0} + ters DFT", en_iyi)]:
            z, m, p, s = olcumler(orijinal, sonuc)
            print(f"{ad:10s} {yontem:30s} {z:7.4f} {m:8.1f} {p:7.2f} {s:6.3f}")
        print(f"{ad}: max|doğrudan ters DFT - gürültülü| = {np.abs(dogrudan.astype(int) - gurultulu).max()}")
        cv2.imwrite(str(CIKTI / f"{ad}_ters_dft_d0_{en_iyi_d0}.png"), en_iyi)
        satirlar.append(np.hstack([gurultulu, np.full((orijinal.shape[0], 8), 255, np.uint8), dogrudan,
                                   np.full((orijinal.shape[0], 8), 255, np.uint8), en_iyi]))

    eksen.set_xlabel("Gauss alçak geçiren kesim frekansı D0")
    eksen.set_ylabel("ZNCC (orijinal ile)")
    eksen.set_title("Ters DFT öncesi alçak geçiren maske, noktalı çizgi: maskesiz ters DFT")
    eksen.grid(alpha=0.3)
    eksen.legend()
    sekil.tight_layout()
    sekil.savefig(CIKTI / "zncc_d0.png", dpi=120)
    plt.close(sekil)
    cv2.imwrite(str(CIKTI / "gurultulu_dogrudan_filtreli.png"),
                np.vstack([satirlar[0], np.full((8, satirlar[0].shape[1]), 255, np.uint8), satirlar[1]]))

    # ZNCC'nin parlaklık/kontrast değişimine duyarsızlığı
    koyu = np.clip(0.5 * orijinal.astype(float) + 20, 0, 255).astype(np.uint8)
    z, m, p, s = olcumler(orijinal, koyu)
    print(f"Orijinal vs 0.5*orijinal+20: ZNCC {z:.4f}, MSE {m:.1f}, PSNR {p:.2f}, SSIM {s:.3f}")


if __name__ == "__main__":
    calistir()
