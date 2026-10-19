"""5.5 ortak yardımcı: X-ray yoğunluğunu güvenlik tarayıcısı kuralına göre renklendiren palet.

Gri görüntüde parlak = boş, koyu = yoğun. Yoğunluk d = 255 - gri değeridir.
Renk sırası sektör kuralını izler: düşük yoğunluk turuncu (organik), orta yeşil,
yüksek mavi (metal/inorganik), çok yüksek koyu lacivert. Boşluk beyaz kalır.
"""

import numpy as np

# (yoğunluk, (R, G, B))
DURAKLAR = (
    (0, (255, 255, 255)),   # boşluk
    (20, (255, 250, 225)),  # çok hafif yoğunluk, krem
    (45, (250, 195, 125)),  # açık turuncu
    (70, (235, 130, 50)),   # turuncu: organik
    (95, (170, 170, 60)),   # zeytin yeşili: karışık
    (120, (60, 150, 110)),  # yeşil
    (150, (40, 110, 200)),  # mavi: metal
    (190, (20, 50, 150)),   # koyu mavi
    (255, (10, 10, 60)),    # çok yoğun, lacivert
)


def palet_olustur() -> np.ndarray:
    """256 gri değeri için BGR renk tablosu (256, 3) uint8 üretir."""
    yogunluk = np.arange(256)
    xs = [d for d, _ in DURAKLAR]
    kanallar = [np.interp(yogunluk, xs, [renk[i] for _, renk in DURAKLAR]) for i in range(3)]
    rgb = np.stack(kanallar, axis=1)
    return np.rint(rgb[::-1][:, ::-1]).astype(np.uint8)  # gri g -> yoğunluk 255-g; RGB -> BGR
