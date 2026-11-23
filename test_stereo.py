import unittest

import numpy as np

from soru3_sad_ssd_zncc import blok_eslestirme, maliyet_sad, maliyet_ssd, maliyet_zncc
from soru4_sgbm_sifirdan import kendi_sgbm, yonlu_maliyet_topla
from stereo_eslestirme import colorize_disparity, compute_disparity_bm, compute_disparity_sgbm
from stereo_eslestirme_odev import soru2_kendi_blok_eslestirme


def sentetik_cift(kazanc=1.0, ofset=0):
    """Rektifiye, 6 px kaydırılmış çift. Sağ parlaklık değişimi kırpma yapmaz."""
    rng = np.random.default_rng(42)
    sol = rng.integers(25, 141, (80, 128), dtype=np.uint8)
    sag = rng.integers(25, 141, sol.shape, dtype=np.uint8)
    sag[:, :-6] = np.rint(kazanc * sol[:, 6:] + ofset).astype(np.uint8)
    return sol, sag


class StereoTest(unittest.TestCase):
    def test_uint8_costs_do_not_wrap(self):
        left = np.full((3, 3), 10, dtype=np.uint8)
        right = np.full((3, 3), 250, dtype=np.uint8)
        self.assertEqual(maliyet_sad(left, right, 3)[1, 1], 9 * 240)
        self.assertEqual(maliyet_ssd(left, right, 3)[1, 1], 9 * 240**2)

    def test_known_shift(self):
        left, right = sentetik_cift()
        roi = np.s_[12:-12, 24:-12]
        for match in (compute_disparity_bm, compute_disparity_sgbm):
            disp = match(left, right, num_disparities=16)
            self.assertGreater(np.mean(np.abs(disp[roi] - 6) < 0.5), 0.99)
        for cost in (maliyet_sad, maliyet_ssd, maliyet_zncc):
            np.testing.assert_array_equal(blok_eslestirme(left, right, cost, 16, 5)[roi], 6)
        np.testing.assert_array_equal(kendi_sgbm(left, right, 16, 5)[roi], 6)

    def test_windows_must_fit_both_images(self):
        left, right = sentetik_cift()
        for disp in (
            soru2_kendi_blok_eslestirme(left, right, 16, 5),
            kendi_sgbm(left, right, 16, 5),
        ):
            np.testing.assert_array_equal(disp[:2, :], -1)
            np.testing.assert_array_equal(disp[:, :2], -1)
            self.assertTrue(np.all(disp[2:-2, 2:12] <= np.arange(10)))

    def test_zncc_rejects_constant_windows(self):
        image = np.full((9, 9), 80, dtype=np.uint8)
        np.testing.assert_array_equal(blok_eslestirme(image, image, maliyet_zncc, 3, 5), -1)

    def test_zncc_with_gain_and_offset(self):
        left, right = sentetik_cift(1.5, 35)
        disp = blok_eslestirme(left, right, maliyet_zncc, 16, 5)
        np.testing.assert_array_equal(disp[12:-12, 24:-12], 6)

    def test_no_smoothing_matches_block_matching(self):
        left, right = sentetik_cift()
        np.testing.assert_array_equal(
            kendi_sgbm(left, right, 16, 5, p1=0, p2=0),
            soru2_kendi_blok_eslestirme(left, right, 16, 5),
        )

    def test_path_penalty_depends_on_disparity_change(self):
        cost = np.array([[[0, 20, 20], [0, 0, 0]]], dtype=np.float32)
        path = yonlu_maliyet_topla(cost, 0, 1, p1=2, p2=7)
        np.testing.assert_array_equal(path[0, 1], [0, 2, 7])

    def test_invalid_and_zero_have_different_colours(self):
        jet = colorize_disparity(np.array([[-1, 0, 7]], dtype=np.float32), 8)
        np.testing.assert_array_equal(jet[0, 0], [0, 0, 0])
        self.assertTrue(jet[0, 1].any())
        self.assertTrue(jet[0, 2].any())


if __name__ == "__main__":
    unittest.main()
