import unittest

import numpy as np

from goku_effect import (
    TELEPORT_SECONDS, make_speed_lines, render_teleport, salute_fingertips,
)


def salute_hand() -> np.ndarray:
    points = np.zeros((21, 2))
    points[0] = (100, 200)
    points[1:5] = [(80, 185), (70, 170), (65, 158), (62, 148)]
    points[5:9] = [(85, 150), (88, 120), (89, 100), (90, 80)]
    points[9:13] = [(100, 148), (101, 118), (102, 98), (103, 78)]
    points[13:17] = [(115, 150), (117, 135), (116, 150), (114, 160)]
    points[17:21] = [(128, 155), (128, 140), (126, 152), (124, 160)]
    return points


class SalutePoseTests(unittest.TestCase):
    def test_two_fingers_together_is_a_salute(self):
        tips = salute_fingertips(salute_hand())
        self.assertIsNotNone(tips)
        np.testing.assert_allclose(tips, (96.5, 79))

    def test_spread_v_sign_is_not_a_salute(self):
        points = salute_hand()
        points[10:13] = [(115, 118), (125, 98), (135, 78)]
        self.assertIsNone(salute_fingertips(points))

    def test_open_hand_is_not_a_salute(self):
        points = salute_hand()
        points[13:17] = [(115, 150), (117, 120), (118, 100), (119, 82)]
        points[17:21] = [(128, 155), (132, 128), (134, 110), (136, 95)]
        self.assertIsNone(salute_fingertips(points))


class TeleportRenderTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((120, 160, 3), 200, np.uint8)
        self.background = np.zeros_like(self.frame)
        self.mask = np.zeros((120, 160), np.float32)
        self.mask[20:110, 60:100] = 1.0
        self.lines = make_speed_lines(self.mask, seed=3)

    def test_person_is_gone_at_the_end_and_back_after_reverse(self):
        gone = render_teleport(self.frame, self.background, self.mask, self.mask, self.lines,
                               TELEPORT_SECONDS)
        self.assertEqual(gone[60, 80].tolist(), [0, 0, 0])
        back = render_teleport(self.frame, self.background, self.mask, self.mask, self.lines,
                               TELEPORT_SECONDS, reverse=True)
        self.assertEqual(back[60, 80].tolist(), [200, 200, 200])

    def test_lines_are_drawn_mid_effect(self):
        middle = render_teleport(self.frame, self.background, self.mask, self.mask, self.lines,
                                 TELEPORT_SECONDS * 0.6)
        self.assertGreater(middle[:, 55:105].max(), 0)
        self.assertEqual(middle[:, :20].max(), 0)


if __name__ == "__main__":
    unittest.main()
