import unittest
import numpy as np
import utils


class TestGenerateBoundingBox(unittest.TestCase):

    def test_generates_correct_bbox(self):
        mask = np.zeros((480, 640), dtype=np.bool)
        for i in range(0, 50):
            mask[120 + i, 360 - i:360 + i] = True
        bbox = utils.generate_bounding_box_from_mask(mask)
        self.assertEqual([311, 121, 408, 169], bbox)

    def test_raises_if_no_pixels(self):
        mask = np.zeros((480, 640), dtype=np.bool)
        with self.assertRaises(ValueError):
            utils.generate_bounding_box_from_mask(mask)
