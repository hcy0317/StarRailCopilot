import unittest

import numpy as np

from module.base.utils import color_mask, color_similarity_2d, extract_letters


class TestUpstreamImageCompatibility(unittest.TestCase):
    @staticmethod
    def color_distance(image, color):
        delta = image.astype(np.int16) - np.array(color, dtype=np.int16)
        positive = np.maximum(delta, 0).max(axis=2)
        negative = np.maximum(-delta, 0).max(axis=2)
        return np.minimum(positive + negative, 255)

    def test_color_similarity_preserves_relic_detection_for_both_image_sizes(self):
        rng = np.random.default_rng(0)
        for shape in ((150, 199, 3), (150, 200, 3), (180, 200, 3)):
            image = rng.integers(0, 256, size=shape, dtype=np.uint8)
            original = image.copy()
            for color in ((255, 255, 255), (252, 200, 109), (0, 128, 255)):
                with self.subTest(shape=shape, color=color):
                    expected = (255 - self.color_distance(image, color)).astype(np.uint8)

                    actual = color_similarity_2d(image, color)

                    np.testing.assert_array_equal(actual, expected)
                    np.testing.assert_array_equal(image, original)

    def test_color_mask_uses_distance_tolerance_including_boundary(self):
        pixels = np.array([[[100, 100, 100], [130, 100, 100], [131, 100, 100],
                            [115, 85, 100], [255, 0, 100]]], dtype=np.uint8)
        for image in (pixels, np.tile(pixels, (150, 40, 1))):
            for threshold in (0, 30, 75, 255):
                with self.subTest(shape=image.shape, threshold=threshold):
                    expected = np.where(self.color_distance(image, (100, 100, 100)) <= threshold,
                                        255, 0).astype(np.uint8)

                    np.testing.assert_array_equal(color_mask(image, (100, 100, 100), threshold), expected)

    def test_letter_extraction_preserves_white_and_orange_relic_text(self):
        rng = np.random.default_rng(1)
        for shape in ((20, 120, 3), (150, 200, 3)):
            image = rng.integers(0, 256, size=shape, dtype=np.uint8)
            original = image.copy()
            for letter in ((255, 255, 255), (235, 161, 66)):
                for threshold in (128, 255):
                    with self.subTest(shape=shape, letter=letter, threshold=threshold):
                        distance = self.color_distance(image, letter)
                        expected = np.clip(np.rint(distance * (255.0 / threshold)), 0, 255).astype(np.uint8)

                        np.testing.assert_array_equal(extract_letters(image, letter, threshold), expected)
                        np.testing.assert_array_equal(image, original)


if __name__ == '__main__':
    unittest.main()
