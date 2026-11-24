import unittest

import numpy as np

from PIL import Image

from src.preprocessing import preprocess_image


class TestImagePreprocessing(unittest.TestCase):

    def test_rgb_image(self):
        image = Image.new("RGB", (256, 256), color=(120, 180, 80))

        result = preprocess_image(image)

        self.assertEqual(result.shape, (1, 224, 224, 3))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.all(result >= -1.0))
        self.assertTrue(np.all(result <= 1.0))

    def test_rgba_image(self):
        image = Image.new("RGBA", (256, 256), color=(120, 180, 80, 128))

        result = preprocess_image(image)

        self.assertEqual(result.shape, (1, 224, 224, 3))

    def test_grayscale_image(self):
        image = Image.new("L", (256, 256), color=150)

        result = preprocess_image(image)

        self.assertEqual(result.shape, (1, 224, 224, 3))

    def test_invalid_input(self):
        with self.assertRaises(TypeError):
            preprocess_image("not an image")


if __name__ == "__main__":
    unittest.main()
