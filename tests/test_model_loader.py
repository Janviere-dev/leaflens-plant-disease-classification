import unittest

from src.model_loader import load_artifacts


class TestModelLoader(unittest.TestCase):

    def test_load_artifacts(self):
        model, class_names, config = load_artifacts()

        # Verify model input and output
        self.assertEqual(model.input_shape, (None, 224, 224, 3))
        self.assertEqual(model.output_shape, (None, 38))

        # Verify class names
        self.assertEqual(len(class_names), 38)
        self.assertIn("Apple___Apple_scab", class_names)
        self.assertIn("Tomato___healthy", class_names)

        # Verify preprocessing configuration
        self.assertEqual(config["architecture"], "MobileNetV2")
        self.assertEqual(config["image_size"], [224, 224])
        self.assertEqual(config["color_mode"], "RGB")
        self.assertEqual(config["input_range"], [-1, 1])

        print("\nModel loader verification passed.")


if __name__ == "__main__":
    unittest.main()
