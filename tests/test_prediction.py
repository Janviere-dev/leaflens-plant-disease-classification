
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.prediction import predict_image, format_class_name


class FakeModel:
    """A simple model that returns predefined prediction probabilities."""

    def __init__(self, probabilities):
        self.probabilities = np.array([probabilities], dtype=np.float32)

    def predict(self, image, verbose=0):
        return self.probabilities


class TestPrediction(unittest.TestCase):

    def setUp(self):
        self.image = Image.new("RGB", (256, 256), color=(120, 180, 80))

        self.class_names = [
            "Tomato___Early_blight",
            "Tomato___healthy",
            "Potato___Late_blight",
            "Apple___Apple_scab"
        ]

        self.config = {"architecture": "MobileNetV2"}

    @patch("src.prediction.load_artifacts")
    def test_disease_prediction(self, mock_load):
        fake_model = FakeModel([0.85, 0.05, 0.07, 0.03])
        mock_load.return_value = (
            fake_model,
            self.class_names,
            self.config
        )

        result = predict_image(self.image)

        self.assertEqual(result["crop"], "Tomato")
        self.assertEqual(result["condition"], "Early blight")
        self.assertEqual(result["status"], "Disease detected")
        self.assertFalse(result["is_healthy"])
        self.assertAlmostEqual(result["confidence"], 0.85, places=5)

    @patch("src.prediction.load_artifacts")
    def test_healthy_prediction(self, mock_load):
        fake_model = FakeModel([0.02, 0.90, 0.05, 0.03])
        mock_load.return_value = (
            fake_model,
            self.class_names,
            self.config
        )

        result = predict_image(self.image)

        self.assertEqual(result["crop"], "Tomato")
        self.assertEqual(result["condition"], "healthy")
        self.assertEqual(result["status"], "Healthy")
        self.assertTrue(result["is_healthy"])
        self.assertAlmostEqual(result["confidence"], 0.90, places=5)

    @patch("src.prediction.load_artifacts")
    def test_top_three_predictions(self, mock_load):
        fake_model = FakeModel([0.85, 0.05, 0.07, 0.03])
        mock_load.return_value = (
            fake_model,
            self.class_names,
            self.config
        )

        result = predict_image(self.image)

        self.assertEqual(len(result["top_predictions"]), 3)
        self.assertEqual(
            result["top_predictions"][0]["condition"],
            "Early blight"
        )

    def test_format_class_name(self):
        crop, condition = format_class_name(
            "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot"
        )

        self.assertEqual(crop, "Corn (maize)")
        self.assertEqual(
            condition,
            "Cercospora leaf spot Gray leaf spot"
        )

    def test_invalid_class_name(self):
        with self.assertRaises(ValueError):
            format_class_name("InvalidClassName")


if __name__ == "__main__":
    unittest.main()
