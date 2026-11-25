
import unittest

from src.disease_guidance import get_disease_guidance
from src.model_loader import load_class_names
from src.prediction import format_class_name


class TestDiseaseGuidance(unittest.TestCase):

    def test_healthy_guidance(self):
        result = get_disease_guidance("Tomato", "healthy")

        self.assertTrue(result["is_healthy"])
        self.assertEqual(result["priority"], "Routine monitoring")
        self.assertGreater(len(result["actions"]), 0)
        self.assertIn("explanation", result)

    def test_known_disease_guidance(self):
        result = get_disease_guidance("Tomato", "Early blight")

        self.assertFalse(result["is_healthy"])
        self.assertIn("fungal disease", result["explanation"])
        self.assertGreater(len(result["actions"]), 0)

    def test_huanglongbing_guidance(self):
        result = get_disease_guidance(
            "Orange",
            "Haunglongbing (Citrus greening)"
        )

        self.assertFalse(result["is_healthy"])
        self.assertEqual(
            result["priority"],
            "Seek specialist advice promptly"
        )

    def test_unknown_condition_fallback(self):
        result = get_disease_guidance(
            "Apple",
            "Unknown leaf condition"
        )

        self.assertFalse(result["is_healthy"])
        self.assertIn(
            "Specific guidance",
            result["explanation"]
        )
        self.assertGreater(len(result["actions"]), 0)

    def test_all_38_model_classes_have_guidance(self):
        class_names = load_class_names()

        self.assertEqual(len(class_names), 38)

        for class_name in class_names:
            crop, condition = format_class_name(class_name)

            result = get_disease_guidance(crop, condition)

            with self.subTest(class_name=class_name):
                self.assertIn("explanation", result)
                self.assertIn("actions", result)
                self.assertIn("priority", result)
                self.assertIn("is_healthy", result)
                self.assertGreater(len(result["actions"]), 0)


if __name__ == "__main__":
    unittest.main()

