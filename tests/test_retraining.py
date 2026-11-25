import numpy as np
import io
import unittest
from unittest.mock import patch
from PIL import Image
from src.retraining import load_approved_training_images
from src.retraining import load_original_dataset_splits
from src.retraining import load_retraining_data
from src.retraining import create_training_dataset
from src.retraining import create_retraining_model
from src.retraining import create_evaluation_dataset
from src.retraining import train_retraining_model
import tensorflow as tf
from src.retraining import evaluate_retrained_model


class TestRetrainingData(unittest.TestCase):

    def setUp(self):
        buffer = io.BytesIO()

        test_image = Image.new(
            "RGB",
            (256, 256),
            color=(120, 180, 80)
        )

        test_image.save(buffer, format="JPEG")
        self.image_bytes = buffer.getvalue()

        self.class_names = [
            "Apple___Apple_scab",
            "Apple___Black_rot",
            "Tomato___Early_blight",
            "Tomato___healthy",
        ]

    @patch("src.retraining.load_class_names")
    @patch("src.retraining.get_approved_images")
    def test_load_approved_images(self, mock_get_images, mock_class_names):
        mock_class_names.return_value = self.class_names

        mock_get_images.return_value = [{
            "id": 1,
            "image": self.image_bytes,
            "label_index": 2,
            "class_name": "Tomato___Early_blight",
            "leaf_id": "LEAF_001",
        }]

        results = load_approved_training_images()

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["image"].mode, "RGB")
        self.assertEqual(results[0]["label"], 2)
        self.assertEqual(results[0]["class_name"], "Tomato___Early_blight")

    @patch("src.retraining.load_class_names")
    @patch("src.retraining.get_approved_images")
    def test_no_approved_images(self, mock_get_images, mock_class_names):
        mock_class_names.return_value = self.class_names
        mock_get_images.return_value = []

        results = load_approved_training_images()

        self.assertEqual(results, [])

    @patch("src.retraining.load_class_names")
    @patch("src.retraining.get_approved_images")
    def test_invalid_image_rejected(self, mock_get_images, mock_class_names):
        mock_class_names.return_value = self.class_names

        mock_get_images.return_value = [{
            "id": 2,
            "image": b"invalid image data",
            "label_index": 2,
            "class_name": "Tomato___Early_blight",
            "leaf_id": None,
        }]

        with self.assertRaisesRegex(ValueError, "Could not read uploaded image ID 2"):
            load_approved_training_images()

    @patch("src.retraining.load_class_names")
    @patch("src.retraining.get_approved_images")
    def test_invalid_label_index(self, mock_get_images, mock_class_names):
        mock_class_names.return_value = self.class_names

        mock_get_images.return_value = [{
            "id": 3,
            "image": self.image_bytes,
            "label_index": 50,
            "class_name": "Tomato___Early_blight",
            "leaf_id": None,
        }]

        with self.assertRaisesRegex(ValueError, "Invalid label index"):
            load_approved_training_images()

    @patch("src.retraining.load_class_names")
    @patch("src.retraining.get_approved_images")
    def test_class_name_mismatch(self, mock_get_images, mock_class_names):
        mock_class_names.return_value = self.class_names

        mock_get_images.return_value = [{
            "id": 4,
            "image": self.image_bytes,
            "label_index": 0,
            "class_name": "Tomato___Early_blight",
            "leaf_id": None,
        }]

        with self.assertRaisesRegex(ValueError, "Label mismatch"):
            load_approved_training_images()

    @patch("src.retraining.load_approved_training_images")
    def test_prepare_approved_training_data(self, mock_load):
        mock_load.return_value = [
            {
                "image": Image.new("RGB", (256, 256), color="green"),
                "label": 2,
                "class_name": "Tomato___Early_blight",
                "leaf_id": None,
                "database_id": 1,
            }
        ]

        from src.retraining import prepare_approved_training_data

        images, labels = prepare_approved_training_data()

        self.assertEqual(images.shape, (1, 224, 224, 3))
        self.assertEqual(labels.shape, (1,))
        self.assertEqual(images.dtype, np.float32)
        self.assertEqual(labels.dtype, np.int32)

        self.assertGreaterEqual(images.min(), -1.0)
        self.assertLessEqual(images.max(), 1.0)

        np.testing.assert_array_equal(labels, [2])


    @patch("src.retraining.load_approved_training_images")
    def test_prepare_empty_approved_training_data(self, mock_load):
        mock_load.return_value = []

        from src.retraining import prepare_approved_training_data

        images, labels = prepare_approved_training_data()

        self.assertEqual(images.shape, (0, 224, 224, 3))
        self.assertEqual(labels.shape, (0,))
        self.assertEqual(images.dtype, np.float32)
        self.assertEqual(labels.dtype, np.int32)

    def test_load_original_dataset_splits(self):
        splits = load_original_dataset_splits()

        self.assertEqual(
            set(splits.keys()),
            {"train", "validation", "test"}
        )

        self.assertEqual(len(splits["train"]), 34880)
        self.assertEqual(len(splits["validation"]), 8702)
        self.assertEqual(len(splits["test"]), 10464)

        self.assertEqual(
            splits["train"].features["label"],
            splits["validation"].features["label"]
        )

        self.assertEqual(
            splits["train"].features["label"],
            splits["test"].features["label"]
        )

    @patch("src.retraining.prepare_approved_training_data")
    @patch("src.retraining.load_original_dataset_splits")
    def test_load_retraining_data(self, mock_splits, mock_approved):
        mock_splits.return_value = {
            "train": "original_train",
            "validation": "original_validation",
            "test": "original_test",
        }

        mock_approved.return_value = (
            np.zeros((2, 224, 224, 3), dtype=np.float32),
            np.array([2, 3], dtype=np.int32),
        )

        data = load_retraining_data()

        self.assertEqual(data["train"], "original_train")
        self.assertEqual(data["validation"], "original_validation")
        self.assertEqual(data["test"], "original_test")
        self.assertEqual(len(data["approved_images"]), 2)
        self.assertEqual(len(data["approved_labels"]), 2)


    @patch("src.retraining.prepare_approved_training_data")
    @patch("src.retraining.load_original_dataset_splits")
    def test_load_retraining_data_without_approved_images(
        self, mock_splits, mock_approved
    ):
        mock_splits.return_value = {
            "train": "original_train",
            "validation": "original_validation",
            "test": "original_test",
        }

        mock_approved.return_value = (
            np.empty((0, 224, 224, 3), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

        with self.assertRaisesRegex(
            ValueError, "No approved images available"
        ):
            load_retraining_data()

    def test_create_training_dataset(self):
        original_train = [
            {
                "image": Image.new("RGB", (256, 256), color="green"),
                "label": 0,
            },
            {
                "image": Image.new("RGB", (256, 256), color="red"),
                "label": 1,
            },
        ]

        approved_images = np.zeros(
            (1, 224, 224, 3),
            dtype=np.float32,
        )

        approved_labels = np.array([2], dtype=np.int32)

        dataset = create_training_dataset(
            original_train=original_train,
            approved_images=approved_images,
            approved_labels=approved_labels,
            batch_size=2,
        )

        batches = list(dataset.as_numpy_iterator())

        # We supplied three images, so expect two batches.
        self.assertEqual(len(batches), 2)

        all_images = np.concatenate(
            [batch[0] for batch in batches],
            axis=0,
        )

        all_labels = np.concatenate(
            [batch[1] for batch in batches],
            axis=0,
        )

        self.assertEqual(all_images.shape, (3, 224, 224, 3))
        self.assertEqual(all_images.dtype, np.float32)

        self.assertEqual(all_labels.shape, (3,))
        self.assertEqual(all_labels.dtype, np.int32)

        np.testing.assert_array_equal(
            np.sort(all_labels),
            [0, 1, 2],
        )

        self.assertTrue(np.isfinite(all_images).all())

    def test_create_training_dataset_without_approved_images(self):
        original_train = []

        approved_images = np.empty(
            (0, 224, 224, 3),
            dtype=np.float32,
        )

        approved_labels = np.empty((0,), dtype=np.int32)

        with self.assertRaisesRegex(
            ValueError, "No approved images available"
        ):
            create_training_dataset(
                original_train,
                approved_images,
                approved_labels,
            )


    def test_create_training_dataset_with_mismatched_counts(self):
        original_train = []

        approved_images = np.zeros(
            (2, 224, 224, 3),
            dtype=np.float32,
        )

        approved_labels = np.array([2], dtype=np.int32)

        with self.assertRaisesRegex(
            ValueError, "Image and label counts do not match"
        ):
            create_training_dataset(
                original_train,
                approved_images,
                approved_labels,
            )
    def test_create_retraining_model_preserves_initial_weights(self):
        from src.model_loader import load_model

        champion = load_model()
        original_weights = [
            weight.copy()
            for weight in champion.get_weights()
        ]

        challenger = create_retraining_model()

        # Both models must have identical initial weights.
        for champion_weight, challenger_weight in zip(
            original_weights,
            challenger.get_weights(),
        ):
            np.testing.assert_array_equal(
                champion_weight,
                challenger_weight,
            )

        # They must be separate model objects.
        self.assertIsNot(champion, challenger)


    def test_create_retraining_model_freezes_batch_normalization(self):
        import tensorflow as tf

        challenger = create_retraining_model()
        backbone = challenger.get_layer(
            "mobilenetv2_1.00_224"
        )

        # No BatchNormalization layer should be trainable.
        for layer in backbone.layers:
            if isinstance(
                layer,
                tf.keras.layers.BatchNormalization,
            ):
                self.assertFalse(layer.trainable)

        # The final backbone layers should be fine-tunable.
        self.assertTrue(backbone.get_layer("block_16_project").trainable)

        # Classification head should remain trainable.
        self.assertTrue(
            challenger.get_layer("classification_head").trainable
        )

    def test_create_evaluation_dataset_without_augmentation(self):
        evaluation_data = [
            {
                "image": Image.new("RGB", (256, 256), color="green"),
                "label": 0,
            },
            {
                "image": Image.new("RGB", (256, 256), color="red"),
                "label": 1,
            },
        ]

        dataset = create_evaluation_dataset(
            evaluation_data,
            batch_size=2,
        )

        batches = list(dataset.as_numpy_iterator())

        self.assertEqual(len(batches), 1)

        images, labels = batches[0]

        self.assertEqual(images.shape, (2, 224, 224, 3))
        self.assertEqual(images.dtype, np.float32)
        self.assertEqual(labels.dtype, np.int32)

        self.assertTrue(np.isfinite(images).all())
        self.assertGreaterEqual(images.min(), -1.0)
        self.assertLessEqual(images.max(), 1.0)

        np.testing.assert_array_equal(labels, [0, 1])

    @patch("src.retraining.create_retraining_model")
    def test_train_retraining_model(self, mock_create_model):
        mock_model = unittest.mock.MagicMock()
        mock_history = unittest.mock.MagicMock()

        mock_model.fit.return_value = mock_history
        mock_create_model.return_value = mock_model

        training_dataset = object()
        validation_dataset = object()

        model, history, candidate_path = train_retraining_model(
            training_dataset,
            validation_dataset,
            epochs=3,
            learning_rate=1e-5,
            patience=2,
        )

        mock_create_model.assert_called_once_with(
            learning_rate=1e-5
        )

        mock_model.fit.assert_called_once()

        fit_kwargs = mock_model.fit.call_args.kwargs

        self.assertIs(fit_kwargs["validation_data"], validation_dataset)
        self.assertEqual(fit_kwargs["epochs"], 3)
        self.assertEqual(len(fit_kwargs["callbacks"]), 3)

        self.assertIs(model, mock_model)
        self.assertIs(history, mock_history)
        self.assertEqual(candidate_path.name, "leaflens_retrained_candidate.keras")

    @patch("src.retraining.create_retraining_model")
    def test_evaluate_retrained_model(self, mock_create_model):
        mock_model = unittest.mock.MagicMock()

        mock_model.evaluate.return_value = (0.2, 0.75)

        mock_model.predict.side_effect = [
            np.array([
                [0.9, 0.1],
                [0.2, 0.8],
            ]),
            np.array([
                [0.7, 0.3],
                [0.8, 0.2],
            ]),
        ]

        test_images = np.zeros(
            (4, 224, 224, 3),
            dtype=np.float32,
        )

        test_labels = np.array([0, 1, 1, 0])

        test_dataset = tf.data.Dataset.from_tensor_slices(
            (test_images, test_labels)
        ).batch(2)

        results = evaluate_retrained_model(
            mock_model,
            test_dataset,
        )

        self.assertAlmostEqual(results["loss"], 0.2)
        self.assertAlmostEqual(results["accuracy"], 0.75)
        self.assertAlmostEqual(results["precision"], 5 / 6)
        self.assertAlmostEqual(results["recall"], 0.75)
        self.assertAlmostEqual(results["f1_score"], (0.8 + 2/3) / 2)

        self.assertEqual(mock_model.evaluate.call_count, 1)
        self.assertEqual(mock_model.predict.call_count, 2)
        self.assertAlmostEqual(results["weighted_f1_score"],(0.8 + 2/3) / 2)

if __name__ == "__main__":
    unittest.main()
