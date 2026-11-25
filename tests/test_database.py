
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import database


class TestDatabase(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_leaflens.db"

        self.db_patcher = patch.object(
            database, "DATABASE_PATH", self.db_path
        )
        self.dir_patcher = patch.object(
            database, "DATABASE_DIR", Path(self.temp_dir.name)
        )

        self.db_patcher.start()
        self.dir_patcher.start()

        database.initialize_database()

    def tearDown(self):
        self.db_patcher.stop()
        self.dir_patcher.stop()
        self.temp_dir.cleanup()

    def test_database_table_and_columns(self):
        with database.get_connection() as connection:
            cursor = connection.execute(
                "PRAGMA table_info(uploaded_images)"
            )
            columns = {row["name"] for row in cursor.fetchall()}

        expected = {
            "id", "leaf_id", "image", "image_hash",
            "image_path", "original_filename", "crop",
            "disease", "class_name", "label_index",
            "status", "created_at", "reviewed_at",
            "review_note"
        }

        self.assertTrue(expected.issubset(columns))

    def test_save_uploaded_image(self):
        record_id = database.save_uploaded_image(
            image_bytes=b"test image content",
            original_filename="tomato.jpg",
            crop="Tomato",
            disease="Early blight",
            class_name="Tomato___Early_blight",
            label_index=29,
            leaf_id="LEAF_001",
            image_path="uploads/tomato.jpg",
        )

        self.assertIsNotNone(record_id)

        with database.get_connection() as connection:
            record = connection.execute(
                "SELECT * FROM uploaded_images WHERE id = ?",
                (record_id,)
            ).fetchone()

        self.assertEqual(record["crop"], "Tomato")
        self.assertEqual(record["leaf_id"], "LEAF_001")
        self.assertEqual(record["status"], "pending")
        self.assertEqual(record["image"], b"test image content")

    def test_duplicate_image_rejected(self):
        arguments = {
            "image_bytes": b"same image",
            "original_filename": "tomato.jpg",
            "crop": "Tomato",
            "disease": "Early blight",
            "class_name": "Tomato___Early_blight",
            "label_index": 29,
        }

        database.save_uploaded_image(**arguments)

        with self.assertRaisesRegex(ValueError, "already been uploaded"):
            database.save_uploaded_image(**arguments)

    def test_empty_image_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            database.save_uploaded_image(
                image_bytes=b"",
                original_filename="empty.jpg",
                crop="Tomato",
                disease="Early blight",
                class_name="Tomato___Early_blight",
                label_index=29,
            )


    def test_pending_images_retrieval(self):
        database.save_uploaded_image(
            image_bytes=b"pending image",
            original_filename="pending.jpg",
            crop="Tomato",
            disease="Early blight",
            class_name="Tomato___Early_blight",
            label_index=29,
        )

        pending = database.get_pending_images()

        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["status"] if "status" in pending[0].keys() else "pending", "pending")
        self.assertEqual(pending[0]["crop"], "Tomato")

    def test_approve_image(self):
        image_id = database.save_uploaded_image(
            image_bytes=b"approved image",
            original_filename="approved.jpg",
            crop="Tomato",
            disease="Early blight",
            class_name="Tomato___Early_blight",
            label_index=29,
        )

        database.update_image_status(
            image_id,
            "approved",
            "Label verified."
        )

        approved = database.get_approved_images()

        self.assertEqual(len(approved), 1)
        self.assertEqual(approved[0]["id"], image_id)
        self.assertEqual(approved[0]["image"], b"approved image")
        self.assertEqual(approved[0]["review_note"], "Label verified.")

    def test_rejected_image_not_in_approved_data(self):
        image_id = database.save_uploaded_image(
            image_bytes=b"rejected image",
            original_filename="rejected.jpg",
            crop="Tomato",
            disease="Early blight",
            class_name="Tomato___Early_blight",
            label_index=29,
        )

        database.update_image_status(image_id, "rejected")

        self.assertEqual(database.get_approved_images(), [])
        self.assertEqual(database.get_pending_images(), [])

    def test_invalid_status_rejected(self):
        with self.assertRaises(ValueError):
            database.update_image_status(1, "unknown")

    def test_cannot_review_image_twice(self):
        image_id = database.save_uploaded_image(
            image_bytes=b"reviewed image",
            original_filename="reviewed.jpg",
            crop="Tomato",
            disease="Early blight",
            class_name="Tomato___Early_blight",
            label_index=29,
        )

        database.update_image_status(image_id, "approved")

        with self.assertRaises(ValueError):
            database.update_image_status(image_id, "rejected")


if __name__ == "__main__":
    unittest.main()
