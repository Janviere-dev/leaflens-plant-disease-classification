
import hashlib
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATABASE_DIR = PROJECT_ROOT / "data"
DATABASE_PATH = DATABASE_DIR / "leaflens.db"


def get_connection():
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row

    return connection


def initialize_database():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS uploaded_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                leaf_id TEXT,
                image BLOB NOT NULL,
                image_hash TEXT NOT NULL UNIQUE,
                image_path TEXT,
                original_filename TEXT NOT NULL,
                crop TEXT NOT NULL,
                disease TEXT NOT NULL,
                class_name TEXT NOT NULL,
                label_index INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'approved', 'rejected')),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                reviewed_at TEXT,
                review_note TEXT
            )
        """)


def save_uploaded_image(
    image_bytes: bytes,
    original_filename: str,
    crop: str,
    disease: str,
    class_name: str,
    label_index: int,
    leaf_id: str | None = None,
    image_path: str | None = None,
):
    if not image_bytes:
        raise ValueError("Image data cannot be empty.")

    image_hash = hashlib.sha256(image_bytes).hexdigest()

    try:
        with get_connection() as connection:
            cursor = connection.execute("""
                INSERT INTO uploaded_images (
                    leaf_id,
                    image,
                    image_hash,
                    image_path,
                    original_filename,
                    crop,
                    disease,
                    class_name,
                    label_index
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                leaf_id,
                image_bytes,
                image_hash,
                image_path,
                original_filename,
                crop,
                disease,
                class_name,
                label_index,
            ))

            return cursor.lastrowid

    except sqlite3.IntegrityError as error:
        if "image_hash" in str(error):
            raise ValueError(
                "This image has already been uploaded."
            ) from error
        raise

def get_pending_images():
    """Retrieve all uploaded images awaiting review."""

    with get_connection() as connection:
        cursor = connection.execute("""
            SELECT
                id,
                image,
                original_filename,
                crop,
                disease,
                class_name,
                label_index,
                created_at
            FROM uploaded_images
            WHERE status = 'pending'
            ORDER BY created_at DESC
        """)

        return cursor.fetchall()

def update_image_status(
    image_id: int,
    status: str,
    review_note: str | None = None,
):
    """Approve or reject an uploaded image."""

    if status not in ("approved", "rejected"):
        raise ValueError("Status must be 'approved' or 'rejected'.")

    with get_connection() as connection:
        cursor = connection.execute("""
            UPDATE uploaded_images
            SET
                status = ?,
                reviewed_at = CURRENT_TIMESTAMP,
                review_note = ?
            WHERE id = ? AND status = 'pending'
        """, (status, review_note, image_id))

        if cursor.rowcount == 0:
            raise ValueError(
                f"Pending image with ID {image_id} was not found."
            )

def get_approved_images():
    """Retrieve approved images for the retraining process."""

    with get_connection() as connection:
        cursor = connection.execute("""
            SELECT
                id,
                leaf_id,
                image,
                image_hash,
                image_path,
                original_filename,
                crop,
                disease,
                class_name,
                label_index,
                created_at,
                reviewed_at,
                review_note
            FROM uploaded_images
            WHERE status = 'approved'
            ORDER BY created_at ASC
        """)

        return cursor.fetchall()