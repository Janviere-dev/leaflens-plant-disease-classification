import json
from functools import lru_cache
from pathlib import Path

import tensorflow as tf


# Project and model artifact locations
PROJECT_ROOT = Path(__file__).resolve().parents[1]

ARTIFACT_DIR = (
    PROJECT_ROOT
    / "models"
    / "leaflens-trained-models"
    / "leaflens-trained-models"
)

MODEL_PATH = ARTIFACT_DIR / "leaflens_final_model.keras"
CLASS_NAMES_PATH = ARTIFACT_DIR / "class_names.json"
CONFIG_PATH = ARTIFACT_DIR / "preprocessing_config.json"


@lru_cache(maxsize=1)
def load_model():
    """Load and cache the final trained model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    return model


def load_class_names():
    """Load the class names in their original model output order."""

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Class names file not found: {CLASS_NAMES_PATH}"
        )

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    if not isinstance(class_names, list):
        raise ValueError("Class names must be stored as a list.")

    if len(class_names) != 38:
        raise ValueError(
            f"Expected 38 classes, but found {len(class_names)}."
        )

    return class_names


def load_preprocessing_config():
    """Load the model preprocessing configuration."""

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessing configuration not found: {CONFIG_PATH}"
        )

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)

    if config.get("num_classes") != 38:
        raise ValueError("Configuration does not specify 38 classes.")

    return config


def load_artifacts():
    """Load and validate all artifacts required for prediction."""

    model = load_model()
    class_names = load_class_names()
    config = load_preprocessing_config()

    if model.output_shape[-1] != len(class_names):
        raise ValueError(
            "Model output count does not match the number of class names."
        )

    return model, class_names, config
