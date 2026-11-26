from io import BytesIO
from PIL import Image
from src.database import get_approved_images
from src.model_loader import load_class_names
import numpy as np
from src.preprocessing import preprocess_image
from datasets import load_from_disk
from pathlib import Path
import tensorflow as tf
from src.model_loader import load_model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)


def load_approved_training_images():
    """
    Load approved images from SQLite and validate their labels.
    """

    records = get_approved_images()
    class_names = load_class_names()
    training_images = []

    for record in records:
        database_id = record["id"]
        label = record["label_index"]
        class_name = record["class_name"]

        # Validate label index.
        if not isinstance(label, int) or not 0 <= label < len(class_names):
            raise ValueError(
                f"Invalid label index for image ID {database_id}: {label}"
            )

        # Validate class name against the model's class ordering.
        expected_class = class_names[label]

        if class_name != expected_class:
            raise ValueError(
                f"Label mismatch for image ID {database_id}. "
                f"Expected {expected_class}, received {class_name}."
            )

        # Reconstruct image.
        try:
            image = Image.open(BytesIO(record["image"]))
            image = image.convert("RGB")

            training_images.append({
                "image": image,
                "label": label,
                "class_name": class_name,
                "leaf_id": record["leaf_id"],
                "database_id": database_id,
            })

        except (OSError, ValueError) as error:
            raise ValueError(
                f"Could not read uploaded image ID {database_id}: {error}"
            ) from error

    return training_images


def prepare_approved_training_data():
    """
    Preprocess approved images into NumPy arrays
    ready for TensorFlow training.

    Returns:
        images: float32 array of shape (N, 224, 224, 3)
        labels: int32 array of shape (N,)
    """

    approved_images = load_approved_training_images()

    if not approved_images:
        return (
            np.empty((0, 224, 224, 3), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    images = []
    labels = []

    for record in approved_images:
        processed_image = preprocess_image(record["image"])

        # Remove the individual image's batch dimension.
        images.append(processed_image[0])
        labels.append(record["label"])

    images = np.stack(images).astype(np.float32)
    labels = np.asarray(labels, dtype=np.int32)

    return images, labels



PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREPARED_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "cleaned_data"
    / "plantvillage_prepared"
)

CANDIDATE_MODEL_DIR = PROJECT_ROOT / "models" / "retraining"
CANDIDATE_MODEL_PATH = (
    CANDIDATE_MODEL_DIR / "leaflens_retrained_candidate.keras"
)

def load_original_dataset_splits():
    """
    Load the original PlantVillage training, validation,
    and test splits without modifying them.
    """
    if not PREPARED_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found: {PREPARED_DATASET_PATH}"
        )

    dataset = load_from_disk(str(PREPARED_DATASET_PATH))

    required_splits = {"train", "validation", "test"}

    if not required_splits.issubset(dataset.keys()):
        raise ValueError(
            "Dataset must contain train, validation, and test splits."
        )

    return {
        "train": dataset["train"],
        "validation": dataset["validation"],
        "test": dataset["test"],
    }

def load_retraining_data():
    """
    Load original dataset splits and approved uploaded images.

    Returns the original splits and the approved training images.
    """
    splits = load_original_dataset_splits()
    approved_images, approved_labels = prepare_approved_training_data()

    if len(approved_images) == 0:
        raise ValueError(
            "No approved images available for retraining."
        )

    if len(approved_images) != len(approved_labels):
        raise ValueError(
            "Approved image and label counts do not match."
        )

    return {
        "train": splits["train"],
        "validation": splits["validation"],
        "test": splits["test"],
        "approved_images": approved_images,
        "approved_labels": approved_labels,
    }

def create_training_dataset(
    original_train,
    approved_images,
    approved_labels,
    batch_size=32,
):
    """
    Combine original training images and approved uploads
    into one shuffled, augmented TensorFlow dataset.
    """

    if len(approved_images) == 0:
        raise ValueError("No approved images available.")

    if len(approved_images) != len(approved_labels):
        raise ValueError("Image and label counts do not match.")

    # Read original images from the Hugging Face dataset.
    def original_image_generator():
        for record in original_train:
            image = record["image"].convert("RGB")
            image = np.asarray(image, dtype=np.uint8)
            label = np.int32(record["label"])

            yield image, label

    original_dataset = tf.data.Dataset.from_generator(
        original_image_generator,
        output_signature=(
            tf.TensorSpec(shape=(256, 256, 3), dtype=tf.uint8),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )

    # Resize and normalize original images.
    def preprocess_original(image, label):
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, [224, 224])
        image = tf.keras.applications.mobilenet_v2.preprocess_input(
            image
        )
        return image, label

    original_dataset = original_dataset.map(
        preprocess_original,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Create a dataset from approved, already-preprocessed uploads.
    approved_dataset = tf.data.Dataset.from_tensor_slices(
        (approved_images, approved_labels)
    )

    # Combine both datasets.
    combined_dataset = original_dataset.concatenate(approved_dataset)

    # Apply the same augmentation strategy to both groups.
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(
            factor=0.08,
            fill_mode="reflect",
        ),
        tf.keras.layers.RandomZoom(
            height_factor=0.10,
            width_factor=0.10,
            fill_mode="reflect",
        ),
    ])

    def augment_image(image, label):
        image = augmentation(
            tf.expand_dims(image, axis=0),
            training=True,
        )
        image = tf.squeeze(image, axis=0)
        return image, label

    combined_dataset = combined_dataset.map(
        augment_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Shuffle, batch, and prefetch.
    total_images = len(original_train) + len(approved_images)

    combined_dataset = combined_dataset.shuffle(
        buffer_size=min(total_images, 10000),
        reshuffle_each_iteration=True,
    )
    combined_dataset = combined_dataset.batch(batch_size)

    expected_batches = (
        total_images + batch_size - 1
        ) // batch_size

    combined_dataset = combined_dataset.apply(
        tf.data.experimental.assert_cardinality(expected_batches)
        )

    combined_dataset = combined_dataset.prefetch(tf.data.AUTOTUNE)
    return combined_dataset

def create_retraining_model(learning_rate=1e-5):
    """
    Create an independent copy of the current LeafLens model
    for controlled fine-tuning.
    """

    # Load the current production model.
    original_model = load_model()

    # Clone its architecture.
    retraining_model = tf.keras.models.clone_model(
        original_model
    )

    # Copy all learned weights.
    retraining_model.set_weights(original_model.get_weights())

    # Configure the MobileNetV2 backbone.
    backbone = retraining_model.get_layer(
        "mobilenetv2_1.00_224"
    )

    backbone.trainable = True

    # Freeze early layers and keep BatchNormalization frozen.
    for layer in backbone.layers:
        layer.trainable = False

    for layer in backbone.layers[-30:]:
        if not isinstance(
            layer,
            tf.keras.layers.BatchNormalization,
        ):
            layer.trainable = True

    # Keep the classification head trainable.
    for layer in retraining_model.layers:
        if layer.name != "mobilenetv2_1.00_224":
            layer.trainable = True

    # Compile with a small learning rate.
    retraining_model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return retraining_model

def create_evaluation_dataset(dataset, batch_size=32):
    """
    Prepare validation or test data for model evaluation.

    No augmentation is applied.
    """

    def image_generator():
        for record in dataset:
            image = record["image"].convert("RGB")
            image = np.asarray(image, dtype=np.uint8)
            label = np.int32(record["label"])

            yield image, label

    evaluation_dataset = tf.data.Dataset.from_generator(
        image_generator,
        output_signature=(
            tf.TensorSpec(
                shape=(256, 256, 3),
                dtype=tf.uint8,
            ),
            tf.TensorSpec(
                shape=(),
                dtype=tf.int32,
            ),
        ),
    )

    def preprocess(image, label):
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, [224, 224])
        image = tf.keras.applications.mobilenet_v2.preprocess_input(
            image
        )

        return image, label

    evaluation_dataset = evaluation_dataset.map(
        preprocess,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    evaluation_dataset = evaluation_dataset.batch(batch_size)

    expected_batches = (
        len(dataset) + batch_size - 1
    ) // batch_size

    evaluation_dataset = evaluation_dataset.apply(
        tf.data.experimental.assert_cardinality(expected_batches)
    )

    evaluation_dataset = evaluation_dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return evaluation_dataset

def train_retraining_model(
    training_dataset,
    validation_dataset,
    epochs=20,
    learning_rate=1e-5,
    patience=5,
):
    """
    Fine-tune the existing model using approved new images
    alongside the original training dataset.

    Saves the best candidate model separately from the
    production model.
    """

    CANDIDATE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model = create_retraining_model(
        learning_rate=learning_rate
    )

    callbacks = [
        ModelCheckpoint(
            filepath=str(CANDIDATE_MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),

        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    history = model.fit(
        training_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=callbacks,
    )
    model.save(CANDIDATE_MODEL_PATH)

    return model, history, CANDIDATE_MODEL_PATH

def evaluate_retrained_model(
    model,
    test_dataset,
):
    """
    Evaluate a retrained candidate model on the untouched
    test dataset.

    Returns loss, accuracy, precision, recall, and F1-score.
    """

    test_loss, test_accuracy = model.evaluate(
        test_dataset,
        verbose=1,
    )

    true_labels = []
    predicted_labels = []

    for images, labels in test_dataset:
        probabilities = model.predict(
            images,
            verbose=0,
        )

        predictions = np.argmax(
            probabilities,
            axis=1,
        )

        true_labels.extend(labels.numpy())
        predicted_labels.extend(predictions)

    precision = precision_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )

    f1 = f1_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0,
    )

    return {
        "loss": float(test_loss),
        "accuracy": float(test_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "weighted_f1_score": float(weighted_f1),
    }