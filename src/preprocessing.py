import numpy as np
import tensorflow as tf

from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Prepare an uploaded plant image for MobileNetV2 prediction.

    Steps:
    1. Convert image to RGB.
    2. Resize to 224 x 224.
    3. Convert image to a NumPy array.
    4. Apply MobileNetV2 preprocessing.
    5. Add a batch dimension.

    Returns:
        np.ndarray: Preprocessed image with shape
                    (1, 224, 224, 3).
    """

    if not isinstance(image, Image.Image):
        raise TypeError("Input must be a PIL Image.")

    # Load and standardize image color mode
    image = image.convert("RGB")

    # Resize image to the model's expected dimensions
    image = image.resize((224, 224), Image.Resampling.BILINEAR)

    # Convert image to NumPy array
    image_array = np.asarray(image, dtype=np.float32)

    # Apply MobileNetV2 preprocessing: pixel values become [-1, 1]
    image_array = preprocess_input(image_array)

    # Add batch dimension for model prediction
    image_array = np.expand_dims(image_array, axis=0)

    return image_array
