
import numpy as np
from PIL import Image

from src.model_loader import load_artifacts
from src.preprocessing import preprocess_image


def format_class_name(class_name: str) -> tuple[str, str]:
    """
    Convert a PlantVillage class label into a readable crop and condition.

    Example:
    Tomato___Early_blight -> ("Tomato", "Early blight")
    Tomato___healthy -> ("Tomato", "Healthy")
    """
    if "___" not in class_name:
        raise ValueError(f"Invalid class name format: {class_name}")

    crop, condition = class_name.split("___", maxsplit=1)

    crop = " ".join(crop.replace("_", " ").split())
    condition = " ".join(condition.replace("_", " ").split())

    return crop, condition


def predict_image(image: Image.Image) -> dict:
    """
    Predict the health condition of an uploaded plant leaf image.

    Returns the predicted crop, condition, status, confidence,
    and the top three predictions.
    """

    # Load the trained model and its supporting artifacts
    model, class_names, config = load_artifacts()

    # Prepare the image for MobileNetV2
    processed_image = preprocess_image(image)

    # Generate class probabilities
    predictions = model.predict(processed_image, verbose=0)

    probabilities = np.asarray(predictions)[0]

    # Validate the model output
    if probabilities.ndim != 1:
        raise ValueError("Unexpected prediction output shape.")

    if len(probabilities) != len(class_names):
        raise ValueError("Prediction output does not match the class names.")

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Model returned invalid prediction values.")

    # Find the class with the highest probability
    predicted_index = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])

    # Extract crop and condition
    crop, condition = format_class_name(predicted_class)

    # Determine whether the leaf is healthy or diseased
    is_healthy = condition.casefold() == "healthy"

    status = "Healthy" if is_healthy else "Disease detected"

    # Get the three most likely predictions
    top_indices = np.argsort(probabilities)[::-1][:3]

    top_predictions = []

    for index in top_indices:
        class_name = class_names[int(index)]
        top_crop, top_condition = format_class_name(class_name)

        top_is_healthy = top_condition.casefold() == "healthy"

        top_predictions.append({
            "class_name": class_name,
            "crop": top_crop,
            "condition": top_condition,
            "status": "Healthy" if top_is_healthy else "Disease detected",
            "confidence": float(probabilities[index])
        })

    return {
        "class_name": predicted_class,
        "crop": crop,
        "condition": condition,
        "status": status,
        "is_healthy": is_healthy,
        "confidence": confidence,
        "top_predictions": top_predictions
    }
