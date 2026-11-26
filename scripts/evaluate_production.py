import csv
import json
import time
from pathlib import Path
import numpy as np
from datasets import load_from_disk
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_loader import load_artifacts
from src.preprocessing import preprocess_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "plantvillage"
REPORT_DIR = PROJECT_ROOT / "reports"
BATCH_SIZE = 32


def main():
    print("Loading production model and artifacts...")
    model, class_names, config = load_artifacts()

    print("Loading held-out test dataset...")
    dataset = load_from_disk(str(DATASET_PATH))["test"]

    dataset_class_names = dataset.features["label"].names
    if class_names != dataset_class_names:
        raise ValueError(
            "Class order mismatch between model artifacts and test dataset.\n"
            f"Model classes: {class_names}\n"
            f"Dataset classes: {dataset_class_names}"
        )

    if len(class_names) != 38:
        raise ValueError(f"Expected 38 classes, got {len(class_names)}")

    total = len(dataset)
    print(f"Test images: {total}")
    print(f"Classes: {len(class_names)}")
    print(f"Batch size: {BATCH_SIZE}")
    print("Starting evaluation...")

    y_true = []
    y_pred = []
    preprocessing_seconds = 0.0
    inference_seconds = 0.0

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch_images = []
        batch_labels = []

        preprocessing_start = time.perf_counter()
        for index in range(start, end):
            sample = dataset[index]
            batch_images.append(preprocess_image(sample["image"])[0])
            batch_labels.append(sample["label"])
        preprocessing_seconds += time.perf_counter() - preprocessing_start

        batch_array = np.stack(batch_images).astype(np.float32)

        inference_start = time.perf_counter()
        probabilities = model.predict(batch_array, verbose=0)
        inference_seconds += time.perf_counter() - inference_start

        predictions = np.argmax(probabilities, axis=1)
        y_true.extend(batch_labels)
        y_pred.extend(predictions.tolist())

        processed = end
        print(f"Processed {processed}/{total} images", end="\r", flush=True)

    print("\nCalculating evaluation metrics...")

    accuracy = accuracy_score(y_true, y_pred)
    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
    )
    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true, y_pred, labels=list(range(len(class_names)))
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "model": config["model_filename"],
        "architecture": config["architecture"],
        "dataset": str(DATASET_PATH),
        "split": "test",
        "test_samples": total,
        "num_classes": len(class_names),
        "batch_size": BATCH_SIZE,
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "preprocessing_seconds": preprocessing_seconds,
        "model_inference_seconds": inference_seconds,
        "average_model_inference_ms_per_image": (
            inference_seconds / total * 1000
        ),
    }

    with open(REPORT_DIR / "production_evaluation.json", "w") as file:
        json.dump(summary, file, indent=4)

    with open(REPORT_DIR / "classification_report.json", "w") as file:
        json.dump(report, file, indent=4)

    with open(REPORT_DIR / "per_class_metrics.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["class", "precision", "recall", "f1_score", "support"]
        )
        for class_name in class_names:
            metrics = report[class_name]
            writer.writerow([
                class_name,
                metrics["precision"],
                metrics["recall"],
                metrics["f1-score"],
                metrics["support"],
            ])

    with open(REPORT_DIR / "confusion_matrix.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["actual/predicted"] + class_names)
        for class_name, row in zip(class_names, matrix):
            writer.writerow([class_name] + row.tolist())

    print("\n PRODUCTION MODEL EVALUATION ")
    print(f"Model:           {summary['model']}")
    print(f"Test samples:    {total}")
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Macro precision:{macro_precision:.4f}")
    print(f"Macro recall:    {macro_recall:.4f}")
    print(f"Macro F1:        {macro_f1:.4f}")
    print(f"Weighted F1:     {weighted_f1:.4f}")
    print(f"Preprocessing:   {preprocessing_seconds:.2f} seconds")
    print(f"Model inference: {inference_seconds:.2f} seconds")
    print(
        "Average inference per image: "
        f"{summary['average_model_inference_ms_per_image']:.2f} ms"
    )
    print(f"\nReports saved in: {REPORT_DIR}")


if __name__ == "__main__":
    main()
