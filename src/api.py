import io
import logging
import threading
import uuid
from typing import Annotated

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.openapi.utils import get_openapi
from PIL import Image, UnidentifiedImageError

from src.database import (
    initialize_database,
    save_uploaded_image,
)
from src.model_loader import load_artifacts
from src.prediction import predict_image
from src.retraining import (
    load_retraining_data,
    create_training_dataset,
    create_evaluation_dataset,
    train_retraining_model,
    evaluate_retrained_model,
)


# Logging and application configuration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LeafLens API",
    description=(
        "Plant disease prediction, image uploads, "
        "and model retraining API."
    ),
    version="1.0.0",
)

# Store retraining job information in memory.
retraining_jobs = {}
jobs_lock = threading.Lock()


# Application startup

@app.on_event("startup")
def startup_event():
    """Initialize the SQLite database when the API starts."""

    initialize_database()
    logger.info("LeafLens API started.")


# Root endpoint

@app.get("/")
def root():
    """Return basic API information."""

    return {
        "message": "Welcome to the LeafLens API",
        "documentation": "/docs",
        "health": "/health",
    }


# Health check endpoint

@app.get("/health")
def health_check():
    """Check API availability and model loading status."""

    try:
        model, class_names, config = load_artifacts()

        return {
            "status": "healthy",
            "api": "running",
            "model_loaded": True,
            "model_classes": len(class_names),
            "model_input_shape": str(model.input_shape),
        }

    except Exception:
        logger.exception("Health check failed.")

        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "api": "running",
                "model_loaded": False,
                "message": "Model artifacts could not be loaded.",
            },
        )


# Single-image prediction endpoint

@app.post("/predict")
async def predict(
    file: Annotated[
        UploadFile,
        File(description="Plant leaf image"),
    ],
):
    """Predict the health condition of one uploaded leaf image."""

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file.",
        )

    try:
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="The uploaded image is empty.",
            )

        image = Image.open(io.BytesIO(image_bytes))
        image.load()

        result = predict_image(image)

        return {
            "filename": file.filename,
            "prediction": result,
        }

    except HTTPException:
        raise

    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    except Exception:
        logger.exception("Prediction failed.")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Please try again.",
        )

    finally:
        await file.close()


# Bulk image upload endpoint

@app.post("/uploads")
async def upload_images(
    files: Annotated[
        list[UploadFile],
        File(description="One or more plant leaf images"),
    ],
    crop: Annotated[str, Form()],
    disease: Annotated[str, Form()],
):
    """
    Upload multiple labelled images for review.

    All images in one request share the supplied crop and disease label.
    Successfully uploaded images remain pending until reviewed.
    """

    crop = crop.strip()
    disease = disease.strip()

    if not crop or not disease:
        raise HTTPException(
            status_code=400,
            detail="Crop and disease labels are required.",
        )

    if not files:
        raise HTTPException(
            status_code=400,
            detail="Please upload at least one image.",
        )

    try:
        _, class_names, _ = load_artifacts()

    except Exception:
        logger.exception("Could not load class names.")

        raise HTTPException(
            status_code=503,
            detail="Model class information is unavailable.",
        )

    class_name = f"{crop}___{disease}"

    if class_name not in class_names:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown class: {class_name}. "
                "Use a crop and disease combination supported "
                "by the trained model."
            ),
        )

    label_index = class_names.index(class_name)

    results = []
    successful = 0
    failed = 0

    for file in files:
        try:
            if (
                not file.content_type
                or not file.content_type.startswith("image/")
            ):
                raise ValueError("File must be an image.")

            image_bytes = await file.read()

            if not image_bytes:
                raise ValueError("Image file is empty.")

            image = Image.open(io.BytesIO(image_bytes))
            image.verify()

            image_id = save_uploaded_image(
                image_bytes=image_bytes,
                original_filename=file.filename or "uploaded_image",
                crop=crop,
                disease=disease,
                class_name=class_name,
                label_index=label_index,
            )

            successful += 1

            results.append({
                "filename": file.filename,
                "status": "uploaded",
                "image_id": image_id,
                "review_status": "pending",
            })

        except ValueError as error:
            failed += 1

            results.append({
                "filename": file.filename,
                "status": "failed",
                "reason": str(error),
            })

        except (UnidentifiedImageError, OSError):
            failed += 1

            results.append({
                "filename": file.filename,
                "status": "failed",
                "reason": "Invalid image file.",
            })

        except Exception:
            failed += 1
            logger.exception(
                "Image upload failed: %s",
                file.filename,
            )

            results.append({
                "filename": file.filename,
                "status": "failed",
                "reason": "Unexpected upload error.",
            })

        finally:
            await file.close()

    return {
        "message": "Bulk upload processing completed.",
        "total_files": len(files),
        "successful_uploads": successful,
        "failed_uploads": failed,
        "crop": crop,
        "disease": disease,
        "results": results,
    }


# Background retraining function

def run_retraining_job(job_id: str, epochs: int):
    """Run retraining and save its status and evaluation results."""

    try:
        with jobs_lock:
            retraining_jobs[job_id]["status"] = "running"
            retraining_jobs[job_id]["message"] = (
                "Loading retraining data."
            )

        # Load original datasets and approved user uploads.
        data = load_retraining_data()

        # Combine original training data with approved uploads.
        training_dataset = create_training_dataset(
            data["train"],
            data["approved_images"],
            data["approved_labels"],
        )

        # Prepare validation and test datasets.
        validation_dataset = create_evaluation_dataset(
            data["validation"]
        )

        test_dataset = create_evaluation_dataset(
            data["test"]
        )

        with jobs_lock:
            retraining_jobs[job_id]["message"] = (
                f"Training candidate model for up to {epochs} epochs."
            )

        # Train a candidate model.
        # Learning rate and early-stopping patience use
        # the defaults defined in src/retraining.py.
        model, history, candidate_path = train_retraining_model(
            training_dataset,
            validation_dataset,
            epochs=epochs,
        )

        with jobs_lock:
            retraining_jobs[job_id]["message"] = (
                "Evaluating candidate model on the test dataset."
            )

        # Evaluate candidate model.
        metrics = evaluate_retrained_model(
            model,
            test_dataset,
        )

        # Save successful results.
        with jobs_lock:
            retraining_jobs[job_id].update({
                "status": "completed",
                "message": "Retraining and evaluation completed.",
                "metrics": metrics,
                "candidate_model_path": str(candidate_path),
                "approved_images_used": len(data["approved_images"]),
                "epochs_requested": epochs,
                "epochs_completed": len(
                    history.history.get("loss", [])
                ),
            })

        logger.info(
            "Retraining job %s completed successfully.",
            job_id,
        )

    except Exception as error:
        logger.exception(
            "Retraining job %s failed.",
            job_id,
        )

        with jobs_lock:
            retraining_jobs[job_id].update({
                "status": "failed",
                "message": str(error),
            })


# Trigger retraining endpoint

@app.post("/retrain", status_code=202)
def trigger_retraining(
    background_tasks: BackgroundTasks,
    epochs: Annotated[int, Form(ge=1, le=50)] = 20,
):
    """Start a background retraining job."""

    job_id = str(uuid.uuid4())

    with jobs_lock:

        # Prevent simultaneous retraining jobs.
        if any(
            job["status"] in ("queued", "running")
            for job in retraining_jobs.values()
        ):
            raise HTTPException(
                status_code=409,
                detail="A retraining job is already running.",
            )

        retraining_jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "message": "Retraining job queued.",
            "epochs_requested": epochs,
            "metrics": None,
        }

    # Execute retraining in the background.
    background_tasks.add_task(
        run_retraining_job,
        job_id,
        epochs,
    )

    return {
        "message": "Retraining job started.",
        "job_id": job_id,
        "status": "queued",
        "epochs": epochs,
        "status_endpoint": f"/retrain/status/{job_id}",
    }


# Retraining status endpoint

@app.get("/retrain/status/{job_id}")
def get_retraining_status(job_id: str):
    """Retrieve the status and results of a retraining job."""

    with jobs_lock:
        job = retraining_jobs.get(job_id)

        if job is None:
            raise HTTPException(
                status_code=404,
                detail="Retraining job not found.",
            )

        return dict(job)



def custom_openapi():
    """Customize the bulk-upload schema for Swagger file selection."""

    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    upload_schema = openapi_schema["components"]["schemas"].get(
        "Body_upload_images_uploads_post"
    )

    if upload_schema:
        files_schema = upload_schema["properties"]["files"]
        files_schema["items"]["format"] = "binary"
        files_schema["items"].pop("contentMediaType", None)

    app.openapi_schema = openapi_schema

    return app.openapi_schema


app.openapi = custom_openapi
