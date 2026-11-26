import os
import mimetypes
from typing import Any

import requests


API_BASE_URL = os.getenv(
    "LEAFLENS_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

DEFAULT_TIMEOUT = 120


class APIClientError(Exception):
    """Raised when communication with the LeafLens API fails."""


def _request(
    method: str,
    endpoint: str,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    **kwargs: Any,
) -> dict:
    """Send an HTTP request and return the JSON response."""

    url = f"{API_BASE_URL}{endpoint}"

    try:
        response = requests.request(
            method,
            url,
            timeout=timeout,
            **kwargs,
        )
        response.raise_for_status()

    except requests.exceptions.Timeout as exc:
        raise APIClientError(
            "The API request timed out. Please try again."
        ) from exc

    except requests.exceptions.ConnectionError as exc:
        raise APIClientError(
            f"Could not connect to the LeafLens API at {API_BASE_URL}. "
            "Make sure the API server is running."
        ) from exc

    except requests.exceptions.HTTPError as exc:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text

        raise APIClientError(
            f"API returned HTTP {response.status_code}: {detail}"
        ) from exc

    try:
        return response.json()

    except ValueError as exc:
        raise APIClientError(
            "The API returned an invalid JSON response."
        ) from exc


def check_api_health() -> dict:
    """Check whether the API and production model are available."""

    return _request(
        "GET",
        "/health",
        timeout=10,
    )


def predict_image_api(
    image_bytes: bytes,
    filename: str,
) -> dict:
    """Send one image to the API for prediction."""

    return _request(
        "POST",
        "/predict",
        files={
            "file": (
                filename,
                image_bytes,
                mimetypes.guess_type(filename)[0] or "image/jpeg",
            )
        },
    )


def upload_images_api(
    files: list[tuple[str, bytes]],
    crop: str,
    disease: str,
) -> dict:
    """Upload multiple labelled images to the API."""

    multipart_files = [
        (
            "files",
            (
                filename,
                image_bytes,
                mimetypes.guess_type(filename)[0] or "image/jpeg",
            ),
        )
        for filename, image_bytes in files
    ]

    return _request(
        "POST",
        "/uploads",
        files=multipart_files,
        data={
            "crop": crop,
            "disease": disease,
        },
    )


def start_retraining_api(epochs: int = 20) -> dict:
    """Ask the API to start a background retraining job."""

    if not 1 <= epochs <= 50:
        raise ValueError("Epochs must be between 1 and 50.")

    return _request(
        "POST",
        "/retrain",
        data={
            "epochs": epochs,
        },
        timeout=30,
    )


def get_retraining_status_api(job_id: str) -> dict:
    """Retrieve the status of a retraining job."""

    return _request(
        "GET",
        f"/retrain/status/{job_id}",
        timeout=30,
    )
