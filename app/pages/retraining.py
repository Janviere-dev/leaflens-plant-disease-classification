import io
import streamlit as st
from PIL import Image

from src.model_loader import load_class_names

from src.database import (
    initialize_database,
    get_pending_images,
    get_approved_images,
    update_image_status,
)

from src.api_client import (
    APIClientError,
    upload_images_api,
    start_retraining_api,
    get_retraining_status_api,
)


# Metrics from the existing production model (Experiment 3).
PRODUCTION_METRICS = {
    "accuracy": 0.9584,
    "loss": 0.1664,
    "precision": 0.9466,
    "recall": 0.9455,
    "f1_score": 0.9452,
    "weighted_f1_score": 0.9571,
}


# CLASS INFORMATION

def get_class_details(class_names):
    """Convert model class names into crop and disease information."""

    class_details = []

    for index, class_name in enumerate(class_names):

        if "___" not in class_name:
            raise ValueError(
                f"Invalid class name in class_names.json: {class_name}"
            )

        raw_crop, raw_disease = class_name.split("___", 1)

        class_details.append({
            "index": index,
            "class_name": class_name,
            "raw_crop": raw_crop,
            "raw_disease": raw_disease,
            "crop": raw_crop.replace("_", " "),
            "disease": raw_disease.replace("_", " "),
        })

    return class_details


# IMAGE UPLOAD SECTION
def render_upload_section(class_details):
    """Allow users to upload multiple labelled plant images through the API."""

    st.subheader("Upload Training Images")

    st.write(
        "Submit new plant images with their correct crop and disease labels. "
        "Every submission is saved for review before it can be used "
        "for model retraining."
    )

    crops = sorted({
        item["crop"] for item in class_details
    })

    selected_crop = st.selectbox(
        "Select crop",
        crops,
        key="retraining_crop",
    )

    crop_classes = [
        item for item in class_details
        if item["crop"] == selected_crop
    ]

    selected_disease = st.selectbox(
        "Select disease or healthy condition",
        [item["disease"] for item in crop_classes],
        key="retraining_disease",
    )

    selected_class = next(
        item for item in crop_classes
        if item["disease"] == selected_disease
    )

    # Bulk image selection.
    uploaded_files = st.file_uploader(
        "Upload plant images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="retraining_images",
    )

    valid_files = []

    if uploaded_files:

        st.caption(
            f"Selected label: {selected_class['class_name']}"
        )

        st.write(
            f"Images selected: **{len(uploaded_files)}**"
        )

        st.markdown("**Image Preview**")

        preview_columns = st.columns(2)

        for index, uploaded_file in enumerate(uploaded_files):

            image_bytes = uploaded_file.getvalue()

            try:
                with Image.open(io.BytesIO(image_bytes)) as image:
                    image.verify()

                with Image.open(io.BytesIO(image_bytes)) as image:
                    preview_image = image.convert("RGB")

                valid_files.append((
                    uploaded_file.name,
                    image_bytes,
                ))

                with preview_columns[index % 2]:
                    st.image(
                        preview_image,
                        caption=uploaded_file.name,
                        use_container_width=True,
                    )

            except Exception:
                with preview_columns[index % 2]:
                    st.error(
                        f"{uploaded_file.name} is not a valid image."
                    )

        st.caption(
            f"{len(valid_files)} valid image(s) ready for submission."
        )

    st.info(
        "All images in one batch must belong to the same crop and disease "
        "class. You can submit another batch for a different class."
    )

    if st.button(
        "Submit Images for Review",
        type="primary",
        disabled=not valid_files,
        key="submit_training_images",
    ):

        try:
            with st.spinner(
                f"Uploading {len(valid_files)} image(s) to LeafLens..."
            ):

                response = upload_images_api(
                    files=valid_files,
                    crop=selected_class["raw_crop"],
                    disease=selected_class["raw_disease"],
                )

            total_files = response.get("total_files", len(valid_files))
            successful = response.get("successful_uploads", 0)
            failed = response.get("failed_uploads", 0)

            st.divider()
            st.subheader("Upload Summary")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Images", total_files)

            with col2:
                st.metric("Successfully Uploaded", successful)

            with col3:
                st.metric("Failed Uploads", failed)

            st.divider()

            st.markdown("### Submission Details")

            st.write(
                f"**Crop:** {selected_class['crop']}  \n"
                f"**Disease:** {selected_class['disease']}"
            )

            for item in response.get("results", []):

                filename = item.get(
                    "filename",
                    "Unknown image",
                )

                status = item.get("status", "unknown")

                with st.container(border=True):

                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.write(f"**{filename}**")

                        if item.get("image_id") is not None:
                            st.caption(
                                f"Reference ID: {item['image_id']}"
                            )

                        # The API returns "reason", not "error".
                        reason = item.get("reason")

                        if reason:
                            st.caption(f"Reason: {reason}")

                    with col2:

                        if status == "uploaded":
                            st.success("Uploaded")

                        else:
                            st.error(
                                status.replace("_", " ").title()
                            )

            if successful > 0:

                st.success(
                    f"{successful} image(s) successfully submitted."
                )

                st.info(
                    "Your images are now waiting for review. "
                    "Visit the Review & Approve tab to inspect and "
                    "approve them before retraining."
                )

            if failed > 0:
                st.warning(
                    f"{failed} image(s) could not be uploaded. "
                    "Check their failure reasons above."
                )

        except APIClientError as error:
            st.error(f"Image upload failed: {error}")

        except Exception as error:
            st.error(f"Unexpected upload error: {error}")


# IMAGE REVIEW SECTION

def render_review_section():
    """Allow users to review and approve uploaded images."""

    st.subheader("Review Uploaded Images")

    pending_images = get_pending_images()
    approved_images = get_approved_images()

    pending_count = len(pending_images)
    approved_count = len(approved_images)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Pending Review", pending_count)

    with col2:
        st.metric("Approved Images", approved_count)

    st.divider()

    if not pending_images:
        st.success("There are no images waiting for review.")

    for record in pending_images:

        with st.container(border=True):

            col1, col2 = st.columns([1, 2])

            with col1:
                st.image(
                    Image.open(io.BytesIO(record["image"])),
                    use_container_width=True,
                )

            with col2:

                st.markdown(
                    f"**{record['original_filename']}**"
                )

                st.write(f"Crop: {record['crop']}")
                st.write(f"Label: {record['disease']}")

                st.caption(
                    f"Image ID: {record['id']} | "
                    f"Uploaded: {record['created_at']}"
                )

                review_note = st.text_input(
                    "Review note (optional)",
                    key=f"review_note_{record['id']}",
                )

                approve_col, reject_col = st.columns(2)

                with approve_col:

                    if st.button(
                        "Approve",
                        key=f"approve_{record['id']}",
                        type="primary",
                    ):

                        try:
                            update_image_status(
                                record["id"],
                                "approved",
                                review_note,
                            )

                            st.success("Image approved.")
                            st.rerun()

                        except Exception as error:
                            st.error(str(error))

                with reject_col:

                    if st.button(
                        "Reject",
                        key=f"reject_{record['id']}",
                    ):

                        try:
                            update_image_status(
                                record["id"],
                                "rejected",
                                review_note,
                            )

                            st.warning("Image rejected.")
                            st.rerun()

                        except Exception as error:
                            st.error(str(error))

    st.divider()

    st.markdown("**Recently Approved Images**")

    if approved_images:

        for record in approved_images[-5:][::-1]:

            st.write(
                f"ID {record['id']} — "
                f"{record['crop']} / {record['disease']}"
            )

    else:
        st.info("No approved images yet.")


# RETRAINING SECTION

def render_training_section():
    """Start and monitor retraining through the FastAPI service."""

    st.subheader("Retrain Model")

    approved_images = get_approved_images()
    approved_count = len(approved_images)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Approved Training Images", approved_count)

    with col2:
        st.metric("Training Classes", 38)

    st.write(
        "The retraining process combines the original training dataset "
        "with approved uploaded images. This helps the model learn from "
        "new examples while retaining its existing knowledge."
    )

    st.info(
        "Only approved images are used. Pending and rejected images "
        "are excluded from training."
    )

    # The API accepts epochs between 1 and 50.
    epochs = st.number_input(
        "Maximum training epochs",
        min_value=1,
        max_value=50,
        value=5,
        step=1,
        help="Early stopping can end training before this limit.",
        key="retraining_epochs",
    )

    current_job_id = st.session_state.get(
        "retraining_job_id"
    )

    current_status = st.session_state.get(
        "retraining_job_status"
    )

    # Start a new retraining job.
    if st.button(
        "Start Model Retraining",
        type="primary",
        disabled=(
            approved_count == 0
            or current_status in ("queued", "running")
        ),
        key="start_retraining",
    ):

        try:
            with st.spinner(
                "Submitting retraining request to LeafLens..."
            ):

                response = start_retraining_api(
                    epochs=int(epochs)
                )

            st.session_state["retraining_job_id"] = response["job_id"]

            st.session_state["retraining_job_status"] = response.get(
                "status",
                "queued",
            )

            st.session_state["retraining_completed"] = False

            st.session_state.pop("retraining_metrics", None)
            st.session_state.pop("candidate_path", None)

            st.success("Retraining job successfully submitted.")

            st.rerun()

        except APIClientError as error:
            st.error(f"Could not start retraining: {error}")

        except (KeyError, TypeError) as error:
            st.error(
                f"The API returned an unexpected response: {error}"
            )

    # Monitor an existing job.
    current_job_id = st.session_state.get(
        "retraining_job_id"
    )

    if current_job_id:

        st.divider()
        st.markdown("**Retraining Progress**")

        current_status = st.session_state.get(
            "retraining_job_status",
            "unknown",
        )

        if current_status in ("queued", "running"):

            st.info(
                "Your candidate model is being trained in the background."
            )

        elif current_status == "completed":

            st.success(
                "Retraining and evaluation completed successfully!"
            )

        elif current_status == "failed":

            st.error(
                "The retraining job failed. Check the API server logs "
                "for more information."
            )

        else:
            st.write(
                f"Current status: **{current_status.title()}**"
            )

        if current_status in ("queued", "running"):

            st.caption(
                "Training is running in the API background process. "
                "Refresh the status to check the latest update."
            )

        if st.button(
            "Refresh Retraining Status",
            key="refresh_retraining_status",
        ):

            try:
                with st.spinner("Checking retraining status..."):

                    status_response = get_retraining_status_api(
                        current_job_id
                    )

                job_status = status_response.get(
                    "status",
                    "unknown",
                )

                st.session_state["retraining_job_status"] = job_status

                if job_status == "completed":

                    metrics = status_response.get("metrics")

                    if metrics:
                        st.session_state["retraining_metrics"] = metrics

                    candidate_path = status_response.get(
                        "candidate_model_path"
                    )

                    if candidate_path:
                        st.session_state["candidate_path"] = str(
                            candidate_path
                        )

                    st.session_state["retraining_completed"] = True

                elif job_status == "failed":

                    st.session_state["retraining_error"] = (
                        status_response.get(
                            "message",
                            "The retraining process failed.",
                        )
                    )

                st.session_state["retraining_job_message"] = (
                    status_response.get("message", "")
                )

                st.rerun()

            except APIClientError as error:
                st.error(
                    f"Could not retrieve retraining status: {error}"
                )

            except Exception as error:
                st.error(
                    f"Unexpected status response: {error}"
                )

        # Display the latest message returned by the API.
        job_message = st.session_state.get(
            "retraining_job_message"
        )

        if job_message:
            st.caption(job_message)

        if current_status == "failed":

            error_message = st.session_state.get(
                "retraining_error",
                "Please check the API logs.",
            )

            st.error(error_message)

            if st.button(
                "Clear Failed Job",
                key="clear_failed_job",
            ):

                st.session_state.pop("retraining_job_id", None)
                st.session_state.pop("retraining_job_status", None)
                st.session_state.pop("retraining_job_message", None)
                st.session_state.pop("retraining_error", None)

                st.rerun()

        elif current_status == "completed":

            st.success(
                "The candidate model is ready for evaluation."
            )

            st.info(
                "The production model has not been changed. "
                "The new model is saved separately as a candidate."
            )

            st.caption(
                "Open the Evaluation tab to compare both models."
            )


# MODEL EVALUATION SECTION
def render_evaluation_section():
    """Compare candidate metrics with the production model."""

    st.subheader("Model Evaluation")

    st.write(
        "Evaluate the candidate model against the existing production "
        "model using the same untouched test dataset."
    )

    candidate_metrics = st.session_state.get(
        "retraining_metrics"
    )

    if candidate_metrics is None:

        st.info(
            "No candidate evaluation is available yet. "
            "Approve at least one image, complete retraining, "
            "and refresh the job status."
        )

        return

    st.success("Candidate model evaluation is available.")

    st.markdown("**Candidate Model Performance**")

    metrics_columns = st.columns(3)

    displayed_metrics = [
        ("Accuracy", "accuracy", True),
        ("Loss", "loss", False),
        ("Macro Precision", "precision", True),
        ("Macro Recall", "recall", True),
        ("Macro F1-score", "f1_score", True),
        ("Weighted F1-score", "weighted_f1_score", True),
    ]

    for index, (label, key, higher_is_better) in enumerate(
        displayed_metrics
    ):

        candidate_value = candidate_metrics[key]
        production_value = PRODUCTION_METRICS[key]

        with metrics_columns[index % 3]:

            st.metric(
                label,
                f"{candidate_value:.4f}",
                delta=f"{candidate_value - production_value:+.4f}",
                delta_color=(
                    "normal" if higher_is_better else "inverse"
                ),
            )

    st.divider()

    st.markdown("**Production vs Candidate**")

    comparison_rows = [
        ("Accuracy", "accuracy"),
        ("Loss", "loss"),
        ("Macro Precision", "precision"),
        ("Macro Recall", "recall"),
        ("Macro F1-score", "f1_score"),
        ("Weighted F1-score", "weighted_f1_score"),
    ]

    header1, header2, header3 = st.columns([2, 1, 1])

    with header1:
        st.write("**Metric**")

    with header2:
        st.write("**Production**")

    with header3:
        st.write("**Candidate**")

    for label, key in comparison_rows:

        production_value = PRODUCTION_METRICS[key]
        candidate_value = candidate_metrics[key]

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.write(label)

        with col2:
            st.write(f"{production_value:.4f}")

        with col3:
            st.write(f"{candidate_value:.4f}")

    st.caption(
        "The left value is the existing production model. "
        "The right value is the newly retrained candidate."
    )

    st.divider()

    st.markdown("**Candidate Model Decision**")

    accuracy_improved = (
        candidate_metrics["accuracy"]
        >= PRODUCTION_METRICS["accuracy"]
    )

    f1_improved = (
        candidate_metrics["f1_score"]
        >= PRODUCTION_METRICS["f1_score"]
    )

    loss_improved = (
        candidate_metrics["loss"]
        <= PRODUCTION_METRICS["loss"]
    )

    if accuracy_improved and f1_improved and loss_improved:

        st.success(
            "The candidate matches or improves all three main "
            "comparison metrics. It is a promising promotion candidate."
        )

    else:

        st.warning(
            "The candidate does not improve all three main comparison "
            "metrics. Keep the existing production model while "
            "investigating its performance."
        )

    st.warning(
        "This is an evaluation recommendation, not an automatic "
        "deployment. The production model remains unchanged."
    )

    candidate_path = st.session_state.get("candidate_path")

    if candidate_path:
        st.write(f"Saved candidate artifact: `{candidate_path}`")


# MAIN RETRAINING PAGE
def render_retraining_page():
    """Render the complete LeafLens model retraining page."""

    st.title("Model Retraining")

    st.write(
        "Manage new training data, review uploaded images, "
        "retrain the plant disease model, and evaluate its performance."
    )

    try:
        initialize_database()

        class_names = load_class_names()
        class_details = get_class_details(class_names)

    except Exception as error:
        st.error(
            f"Could not initialize the retraining page: {error}"
        )
        return

    upload_tab, review_tab, training_tab, evaluation_tab = st.tabs([
        "Upload Images",
        "Review & Approve",
        "Retrain Model",
        "Evaluation",
    ])

    with upload_tab:
        render_upload_section(class_details)

    with review_tab:
        render_review_section()

    with training_tab:
        render_training_section()

    with evaluation_tab:
        render_evaluation_section()


if __name__ == "__main__":
    render_retraining_page()
