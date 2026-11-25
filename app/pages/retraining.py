
import io

import streamlit as st
from PIL import Image

from src.model_loader import load_class_names
from src.database import (
    initialize_database,
    save_uploaded_image,
    get_pending_images,
    get_approved_images,
    update_image_status,
)

from src.retraining import (
    load_retraining_data,
    create_training_dataset,
    create_evaluation_dataset,
    train_retraining_model,
    evaluate_retrained_model,
    CANDIDATE_MODEL_PATH,
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


def get_class_details(class_names):
    """Convert model class names into crop and disease information."""

    class_details = []

    for index, class_name in enumerate(class_names):

        if "___" not in class_name:
            raise ValueError(
                f"Invalid class name in class_names.json: {class_name}"
            )

        crop, disease = class_name.split("___", 1)

        class_details.append({
            "index": index,
            "class_name": class_name,
            "crop": crop.replace("_", " "),
            "disease": disease.replace("_", " "),
        })

    return class_details


def render_upload_section(class_details):
    """Allow users to upload labelled plant images."""

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

    uploaded_file = st.file_uploader(
        "Upload plant image",
        type=["jpg", "jpeg", "png"],
        key="retraining_image",
    )

    image_bytes = None

    if uploaded_file is not None:

        image_bytes = uploaded_file.getvalue()

        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                image.verify()

            with Image.open(io.BytesIO(image_bytes)) as image:
                st.image(
                    image.convert("RGB"),
                    caption=uploaded_file.name,
                    width=300,
                )

        except Exception:
            st.error(
                "This file is not a valid image. "
                "Please upload a valid JPG, JPEG, or PNG image."
            )
            return

        st.caption(
            f"Selected label: {selected_class['class_name']}"
        )

    if st.button(
        "Submit Image for Review",
        type="primary",
        disabled=uploaded_file is None,
        key="submit_training_image",
    ):

        try:
            image_id = save_uploaded_image(
                image_bytes=image_bytes,
                original_filename=uploaded_file.name,
                crop=selected_class["crop"],
                disease=selected_class["disease"],
                class_name=selected_class["class_name"],
                label_index=selected_class["index"],
            )

            st.success(
                f"Image successfully submitted! Reference ID: {image_id}"
            )

        except ValueError as error:
            st.warning(str(error))

        except Exception as error:
            st.error(f"Could not save image: {error}")


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

                st.write(
                    f"Crop: {record['crop']}"
                )

                st.write(
                    f"Label: {record['disease']}"
                )

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


def render_training_section():
    """Train a candidate model using approved new images."""

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

    epochs = st.number_input(
        "Maximum training epochs",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        help="Early stopping can end training before this limit.",
    )

    if st.button(
        "Start Model Retraining",
        type="primary",
        disabled=approved_count == 0,
        key="start_retraining",
    ):

        try:
            with st.spinner(
                "Loading original dataset and approved images..."
            ):

                data = load_retraining_data()

                training_dataset = create_training_dataset(
                    original_train=data["train"],
                    approved_images=data["approved_images"],
                    approved_labels=data["approved_labels"],
                )

                validation_dataset = create_evaluation_dataset(
                    data["validation"]
                )

                test_dataset = create_evaluation_dataset(
                    data["test"]
                )

            with st.spinner(
                "Training candidate model. This may take some time..."
            ):

                model, history, candidate_path = train_retraining_model(
                    training_dataset=training_dataset,
                    validation_dataset=validation_dataset,
                    epochs=int(epochs),
                )

            with st.spinner(
                "Evaluating candidate model on the untouched test dataset..."
            ):

                metrics = evaluate_retrained_model(
                    model,
                    test_dataset,
                )

            st.session_state["retraining_metrics"] = metrics
            st.session_state["candidate_path"] = str(candidate_path)
            st.session_state["retraining_completed"] = True

            st.success(
                "Retraining and evaluation completed successfully!"
            )

            st.info(
                "Your production model has not been changed. "
                "The new model is saved separately as a candidate."
            )

            st.write(f"Candidate model: `{candidate_path}`")

            st.caption(
                "Open the Evaluation section to compare the candidate "
                "against the current production model."
            )

        except Exception as error:
            st.error("The retraining process failed.")
            st.exception(error)


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
            "Approve at least one image and complete retraining first."
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

        difference = candidate_value - production_value

        if not higher_is_better:
            difference = -difference

        with metrics_columns[index % 3]:

            st.metric(
                label,
                f"{candidate_value:.4f}",
                delta=f"{candidate_value - production_value:+.4f}",
                delta_color="normal" if higher_is_better else "inverse",
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
        st.error(f"Could not initialize the retraining page: {error}")
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
