import streamlit as st
from PIL import Image

from src.api_client import (
    APIClientError,
    predict_image_api,
)
from src.disease_guidance import get_disease_guidance


def render_prediction_page():
    """Render the LeafLens plant disease detection interface."""

    st.title("🌿 Plant Disease Detection")

    st.write(
        "Upload a plant leaf image to identify its crop, detect possible "
        "diseases, and receive practical plant-health guidance."
    )

    st.info(
        "LeafLens automatically identifies the crop and predicts its "
        "condition. No more guessing what's wrong with your plants."
    )

    uploaded_file = st.file_uploader(
        "Upload a plant leaf image",
        type=["jpg", "jpeg", "png"],
        key="prediction_upload",
    )

    if uploaded_file is None:
        st.info("Upload a leaf image to begin disease detection.")
        return

    try:
        image_bytes = uploaded_file.getvalue()

        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Uploaded Image")

        st.image(
            image,
            caption=uploaded_file.name,
            width=350,
        )

    except Exception as error:
        st.error(f"Could not read the uploaded image: {error}")
        return

    if st.button(
        "🔍 Analyze Plant Image",
        type="primary",
        use_container_width=True,
    ):
        try:
            with st.spinner(
                "Analyzing your plant image with the LeafLens model..."
            ):
                response = predict_image_api(
                    image_bytes=image_bytes,
                    filename=uploaded_file.name,
                )

            prediction = response["prediction"]

            crop = prediction["crop"]
            condition = prediction["condition"]
            confidence = float(prediction["confidence"])
            is_healthy = bool(prediction["is_healthy"])

            # Retrieve guidance for the exact predicted crop and condition.
            guidance = get_disease_guidance(
                crop,
                condition,
            )

            st.divider()
            st.subheader("🌱 Disease Detection Results")

            # Main prediction result.
            result_col1, result_col2 = st.columns([2, 1])

            with result_col1:
                st.markdown("**Detected Crop**")
                st.markdown(f"### {crop}")

                st.markdown("**Detected Condition**")
                st.markdown(
                    f"### {condition.replace('_', ' ')}"
                )

                if is_healthy:
                    st.success("🌿 Healthy plant detected")
                else:
                    st.warning("⚠️ Disease detected")

            with result_col2:
                st.metric(
                    "Model Confidence",
                    f"{confidence * 100:.2f}%",
                )

            st.progress(
                min(max(confidence, 0.0), 1.0),
                text=f"Prediction confidence: {confidence * 100:.2f}%",
            )

            st.divider()

            # Disease-specific explanation.
            st.subheader("What This Means")

            st.write(guidance["explanation"])

            # Priority indicator.
            st.subheader("Recommended Priority")

            priority = guidance["priority"]

            if is_healthy:
                st.success(f"🌿 {priority}")
            elif "promptly" in priority.lower() or "specialist" in priority.lower():
                st.error(f"{priority}")
            else:
                st.warning(f"{priority}")

            # Practical recommendations.
            st.subheader("Recommended Actions")

            for index, action in enumerate(
                guidance["actions"],
                start=1,
            ):
                st.markdown(f"**{index}.** {action}")

            st.divider()

            st.caption(
                "LeafLens predictions are educational decision-support "
                "information, not confirmed diagnoses. Visual symptoms "
                "can overlap between diseases. Consult a local agricultural "
                "extension officer or plant-health specialist when needed."
            )

        except APIClientError as error:
            st.error(f"Prediction service error: {error}")

        except (KeyError, TypeError, ValueError) as error:
            st.error(
                f"Unexpected prediction response: {error}"
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")


if __name__ == "__main__":
    render_prediction_page()
