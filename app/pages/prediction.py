
import streamlit as st
from PIL import Image

from src.model_loader import load_class_names
from src.prediction import format_class_name, predict_image
from src.disease_guidance import get_disease_guidance


def get_confidence_message(confidence):
    """Return farmer-friendly wording based on the model confidence score."""

    if confidence >= 0.80:
        return (
            "Higher confidence",
            "The model has a relatively strong prediction. "
            "Still inspect the plant and verify visible symptoms."
        )

    elif confidence >= 0.60:
        return (
            "Moderate confidence",
            "The model has a possible prediction, but it is less certain. "
            "Compare the result with the visible symptoms and consider "
            "taking another clear image."
        )

    return (
        "Low confidence",
        "The model is uncertain about this image. Try uploading a clearer "
        "photo showing a single leaf, and seek agricultural advice "
        "before making treatment decisions."
    )


def render_prediction_page():

    # Page header
    st.markdown(
        '<p class="main-title">Disease Detection 🔍</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="main-subtitle">'
        'Upload a plant leaf image to identify its health condition.'
        '</p>',
        unsafe_allow_html=True
    )

    st.write("")

    # Load available crop types
    try:
        class_names = load_class_names()

        crop_types = sorted({
            format_class_name(name)[0]
            for name in class_names
        })

    except Exception as error:
        st.error(f"Could not load the available crop types: {error}")
        return

    # Image input section
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("### 1. Select your crop")

        selected_crop = st.selectbox(
            "Crop type",
            options=crop_types,
            help="Choose the crop shown in your leaf image."
        )

        st.markdown("### 2. Upload a leaf image")

        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of a single plant leaf."
        )

        st.caption("Supported formats: JPG, JPEG and PNG")

    # Image preview
    image = None

    with right:
        st.markdown("### 3. Image preview")

        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert("RGB")

                st.image(
                    image,
                    caption="Uploaded leaf image",
                    use_container_width=True
                )

            except Exception:
                st.error("Unable to read this image. Please upload another.")
                image = None

        else:
            st.info("Your uploaded leaf image will appear here.")

    st.write("")
    st.divider()

    # Prediction
    st.markdown("### 4. Analyze your plant")

    if st.button(
        "🔍 Analyze Image",
        type="primary",
        use_container_width=True,
        disabled=image is None
    ):
        try:
            with st.spinner("Analyzing your leaf image..."):
                result = predict_image(image)

            confidence = result["confidence"]
            crop_matches = (
                result["crop"].casefold() == selected_crop.casefold()
            )

            st.write("")
            st.markdown("## Prediction Result")

            # Confidence-aware prediction message
            confidence_label, confidence_message = (
                get_confidence_message(confidence)
            )

            if not crop_matches:
                st.warning(
                    f"You selected **{selected_crop}**, but the model "
                    f"identified the image as **{result['crop']}**. "
                    "Please check the selected crop and uploaded image."
                )

            elif confidence < 0.60:
                st.warning(
                    "The model could not make a confident prediction. "
                    "Please upload another clear image before relying "
                    "on this result."
                )

            elif result["is_healthy"]:
                st.success(
                    f"Your {result['crop']} leaf appears consistent "
                    "with the healthy class."
                )

            else:
                st.warning(
                    f"Possible {result['condition']} detected in your "
                    f"{result['crop']} leaf. Please verify the symptoms."
                )

            # Main prediction details
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Predicted Crop", result["crop"])

            with col2:
                st.metric("Condition", result["condition"])

            with col3:
                st.metric("Confidence", f"{confidence * 100:.2f}%")

            # Confidence indicator
            st.markdown("#### Prediction confidence")
            st.progress(max(0.0, min(1.0, confidence)))

            st.markdown(f"**{confidence_label}**")
            st.caption(confidence_message)

            # Farmer-friendly guidance
            guidance = get_disease_guidance(
                result["crop"],
                result["condition"]
            )

            st.write("")
            st.divider()

            st.markdown("## What this means")
            st.write(guidance["explanation"])

            st.markdown("## Recommended next steps")

            st.info(f"Priority: {guidance['priority']}")

            for action in guidance["actions"]:
                st.markdown(f"- {action}")

            st.write("")

            st.caption(
                "Prediction confidence is the model's score for its "
                "predicted class. It does not measure disease severity "
                "or guarantee a confirmed diagnosis."
            )

            st.warning(
                "LeafLens provides preliminary AI-based guidance, "
                "not a confirmed diagnosis. Consult a qualified "
                "agricultural specialist for confirmation and "
                "treatment decisions."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")
