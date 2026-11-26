import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st
from datasets import load_from_disk


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "cleaned_data" / "plantvillage_prepared"
METADATA_PATH = DATASET_PATH / "metadata.json"


@st.cache_data(show_spinner="Loading dataset insights...")
def load_insights_data():
    """Load the prepared dataset and calculate statistics without decoding images."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found at: {DATASET_PATH}"
        )

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    dataset = load_from_disk(str(DATASET_PATH))

    class_names = dataset["train"].features["label"].names

    counts_by_split = {}
    split_sizes = {}

    for split_name in ["train", "validation", "test"]:
        split = dataset[split_name]

        labels = split["label"]
        crops = split["crop"]
        diseases = split["disease"]

        class_counts = Counter(labels)
        crop_counts = Counter(crops)
        health_counts = Counter(
            "Healthy" if disease.casefold() == "healthy"
            else "Diseased"
            for disease in diseases
        )

        counts_by_split[split_name] = {
            "classes": {
                class_names[class_id]: class_counts.get(class_id, 0)
                for class_id in range(len(class_names))
            },
            "crops": dict(crop_counts),
            "health": {
                "Healthy": health_counts.get("Healthy", 0),
                "Diseased": health_counts.get("Diseased", 0),
            },
        }

        split_sizes[split_name] = len(split)

    return metadata, counts_by_split, split_sizes


def format_class_name(class_name):
    """Make model class labels easier to read in charts."""

    if "___" not in class_name:
        return class_name.replace("_", " ")

    crop, condition = class_name.split("___", maxsplit=1)

    crop = crop.replace("_", " ").replace(",", ", ")
    crop = " ".join(crop.split())

    condition = condition.replace("_", " ")
    condition = " ".join(condition.split())

    return f"{crop} - {condition}"


def render_metric(label, value):
    """Display a consistent overview metric."""

    st.metric(label=label, value=value)


def render_insights_page():

    # Page-specific styling
    st.markdown(
        """
        <style>
        /* Dataset split selector */
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            background-color: #ffffff;
            border: 1px solid #d5e5da;
            border-radius: 10px;
            color: #173d2b;
        }

        /* Improve information alert contrast */
        [data-testid="stAlert"] {
            background-color: #edf7f0;
            border: 1px solid #c9e2d1;
            border-radius: 12px;
        }

        [data-testid="stAlert"] p {
            color: #173d2b !important;
        }

        /* Consistent section spacing */
        [data-testid="stMain"] hr {
            border-color: #dce6df;
            margin-top: 1.8rem;
            margin-bottom: 1.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


    st.title("Dataset Insights")

    st.markdown(
        '<p class="main-subtitle">'
        'Explore the composition, class distribution and characteristics of the dataset'
        'used to train LeafLens.'
        '</p>',
        unsafe_allow_html=True
    )

    st.write("")

    # Load dataset statistics
    try:
        metadata, counts_by_split, split_sizes = load_insights_data()

    except Exception as error:
        st.error(f"Unable to load dataset insights: {error}")
        return

    total_images = sum(split_sizes.values())
    number_of_crops = len({
        crop
        for split in counts_by_split.values()
        for crop in split["crops"]
    })
    number_of_classes = metadata["number_of_classes"]

    image_dimensions = metadata["image_dimensions"]
    image_size = f"{image_dimensions[0]} × {image_dimensions[1]}"

    # Dataset overview
    st.markdown("### Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_metric("Total Images", f"{total_images:,}")

    with col2:
        render_metric("Crop Types", number_of_crops)

    with col3:
        render_metric("Crop-Condition Classes", number_of_classes)      
    with col4:
        render_metric("Image Dimensions", image_size)

    st.write("")
    st.divider()

    # Dataset split selector
    st.markdown("### Explore Dataset Distribution")

    selected_split = st.selectbox(
        "Select a dataset split to explore its image distribution.",
        options=["All splits", "train", "validation", "test"],
        format_func=lambda value: {
            "All splits": "All dataset splits",
            "train": "Training set",
            "validation": "Validation set",
            "test": "Testing set",
        }[value],
    )

    if selected_split == "All splits":
        selected_data = {
            "classes": Counter(),
            "crops": Counter(),
            "health": Counter(),
        }

        for split_data in counts_by_split.values():
            selected_data["classes"].update(split_data["classes"])
            selected_data["crops"].update(split_data["crops"])
            selected_data["health"].update(split_data["health"])

    else:
        selected_data = counts_by_split[selected_split]

    class_counts = selected_data["classes"]
    crop_counts = selected_data["crops"]
    health_counts = selected_data["health"]

    selected_total = sum(class_counts.values())

    st.caption(
        f"Showing statistics for {selected_total:,} images."
    )

    # Class distribution
    st.markdown("### 1. Disease Class Distribution")

    sorted_classes = sorted(
        class_counts.items(),
        key=lambda item: item[1]
    )

    class_labels = [
        format_class_name(name)
        for name, _ in sorted_classes
    ]

    class_values = [
        count
        for _, count in sorted_classes
    ]

    fig, ax = plt.subplots(figsize=(12, 11))

    ax.barh(
        class_labels,
        class_values,
        color="#3A9363"
    )

    ax.set_xlabel("Number of Images")
    ax.set_ylabel("Crop and Condition")
    ax.set_title("Image Distribution Across 38 Classes")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)

    plt.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.caption(
        "This chart shows the number of images available for each "
        "crop-disease class. Differences in bar lengths indicate "
        "class imbalance."
    )

    st.write("")
    st.divider()

    # Crop distribution
    st.markdown("### 2. Image Distribution by Crop")

    sorted_crops = sorted(
        crop_counts.items(),
        key=lambda item: item[1],
        reverse=True
    )

    crop_labels = [
        crop.replace("_", " ").replace(",", ", ")
        for crop, _ in sorted_crops
    ]

    crop_values = [
        count
        for _, count in sorted_crops
    ]

    fig, ax = plt.subplots(figsize=(11, 6))

    bars = ax.bar(
        crop_labels,
        crop_values,
        color="#4D9B78"
    )

    ax.set_ylabel("Number of Images")
    ax.set_title("Images Across the 14 Crop Types")
    ax.tick_params(axis="x", rotation=40)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, crop_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=8
        )

    plt.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.write("")
    st.divider()

    # Healthy vs diseased
    st.markdown("### 3. Healthy vs. Diseased Images")

    healthy_count = health_counts["Healthy"]
    diseased_count = health_counts["Diseased"]

    col1, col2 = st.columns([1, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(6, 5))

        ax.pie(
            [healthy_count, diseased_count],
            labels=["Healthy", "Diseased"],
            autopct="%1.1f%%",
            startangle=90,
            colors=["#65B891", "#E58B68"],
            wedgeprops={"width": 0.42, "edgecolor": "white"},
            textprops={"fontsize": 10}
        )

        ax.set_title("Plant Health Distribution")

        plt.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    with col2:
        st.metric("Healthy Images", f"{healthy_count:,}")

        st.metric("Diseased Images", f"{diseased_count:,}")

        if selected_total:
            healthy_percentage = healthy_count / selected_total * 100
            diseased_percentage = diseased_count / selected_total * 100

            st.write(
                f"**Healthy:** {healthy_percentage:.2f}%"
            )

            st.write(
                f"**Diseased:** {diseased_percentage:.2f}%"
            )

    st.write("")
    st.divider()

    # Dataset splits
    st.markdown("### 4. Dataset Split Distribution")

    split_labels = ["Training", "Validation", "Testing"]

    split_values = [
        split_sizes["train"],
        split_sizes["validation"],
        split_sizes["test"]
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(
        split_labels,
        split_values,
        color=["#3A9363", "#77B89A", "#A4CEB8"]
    )

    ax.set_ylabel("Number of Images")
    ax.set_title("Training, Validation and Testing Images")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, split_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:,}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    # Split details
    split_col1, split_col2, split_col3 = st.columns(3)

    with split_col1:
        render_metric("Training", f"{split_sizes['train']:,}")

    with split_col2:
        render_metric("Validation", f"{split_sizes['validation']:,}")

    with split_col3:
        render_metric("Testing", f"{split_sizes['test']:,}")

    st.write("")
    st.divider()

    
    st.markdown("### 5. Class Imbalance Analysis")

    nonzero_classes = [
        (name, count)
        for name, count in class_counts.items()
        if count > 0
    ]

    if nonzero_classes:
        largest_class = max(nonzero_classes, key=lambda item: item[1])
        smallest_class = min(nonzero_classes, key=lambda item: item[1])

        ratio = (
            largest_class[1] / smallest_class[1]
            if smallest_class[1] > 0
            else 0
        )

        # Three imbalance metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Largest Class",
                f"{largest_class[1]:,}"
            )
            st.caption(format_class_name(largest_class[0]))

        with col2:
            st.metric(
                "Smallest Class",
                f"{smallest_class[1]:,}"
            )
            st.caption(format_class_name(smallest_class[0]))

        with col3:
            st.metric(
                "Class Imbalance Ratio",
                f"{ratio:.2f}:1"
            )
            st.caption("Largest class compared with smallest class")

        # Full-width explanation box
        st.markdown(
            """
            <div style="
                width: 100%;
                box-sizing: border-box;
                background-color: #edf7f0;
                border: 1px solid #c9e2d1;
                border-left: 5px solid #3a9363;
                padding: 18px 22px;
                border-radius: 12px;
                color: #173d2b;
                line-height: 1.7;
                margin-top: 15px;
            ">
                <strong>Why class imbalance matters</strong><br>
                The dataset contains different numbers of images across
                the crop-condition classes. This imbalance was addressed
                during model development through class-weighted training,
                helping the model give greater consideration to
                underrepresented classes.
            </div>
            """,
            unsafe_allow_html=True
        )


    # Dataset information
    st.write("")
    st.divider()

    st.markdown("### Dataset Information")

    info_col1, info_col2 = st.columns(2)

    with info_col1:
        st.markdown("**Dataset:** PlantVillage")
        st.markdown(f"**Image format:** {metadata['image_mode']}")
        st.markdown(f"**Image dimensions:** {image_size}")

    with info_col2:
        st.markdown(f"**Crop-condition classes:** {number_of_classes}")
        st.markdown(f"**Crop types:** {number_of_crops}")
        st.markdown(f"**Total images:** {total_images:,}")

    st.caption(
        "All statistics are calculated from the cleaned and prepared "
        "dataset used in the LeafLens model training pipeline."
    )
