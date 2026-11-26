import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from pages.prediction import render_prediction_page
from pages.insights import render_insights_page
from pages.retraining import render_retraining_page


# Page configuration

st.set_page_config(
    page_title="LeafLens | Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling 

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #f7f9f7;
    }

    [data-testid="stSidebar"] {
        background-color: #123b2a;
    }

    [data-testid="stSidebar"] * {
        color: #ffffff;
    }

    .brand-title {
        font-size: 30px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0;
    }

    .brand-subtitle {
        font-size: 12px;
        color: #c6ddce;
        margin-top: -5px;
    }

    .main-title {
        font-size: 34px;
        font-weight: 800;
        color: #173d2b;
        margin-bottom: 5px;
    }

    .main-subtitle {
        color: #6b7c70;
        font-size: 15px;
    }

    .metric-card {
        background-color: #ffffff;
        padding: 22px;
        border-radius: 14px;
        border: 1px solid #e5ece6;
    }

    .metric-label {
        color: #718075;
        font-size: 13px;
    }

    .metric-value {
        color: #173d2b;
        font-size: 28px;
        font-weight: 800;
    }

    .section-title {
        color: #173d2b;
        font-size: 20px;
        font-weight: 700;
    }


    /* Hide Streamlit's automatic page navigation */
    [data-testid="stSidebarNav"] {
        display: none;
    }

    /* Improve main content readability */
    [data-testid="stMain"] {
        background-color: #f7f9f7;
        color: #173d2b !important;
    }

    [data-testid="stMain"] p,
    [data-testid="stMain"] li,
    [data-testid="stMain"] label {
        color: #53665a !important;
    }

    [data-testid="stMain"] h1,
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3 {
        color: #173d2b !important;
    }

    /* Improve information message appearance */
    [data-testid="stMain"] .stAlert {
        border-radius: 12px;
    }

</style>
""", unsafe_allow_html=True)

# Sidebar

with st.sidebar:

    st.markdown(
        '<p class="brand-title">🌿 LeafLens</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="brand-subtitle">Intelligent Plant Health Monitoring</p>',
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### NAVIGATION")

    page = st.radio(
        "Go to",
        [
            "Dashboard",
            "Disease Detection",
            "Dataset Insights",
            "Model Retraining"        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.caption("LeafLens Plant Disease Classification")
    st.caption("Powered by MobileNetV2")

# Dashboard

if page == "Dashboard":

    st.title("Welcome to LeafLens 🌱")

    st.markdown(
        '<p class="main-subtitle">'
        'An intelligent platform for identifying plant diseases '
        'and monitoring crop health.'
        '</p>',
        unsafe_allow_html=True
    )

    st.write("")

    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Crop Types</div>
            <div class="metric-value">14</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Classification Classes</div>
            <div class="metric-value">38</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Test Accuracy</div>
            <div class="metric-value">95.84%</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Model Architecture</div>
            <div class="metric-value">MobileNetV2</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    # Introduction section
    left, right = st.columns([1.5, 1])

    with left:
        st.markdown(
            '<p class="section-title">About LeafLens</p>',
            unsafe_allow_html=True
        )

        st.write(
            "LeafLens uses deep learning to analyze plant leaf images "
            "and identify whether a crop is healthy or affected by disease."
        )

        st.write(
            "The platform supports 14 crop types and 38 classification "
            "classes, including both healthy and diseased leaves."
        )

        st.info(
            "Upload a plant leaf image in Disease Detection to get started."
        )

    with right:
        st.markdown(
            '<p class="section-title">What you can do</p>',
            unsafe_allow_html=True
        )

        st.markdown("""
        1. Detect plant diseases from images
        2. Identify healthy leaves
        3. Explore dataset insights
        4. Upload new data for model retraining
        5. Review model evaluation results
        """)

# OTHER PAGES

elif page == "Disease Detection":
    render_prediction_page()

elif page == "Dataset Insights":
    render_insights_page()


elif page == "Model Retraining":
    render_retraining_page()
