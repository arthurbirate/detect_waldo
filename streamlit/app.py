import streamlit as st
import os
import io
from PIL import Image as PILImage
import numpy as np
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction
from sahi.utils.yolov8 import download_yolov8s_model

# Define paths for available YOLO models
model_paths = {
    "YOLOv8": "model/bestModelx8.pt",
}

# Function to download and load the YOLOv8 model
def load_yolov8_model(model_path):
    if not os.path.exists(model_path):
        download_yolov8s_model(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=model_path,
        confidence_threshold=0.50,
        device=device
    )

# Set the page configuration
st.set_page_config(page_title="Find Waldo", page_icon="👓", layout="centered", initial_sidebar_state="collapsed")

# CSS for theme compatibility and centering content
st.markdown(
    """
    <style>
    .main {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
        background-color: var(--background-color);
        margin: 0;
    }
    .centered-container {
        max-width: 800px;
        width: 100%;
        padding: 20px;
        background-color: var(--background-color);
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .stButton>button {
        background-color: var(--primary-color);
        color: var(--text-color);
        border-radius: 10px;
    }
    .stTextInput>div>input, .stFileUploader>div>input {
        background-color: var(--background-secondary);
        color: var(--text-color);
        border: none;
        padding: 10px;
    }
    .stFileUploader>label {
        color: var(--text-color);
    }
    h1, h2, h3, h4, h5, h6, p, div, span, .stMarkdown {
        color: var(--text-color);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Centered container for the main content
with st.container():
    st.markdown('<div class="centered-container">', unsafe_allow_html=True)

    # Title of the app with emoji
    st.title("🔍 Find Waldo: Real-Time Detection with YOLO")

    # Description of the app with styled markdown
    st.markdown(
        """
        **Find Waldo** is an interactive web app that leverages the power of the 
        YOLO (You Only Look Once) object detection algorithm to locate Waldo in images.
        Simply upload your image, and watch as our AI model quickly identifies and highlights Waldo,
        making your search fun and effortless. Ideal for fans of the classic 'Where's Waldo?' series
        and anyone interested in AI-powered image recognition technology.
        """
    )

    # Horizontal divider for better separation
    st.markdown("---")

    # Model selection dropdown
    st.subheader("Model Selection")
    selected_model = st.selectbox("Choose a YOLO model", options=list(model_paths.keys()))
    model_path = model_paths[selected_model]

    # Load the selected model
    detection_model = load_yolov8_model(model_path)
    st.success("Model loaded successfully! 🎉")

    # Main content area for uploading images
    st.subheader("Upload Image")
    uploaded_image = st.file_uploader("Upload a Waldo image", type=["png", "jpg", "jpeg","webp"])

    if uploaded_image is not None:
        # Load image with PIL
        image = PILImage.open(io.BytesIO(uploaded_image.read()))

        # Convert RGBA to RGB if necessary
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        # Display the uploaded image with a caption
        st.image(image, caption="Uploaded Image", use_column_width=True)
    else:
        st.info("Please upload an image to detect Waldo")

    # Detection buttons in columns for better alignment
    col1, col2 = st.columns([1, 1])

    with col1:
        button_yolo = st.button('Detect with YOLO')
    with col2:
        button_sahi = st.button('Detect with YOLO + SAHI')

    result_placeholder = st.empty()

    # Display result based on button clicked
    if button_yolo:
        if uploaded_image is not None:
            result_placeholder.info("Detecting Waldo with YOLO...")

            # Convert PIL image to NumPy array
            image_np = np.array(image)

            result = get_prediction(image_np, detection_model)
            result.export_visuals(export_dir="demo_data/")
            st.image("demo_data/prediction_visual.png", caption="YOLO Prediction", use_column_width=True)
        else:
            st.warning("Please upload an image first.")

    if button_sahi:
        if uploaded_image is not None:
            result_placeholder.info("Detecting Waldo with YOLO + SAHI...")

            # Convert PIL image to NumPy array
            image_np = np.array(image)

            result = get_sliced_prediction(
                image_np,  # Pass NumPy array image
                detection_model,
                slice_height=256,
                slice_width=256,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2
            )
            result.export_visuals(export_dir="demo_data/")
            st.image("demo_data/prediction_visual.png", caption="YOLO + SAHI Prediction", use_column_width=True)
        else:
            st.warning("Please upload an image first.")

    st.markdown('</div>', unsafe_allow_html=True)
