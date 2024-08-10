
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



# Container for the main content
with st.container():
    # Title of the app
    st.title("Find Waldo: Real-Time Detection with YOLO")

    # Description of the app
    st.write("""
    **Find Waldo: Real-Time Detection with YOLO** is an interactive web app that leverages the power of the 
    YOLO (You Only Look Once) object detection algorithm to locate Waldo in images. 
    Simply upload your image, and watch as our AI model quickly identifies and highlights Waldo, 
    making your search fun and effortless. Ideal for fans of the classic 'Where's Waldo?' series and 
    anyone interested in AI-powered image recognition technology.
    """)

    # Dropdown menu to select YOLO model
    selected_model = st.selectbox("Choose a YOLO model", options=list(model_paths.keys()))
    model_path = model_paths[selected_model]
    
    # Load the selected model
    if selected_model == "YOLOv8":
        detection_model = load_yolov8_model(model_path)
    else:
        detection_model = load_yolov5_model(model_path)

    # File uploader for user to upload image
    uploaded_image = st.file_uploader("Choose a Waldo image")

    if uploaded_image is not None:
        # Load image with PIL
        image = PILImage.open(io.BytesIO(uploaded_image.read()))

        # Convert RGBA to RGB if necessary
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        st.image(image, caption="Uploaded Image", use_column_width=True)
    else:
        st.text("Please upload an image to detect Waldo")

    # Detection buttons
    col1, col2 = st.columns(2)

    with col1:
        button_yolo = st.button('Detect with YOLO')

    with col2:
        button_sahi = st.button('Detect with YOLO and SAHI')

    result_placeholder = st.empty()

    # Display result based on button clicked
    if button_yolo:
        if uploaded_image is not None:
            result_placeholder.text("Detecting Waldo with YOLO...")

            # Convert PIL image to NumPy array
            image_np = np.array(image)

            result = get_prediction(image_np, detection_model)
            result.export_visuals(export_dir="demo_data/")
            st.image("demo_data/prediction_visual.png", caption="YOLO Prediction", use_column_width=True)
        else:
            st.text("Please upload an image first.")

    if button_sahi:
        if uploaded_image is not None:
            result_placeholder.text("Detecting Waldo with YOLO and SAHI...")

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
            st.image("demo_data/prediction_visual.png", caption="YOLO and SAHI Prediction", use_column_width=True)
        else:
            st.text("Please upload an image first.")


