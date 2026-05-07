import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import time
import os
from datetime import datetime

# =========================
# GLOBALS & SETTINGS
# =========================
ort.set_default_logger_severity(3) 

CLASS_NAMES = {
    0: "good",
    1: "broken",
    2: "flashover"
}

BBOX_COLOR = (0, 255, 0)  # Bright Green
BBOX_THICKNESS = 3

# Ensure output directory exists locally
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# CACHED MODEL LOADING
# =========================
@st.cache_resource
def load_model():
    model_path = 'optimized_model.onnx'
    
    providers = [
        (
            "TensorrtExecutionProvider",
            {
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": "./trt_cache",
                "trt_max_workspace_size": 1073741824 
            }
        ),
        "CUDAExecutionProvider"
    ]

    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3

    session = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=providers
    )
    
    input_name = session.get_inputs()[0].name
    return session, input_name

# =========================
# INFERENCE FUNCTION
# =========================
def run_inference(img, session, input_name, conf_thresh):
    img_h, img_w = img.shape[:2]

    # Preprocess
    input_img = cv2.resize(img, (640, 640))
    input_img = input_img.transpose(2, 0, 1)
    input_img = np.expand_dims(input_img, axis=0).astype(np.float32) / 255.0

    # Inference
    start_time = time.time()
    outputs = session.run(None, {input_name: input_img})
    infer_time = (time.time() - start_time) * 1000

    # Postprocess
    predictions = np.squeeze(outputs[0]).T
    boxes, scores, class_ids = [], [], []

    x_factor = img_w / 640
    y_factor = img_h / 640

    for row in predictions:
        class_scores = row[4:]
        max_score = np.max(class_scores)

        if max_score > conf_thresh:
            class_id = np.argmax(class_scores)
            xc, yc, w, h = row[0], row[1], row[2], row[3]

            x_min = int((xc - w / 2) * x_factor)
            y_min = int((yc - h / 2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)

            boxes.append([x_min, y_min, width, height])
            scores.append(float(max_score))
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=conf_thresh, nms_threshold=0.4)
    
    defects_found = len(indices) if len(indices) > 0 else 0

    # Draw Boxes
    if defects_found > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            current_class_id = class_ids[i]
            class_name = CLASS_NAMES.get(current_class_id, f"Class {current_class_id}")
            
            cv2.rectangle(img, (x, y), (x + w, y + h), BBOX_COLOR, BBOX_THICKNESS)
            label = f"{class_name}: {scores[i]:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x, y - th - 10), (x + tw, y), BBOX_COLOR, -1)
            cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return img, defects_found, infer_time

# =========================
# STREAMLIT UI LAYOUT
# =========================
st.set_page_config(page_title="Defect Diagnosis UI", layout="wide")

st.title("🔍 Edge AI Defect Diagnosis")
st.markdown("Run TensorRT-optimized YOLOv8 inference directly from your browser.")

# Load Model
with st.spinner('Loading TensorRT Engine...'):
    session, input_name = load_model()

# Sidebar controls
st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.1, max_value=1.0, value=0.3, step=0.05
)

st.sidebar.divider()

# Input Method Selector
input_method = st.sidebar.radio(
    "Choose Input Method:",
    ("📁 Upload Image(s)", "📷 Webcam Capture")
)

st.sidebar.info(f"Processed images are automatically saved to `./{OUTPUT_DIR}/`")

# ----------------------------------------
# MODE 1: FILE UPLOAD (Single or Multiple)
# ----------------------------------------
if input_method == "📁 Upload Image(s)":
    uploaded_files = st.file_uploader(
        "Upload images (Select multiple files to process a batch)", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        cols = st.columns(2) # Create a 2-column grid
        
        for idx, uploaded_file in enumerate(uploaded_files):
            # Read image
            image = Image.open(uploaded_file)
            img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Run Inference
            result_img, defect_count, inf_time = run_inference(
                img_array, session, input_name, confidence_threshold
            )
            
            # Save to output folder
            output_filepath = os.path.join(OUTPUT_DIR, uploaded_file.name)
            cv2.imwrite(output_filepath, result_img)
            
            # Display in UI
            result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            with cols[idx % 2]:
                st.image(result_img_rgb, caption=f"{uploaded_file.name}")
                st.write(f"**Defects Found:** {defect_count} | **Time:** {inf_time:.2f} ms")
                st.divider()

# ----------------------------------------
# MODE 2: WEBCAM CAPTURE
# ----------------------------------------
elif input_method == "📷 Webcam Capture":
    # Streamlit native webcam widget
    camera_image = st.camera_input("Take a picture to analyze")

    if camera_image is not None:
        # Read image from camera buffer
        image = Image.open(camera_image)
        img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Run Inference
        result_img, defect_count, inf_time = run_inference(
            img_array, session, input_name, confidence_threshold
        )
        
        # Generate a unique filename using timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"webcam_{timestamp}.jpg"
        output_filepath = os.path.join(OUTPUT_DIR, filename)
        
        # Save to output folder
        cv2.imwrite(output_filepath, result_img)
        
        # Display in UI
        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        
        st.success(f"Image processed and saved locally as `{filename}`")
        st.image(result_img_rgb, caption="Webcam Analysis Result")
        st.write(f"**Defects Found:** {defect_count}")
        st.write(f"**Inference Time:** {inf_time:.2f} ms")
