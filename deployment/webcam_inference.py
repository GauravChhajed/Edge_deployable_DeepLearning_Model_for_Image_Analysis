import cv2
import numpy as np
import onnxruntime as ort
import time
import os

# =========================
# GLOBALS & SETTINGS
# =========================
# Suppress global TensorRT builder warnings
ort.set_default_logger_severity(3) 

CLASS_NAMES = {
    0: "good",
    1: "broken",
    2: "flashover"
}

CONFIDENCE_THRESHOLD = 0.3 

BBOX_COLOR = (0, 255, 0)  # Bright Green
BBOX_THICKNESS = 3
TEXT_COLOR = (0, 0, 0)    # Black text for the label background

def main():
    # =========================
    # 1. Load ONNX with TensorRT cache
    # =========================
    print("Loading model onto GPU with TensorRT cache...")

    model_path = 'optimized_model.onnx'
    cache_dir = "./trt_cache"
    os.makedirs(cache_dir, exist_ok=True)

    providers = [
        (
            "TensorrtExecutionProvider",
            {
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": cache_dir,
                "trt_max_workspace_size": 1073741824  # 1 GB
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
    print("Model loaded successfully.\n")

    # =========================
    # 2. Initialize Webcam
    # =========================
    # 0 is usually the default built-in/USB webcam. 
    # If the Logitech Brio is not picked up, change this to 1 or 2.
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Optional: Set resolution to 720p or 1080p for the Logitech Brio
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Starting video stream. Press 'q' to quit.")

    # =========================
    # 3. Real-Time Processing Loop
    # =========================
    while True:
        # Start timer for FPS calculation
        start_time = time.time()

        ret, img = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        img_h, img_w = img.shape[:2]

        # -------------------------
        # Preprocess
        # -------------------------
        input_img = cv2.resize(img, (640, 640))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = input_img.transpose(2, 0, 1)
        input_img = np.expand_dims(input_img, axis=0).astype(np.float32)
        input_img /= 255.0

        # -------------------------
        # Inference
        # -------------------------
        outputs = session.run(None, {input_name: input_img})

        # -------------------------
        # Postprocess
        # -------------------------
        predictions = np.squeeze(outputs[0]).T

        boxes = []
        scores = []
        class_ids = []

        x_factor = img_w / 640
        y_factor = img_h / 640

        for row in predictions:
            class_scores = row[4:]
            max_score = np.max(class_scores)

            if max_score > CONFIDENCE_THRESHOLD:
                class_id = np.argmax(class_scores)
                xc, yc, w, h = row[0], row[1], row[2], row[3]

                x_min = int((xc - w / 2) * x_factor)
                y_min = int((yc - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                boxes.append([x_min, y_min, width, height])
                scores.append(float(max_score))
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            score_threshold=CONFIDENCE_THRESHOLD,
            nms_threshold=0.4
        )

        # -------------------------
        # Draw boxes
        # -------------------------
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                current_class_id = class_ids[i]
                class_name = CLASS_NAMES.get(current_class_id, f"Class {current_class_id}")

                cv2.rectangle(img, (x, y), (x + w, y + h), BBOX_COLOR, BBOX_THICKNESS)

                label = f"{class_name}: {scores[i]:.2f}"
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                # Draw label background and text
                cv2.rectangle(img, (x, y - text_height - 10), (x + text_width, y), BBOX_COLOR, -1)
                cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)

        # -------------------------
        # Calculate & Draw Live FPS
        # -------------------------
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        cv2.putText(img, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # -------------------------
        # Display the Video Feed
        # -------------------------
        cv2.imshow("Logitech Brio 100 - Real-Time Inference", img)

        # Listen for the 'q' key to stop the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting video stream...")
            break

    # =========================
    # 4. Cleanup
    # =========================
    cap.release()
    cv2.destroyAllWindows()
    print("Resources released cleanly.")

if __name__ == "__main__":
    main()
