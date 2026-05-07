import cv2
import numpy as np
import onnxruntime as ort
import time
import os
import glob

# =========================
# GLOBALS & SETTINGS
# =========================
# Suppress global TensorRT builder warnings (Must be before session creation)
ort.set_default_logger_severity(3) 

# Map class IDs to specific names
CLASS_NAMES = {
    0: "good",
    1: "broken",
    2: "flashover"
}

# Adjust this value (e.g., 0.25 to 0.4) to catch more/fewer predictions
CONFIDENCE_THRESHOLD = 0.3 

# Visualization settings
BBOX_COLOR = (0, 255, 0)  # Bright Green in BGR
BBOX_THICKNESS = 3
TEXT_COLOR = (0, 255, 0)


def main():
    # =========================
    # 1. Load ONNX with TensorRT cache
    # =========================
    print("Loading model onto GPU with TensorRT cache...")

    model_path = 'optimized_model.onnx'
    cache_dir = "./trt_cache"

    # Create cache folder
    os.makedirs(cache_dir, exist_ok=True)

    # Lowered workspace size to 1GB to prevent "insufficient memory for tactic" warnings
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

    # Suppress warnings (Set logging level to 3: Error only)
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
    # 2. Setup folders
    # =========================
    input_folder = "../data/sample_images"
    output_folder = "output"

    os.makedirs(output_folder, exist_ok=True)

    image_extensions = (
        "*.jpg", "*.jpeg", "*.png",
        "*.JPG", "*.JPEG", "*.PNG"
    )

    image_paths = []
    for ext in image_extensions:
        image_paths.extend(
            glob.glob(os.path.join(input_folder, ext))
        )

    total_images = len(image_paths)

    if total_images == 0:
        print(f"No images found in '{input_folder}'")
        return

    print(f"Found {total_images} images.\n")

    # =========================
    # 3. Metrics
    # =========================
    total_inference_time = 0.0
    start_batch_time = time.time()

    # =========================
    # 4. Process all images
    # =========================
    for idx, image_path in enumerate(image_paths, 1):

        filename = os.path.basename(image_path)
        img = cv2.imread(image_path)

        if img is None:
            print(f"Could not read {filename}")
            continue

        img_h, img_w = img.shape[:2]

        # -------------------------
        # Preprocess
        # -------------------------
        input_img = cv2.resize(img, (640, 640))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

        input_img = input_img.transpose(2, 0, 1)
        input_img = np.expand_dims(input_img, axis=0)

        input_img = input_img.astype(np.float32)
        input_img /= 255.0

        # -------------------------
        # Inference
        # -------------------------
        start_infer_time = time.time()

        outputs = session.run(
            None,
            {input_name: input_img}
        )

        infer_time = time.time() - start_infer_time
        total_inference_time += infer_time

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

            # Updated to use the new lower confidence threshold
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

        # Ensure NMS also respects the new confidence threshold
        indices = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            score_threshold=CONFIDENCE_THRESHOLD,
            nms_threshold=0.4
        )

        # -------------------------
        # Draw boxes
        # -------------------------
        defects_found = len(indices) if len(indices) > 0 else 0

        if defects_found > 0:
            for i in indices.flatten():

                x, y, w, h = boxes[i]
                current_class_id = class_ids[i]
                
                # Fetch string label, fallback to "Unknown" if ID is outside 0-2
                class_name = CLASS_NAMES.get(current_class_id, f"Class {current_class_id}")

                # Draw thicker rectangle
                cv2.rectangle(
                    img,
                    (x, y),
                    (x + w, y + h),
                    BBOX_COLOR,
                    BBOX_THICKNESS
                )

                # Format string: e.g., "broken: 0.45"
                label = f"{class_name}: {scores[i]:.2f}"

                # Add slightly larger/thicker text background for readability
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img, (x, y - text_height - 10), (x + text_width, y), BBOX_COLOR, -1)

                cv2.putText(
                    img,
                    label,
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0), # Black text on the green background
                    2
                )

        # -------------------------
        # Save output
        # -------------------------
        output_filepath = os.path.join(
            output_folder,
            filename
        )

        cv2.imwrite(output_filepath, img)

        if idx % 10 == 0 or idx == total_images:
            print(f"Processed {idx}/{total_images}")

    # =========================
    # 5. Final metrics
    # =========================
    total_batch_time = time.time() - start_batch_time

    inference_fps = (
        total_images / total_inference_time
        if total_inference_time > 0 else 0
    )

    end_to_end_fps = (
        total_images / total_batch_time
        if total_batch_time > 0 else 0
    )

    print("\n" + "=" * 50)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 50)
    print(f"Total images processed : {total_images}")
    print(f"Pure inference FPS     : {inference_fps:.2f}")
    print(f"Pipeline FPS           : {end_to_end_fps:.2f}")
    print(f"Output folder          : ./{output_folder}/")
    print(f"Cache folder           : ./{cache_dir}/")
    print("=" * 50)


if __name__ == "__main__":
    main()
