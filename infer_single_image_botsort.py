# infer_single_image_botsort.py
import cv2
import argparse
import os
from ultralytics import YOLO

# --- Constants ---
# Default model path (update if necessary)
DEFAULT_MODEL_PATH = r"best.pt"
# Default image source (update if necessary)
DEFAULT_IMAGE_SOURCE = "video/test.png"
# Output directory for results
OUTPUT_DIR = "runs/track_single_botsort"

def track_on_image(image_path, model_path):
    """
    Performs tracking on a single image using a specified YOLO model and BOTSORT tracker.

    Args:
        image_path (str): Path to the input image.
        model_path (str): Path to the YOLO model weights file.
    """
    print(f"INFO: Loading model from: {model_path}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"FATAL ERROR: Failed to load model. {e}")
        return

    print(f"INFO: Performing tracking on image: {image_path}")
    if not os.path.exists(image_path):
        print(f"FATAL ERROR: Image not found at path: {image_path}")
        return

    try:
        # Perform tracking
        results = model.track(source=image_path, save=False, imgsz=1280, conf=0.45, tracker="botsort.yaml")

        # Check if any results were returned
        if not results:
            print("WARNING: No objects detected or an error occurred during tracking.")
            return

        # Get the first result object
        r = results[0]
        
        # Use the plot() method to get the image with bounding boxes and track IDs
        frame = r.plot()

        # --- Save the output image ---
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Define the output path
        base_name = os.path.basename(image_path)
        file_name, file_ext = os.path.splitext(base_name)
        output_path = os.path.join(OUTPUT_DIR, f"{file_name}_tracked{file_ext}")

        # Save the image
        cv2.imwrite(output_path, frame)
        print(f"✅ Tracking complete. Result saved to: {output_path}")

    except Exception as e:
        print(f"FATAL ERROR: An error occurred during the tracking process: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLOv12 Single Image Tracking with BOTSORT.")
    parser.add_argument("--image", type=str, default=DEFAULT_IMAGE_SOURCE, help="Path to the input image.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to the model weights file.")
    args = parser.parse_args()

    track_on_image(args.image, args.model)
