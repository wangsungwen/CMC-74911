import argparse
import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
import cv2

def analyze_cosine_similarity(model_path, video_path, yaml_path, output_dir):
    """
    Performs a two-stage process:
    1. Track objects to get bounding boxes and track IDs.
    2. Extract feature embeddings for each tracked object.
    3. Perform cosine similarity analysis.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load Model ---
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # --- 2. First Pass: Tracking ---
    print(f"Processing video for tracking: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        sys.exit(1)

    tracked_objects = []
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(source=frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            boxes = results[0].boxes.xyxy.cpu().numpy()

            for track_id, class_id, box in zip(track_ids, class_ids, boxes):
                # Crop the object from the frame
                x1, y1, x2, y2 = map(int, box)
                cropped_img = frame[y1:y2, x1:x2]
                
                if cropped_img.size > 0:
                    tracked_objects.append({
                        "track_id": track_id,
                        "cls_id": class_id,
                        "image": cropped_img
                    })
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  Processed {frame_idx} frames for tracking...")
    
    cap.release()
    print(f"Tracking complete. Found {len(tracked_objects)} object instances.")

    if not tracked_objects:
        print("Error: No objects were tracked in the video.")
        sys.exit(1)

    # --- 3. Second Pass: Feature Extraction ---
    print("Extracting features for tracked objects...")
    all_features = []
    # Use a model specifically for embedding if possible, here we use the same model
    # The `embed` method is the correct one for this task.
    for i, obj in enumerate(tracked_objects):
        # The embed method expects a list of images
        results = model.embed(source=[obj['image']], verbose=False)
        if results and results[0] is not None:
            all_features.append({
                "track_id": obj['track_id'],
                "cls_id": obj['cls_id'],
                "feature": results[0].flatten().cpu().numpy() # Convert to CPU numpy array
            })
        if (i+1) % 50 == 0:
            print(f"  Extracted features for {i+1}/{len(tracked_objects)} objects...")

    if not all_features:
        print("Error: Could not extract any features from the tracked objects.")
        sys.exit(1)

    df = pd.DataFrame(all_features)
    print(f"Successfully extracted {len(df)} feature vectors.")

    # --- 4. Load Class Names ---
    names = {}
    yaml_path = Path(yaml_path)
    if yaml_path.exists():
        with open(yaml_path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        names = d.get("names", {})
    df['cls_name'] = df['cls_id'].map(names).fillna(df['cls_id'].apply(lambda x: f"cls{x}"))

    # --- 5. Calculate Representative Features ---
    features_df = df.groupby(['track_id', 'cls_name'])['feature'].apply(
        lambda x: np.mean(np.vstack(x), axis=0)
    ).reset_index()
    print(f"Calculated representative features for {len(features_df)} unique tracks.")

    # --- 6. Cosine Distance Calculation ---
    intra_class_dist = []
    inter_class_dist = []
    all_classes = features_df['cls_name'].unique()

    for cls_name in all_classes:
        class_features = features_df[features_df['cls_name'] == cls_name]
        if len(class_features) > 1:
            for (_, track1), (_, track2) in combinations(class_features.iterrows(), 2):
                dist = cosine(track1['feature'], track2['feature'])
                intra_class_dist.append({'class': cls_name, 'distance': dist})

    if len(all_classes) > 1:
        for cls1_name, cls2_name in combinations(all_classes, 2):
            cls1_features = features_df[features_df['cls_name'] == cls1_name]['feature'].tolist()
            cls2_features = features_df[features_df['cls_name'] == cls2_name]['feature'].tolist()
            for feat1 in cls1_features:
                for feat2 in cls2_features:
                    dist = cosine(feat1, feat2)
                    inter_class_dist.append({'class1': cls1_name, 'class2': cls2_name, 'distance': dist})

    # --- 7. Save to CSV ---
    if intra_class_dist:
        pd.DataFrame(intra_class_dist).to_csv(output_dir / "intra_class_cosine_distances.csv", index=False)
        print(f"Saved intra-class distances to CSV.")
    if inter_class_dist:
        pd.DataFrame(inter_class_dist).to_csv(output_dir / "inter_class_cosine_distances.csv", index=False)
        print(f"Saved inter-class distances to CSV.")

    # --- 8. Visualization ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    if intra_class_dist:
        sns.kdeplot(data=pd.DataFrame(intra_class_dist), x='distance', ax=ax, color='blue', label='Intra-class Distance', fill=True)
    if inter_class_dist:
        sns.kdeplot(data=pd.DataFrame(inter_class_dist), x='distance', ax=ax, color='red', label='Inter-class Distance', fill=True)
    ax.set_title('Distribution of Cosine Distances', fontsize=16)
    ax.set_xlabel('Cosine Distance', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend()
    plt.savefig(output_dir / "cosine_distance_distribution.png", dpi=150)
    print(f"Saved visualization to: {output_dir / 'cosine_distance_distribution.png'}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv12 Cosine Similarity Analysis")
    parser.add_argument("--model", type=str, default="best.pt", help="Path to the YOLO model.")
    parser.add_argument("--video", type=str, default="wildlife.mp4", help="Path to the video file.")
    parser.add_argument("--yaml", type=str, default="datasets/wildlife/wildlife.yaml", help="Path to the dataset YAML file.")
    parser.add_argument("--output_dir", type=str, default="runs/analyze/cosine_analysis", help="Directory to save results.")
    args = parser.parse_args()

    analyze_cosine_similarity(args.model, args.video, args.yaml, args.output_dir)
    print("\nAnalysis complete.")
