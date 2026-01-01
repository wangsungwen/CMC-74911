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
import sklearn.metrics.pairwise # Import the module explicitly

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

    # --- 5. Group features by track_id and cls_name ---
    # We need individual features for intra-ID similarity, not just the mean.
    # Store all features for each track_id.
    track_features_grouped = df.groupby('track_id').apply(
        lambda x: {'cls_name': x['cls_name'].iloc[0], 'features': x['feature'].tolist()}
    ).reset_index(name='data')
    print(f"Grouped features for {len(track_features_grouped)} unique tracks.")

    # --- 6. Cosine Similarity Calculation ---
    intra_id_similarities = []
    intra_class_dist = []
    inter_class_dist = []
    
    # Calculate Intra-ID Similarity
    print("Calculating Intra-ID similarities...")
    for index, row in track_features_grouped.iterrows():
        track_id = row['track_id']
        cls_name = row['data']['cls_name']
        features = row['data']['features']
        
        if len(features) > 1:
            # Calculate pairwise cosine similarity for features of the same track
            sim_matrix = sklearn.metrics.pairwise.cosine_similarity(features) # Use fully qualified name
            # Get upper triangle of the similarity matrix (excluding diagonal)
            upper_triangle_indices = np.triu_indices_from(sim_matrix, k=1)
            if len(upper_triangle_indices[0]) > 0:
                avg_sim = np.mean(sim_matrix[upper_triangle_indices])
                # Store as distance (1 - similarity)
                intra_id_similarities.append({'track_id': track_id, 'class': cls_name, 'avg_similarity': avg_sim, 'avg_distance': 1 - avg_sim})

    # Calculate Intra-Class and Inter-Class Distances
    # For this, we use the representative feature (mean) for each track_id
    representative_features_df = df.groupby(['track_id', 'cls_name'])['feature'].apply(
        lambda x: np.mean(np.vstack(x), axis=0)
    ).reset_index()
    print(f"Calculated representative features for {len(representative_features_df)} unique tracks for inter/intra-class analysis.")

    all_classes = representative_features_df['cls_name'].unique()

    # Intra-Class Distance (between different track IDs of the same class)
    for cls_name in all_classes:
        class_tracks_features = representative_features_df[representative_features_df['cls_name'] == cls_name]
        if len(class_tracks_features) > 1:
            # Iterate through all unique pairs of tracks within the same class
            for i in range(len(class_tracks_features)):
                for j in range(i + 1, len(class_tracks_features)):
                    track1_data = class_tracks_features.iloc[i]
                    track2_data = class_tracks_features.iloc[j]
                    
                    dist = cosine(track1_data['feature'], track2_data['feature'])
                    intra_class_dist.append({'class': cls_name, 'distance': dist})

    # Inter-Class Distance (between different classes)
    if len(all_classes) > 1:
        for cls1_name, cls2_name in combinations(all_classes, 2):
            cls1_features_list = representative_features_df[representative_features_df['cls_name'] == cls1_name]['feature'].tolist()
            cls2_features_list = representative_features_df[representative_features_df['cls_name'] == cls2_name]['feature'].tolist()
            
            for feat1 in cls1_features_list:
                for feat2 in cls2_features_list:
                    dist = cosine(feat1, feat2)
                    inter_class_dist.append({'class1': cls1_name, 'class2': cls2_name, 'distance': dist})

    # --- 7. Save to CSV ---
    if intra_id_similarities:
        intra_id_df = pd.DataFrame(intra_id_similarities)
        intra_id_df.to_csv(output_dir / "intra_id_cosine_similarities_bytetrack.csv", index=False)
        print(f"Saved intra-ID similarities to CSV.")
    if intra_class_dist:
        pd.DataFrame(intra_class_dist).to_csv(output_dir / "intra_class_cosine_distances_bytetrack.csv", index=False)
        print(f"Saved intra-class distances to CSV.")
    if inter_class_dist:
        pd.DataFrame(inter_class_dist).to_csv(output_dir / "inter_class_cosine_distances_bytetrack.csv", index=False)
        print(f"Saved inter-class distances to CSV.")

    # --- 8. Visualization ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 14)) # Two plots stacked vertically

    # Plot 1: Intra-ID and Intra-Class Distances
    ax1 = axes[0]
    if intra_id_similarities:
        intra_id_df = pd.DataFrame(intra_id_similarities)
        sns.kdeplot(data=intra_id_df, x='avg_distance', ax=ax1, color='green', label='Intra-ID Distance (Self-Consistency)', fill=True)
    if intra_class_dist:
        intra_class_df = pd.DataFrame(intra_class_dist)
        sns.kdeplot(data=intra_class_df, x='distance', ax=ax1, color='blue', label='Intra-class Distance (Different IDs, Same Class)', fill=True)
    if inter_class_dist:
        inter_class_df = pd.DataFrame(inter_class_dist)
        sns.kdeplot(data=inter_class_df, x='distance', ax=ax1, color='red', label='Inter-class Distance (Different Classes)', fill=True)

    ax1.set_title('Distribution of Cosine Distances', fontsize=16)
    ax1.set_xlabel('Cosine Distance', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.legend()

    # Plot 2: Intra-ID Similarity Distribution (optional, or combined if space allows)
    # For simplicity, let's focus on distances in the first plot.
    # If we want to show similarity, we'd need to convert distances back or plot similarity directly.
    # Let's keep it focused on distances for now, as requested by the original script's output.
    # If needed, a separate plot for intra-ID similarity could be added.
    # For now, we'll just use the first plot for all distance metrics.
    # We can remove the second axes if not used.
    fig.delaxes(axes[1]) # Remove the second unused axes

    plt.tight_layout() # Adjust layout to prevent overlap
    plt.savefig(output_dir / "cosine_distance_distribution_bytetrack.png", dpi=150)
    print(f"Saved visualization to: {output_dir / 'cosine_distance_distribution_bytetrack.png'}")
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
