import os
import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns

# --- Placeholders ---
labels_dir = Path(r"__LABEL_DIR__")
out_dir = Path(r"__SESSION_DIR__")
yaml_path = Path(r"__YAML__") if r"__YAML__" else None
# --- End Placeholders ---

out_dir.mkdir(parents=True, exist_ok=True)

# Load class names
names = {}
if yaml_path and yaml_path.exists():
    with open(yaml_path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    raw = d.get("names", {})
    if isinstance(raw, dict):
        names = {int(k): str(v) for k, v in raw.items()}
    elif isinstance(raw, list):
        names = {i: str(v) for i, v in enumerate(raw)}
else:
    print("Warning: YAML file not found. Using generic class names.")

# Parse tracking files with features
rows = []
print(f"Parsing files from: {labels_dir}")
for txt_file in sorted(labels_dir.glob("*.txt")):
    for line in txt_file.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 8:  # class, bbox, conf, track_id + at least one feature
            continue
        try:
            class_id = int(float(parts[0]))
            track_id = int(float(parts[6]))
            feature_vector = np.array([float(x) for x in parts[7:]])
            rows.append({
                "cls_id": class_id,
                "cls_name": names.get(class_id, f"cls{class_id}"),
                "track_id": track_id,
                "feature": feature_vector
            })
        except (ValueError, IndexError):
            continue

if not rows:
    print("Error: No valid tracking data with features found.")
    sys.exit(1)

df = pd.DataFrame(rows)
print(f"Successfully parsed {len(df)} detections.")

# Calculate a single representative feature vector for each track (by averaging)
features_df = df.groupby(['track_id', 'cls_name'])['feature'].apply(
    lambda x: np.mean(np.vstack(x), axis=0)
).reset_index()
print(f"Calculated representative features for {len(features_df)} unique tracks.")

# --- Cosine Distance Calculation ---
intra_class_dist = []
inter_class_dist = []

all_classes = features_df['cls_name'].unique()
print(f"Found classes: {all_classes}")

# Intra-class distances
for cls_name in all_classes:
    class_features = features_df[features_df['cls_name'] == cls_name]
    if len(class_features) > 1:
        # Get all pairs of tracks within the same class
        for (idx1, track1), (idx2, track2) in combinations(class_features.iterrows(), 2):
            dist = cosine(track1['feature'], track2['feature'])
            intra_class_dist.append({'class': cls_name, 'distance': dist})

# Inter-class distances
if len(all_classes) > 1:
    # Get all pairs of classes
    for cls1_name, cls2_name in combinations(all_classes, 2):
        cls1_features = features_df[features_df['cls_name'] == cls1_name]['feature'].tolist()
        cls2_features = features_df[features_df['cls_name'] == cls2_name]['feature'].tolist()
        # Get all pairs of tracks between the two classes
        for feat1 in cls1_features:
            for feat2 in cls2_features:
                dist = cosine(feat1, feat2)
                inter_class_dist.append({'class1': cls1_name, 'class2': cls2_name, 'distance': dist})

print(f"Calculated {len(intra_class_dist)} intra-class distances.")
print(f"Calculated {len(inter_class_dist)} inter-class distances.")

# --- Save to CSV ---
if intra_class_dist:
    intra_df = pd.DataFrame(intra_class_dist)
    intra_df.to_csv(out_dir / "intra_class_cosine_distances.csv", index=False)
    print(f"Saved intra-class distances to CSV.")

if inter_class_dist:
    inter_df = pd.DataFrame(inter_class_dist)
    inter_df.to_csv(out_dir / "inter_class_cosine_distances.csv", index=False)
    print(f"Saved inter-class distances to CSV.")

# --- Visualization ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 7))

if intra_class_dist:
    sns.kdeplot(data=intra_df, x='distance', ax=ax, color='blue', label='Intra-class Distance', fill=True)

if inter_class_dist:
    sns.kdeplot(data=inter_df, x='distance', ax=ax, color='red', label='Inter-class Distance', fill=True)

ax.set_title('Distribution of Cosine Distances', fontsize=16)
ax.set_xlabel('Cosine Distance', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.legend()

output_path = out_dir / "cosine_distance_distribution.png"
plt.savefig(output_path, dpi=150)
print(f"Saved visualization to: {output_path}")
plt.close()

print("Analysis complete.")
