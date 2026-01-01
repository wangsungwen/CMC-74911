# -*- coding: utf-8 -*-
import cv2
import torch
import numpy as np
import pandas as pd
import argparse
from ultralytics import YOLO
from collections import defaultdict
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

def run_analysis(video_path, track_model_path, embed_model_path, output_dir):
    """
    執行完整的追蹤、特徵提取與餘弦距離分析流程。
    """
    # --- 1. 載入模型 ---
    print(f"Loading tracking model from {track_model_path}...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    track_model = YOLO(track_model_path)
    track_model.to(device)

    print(f"Loading embedding model from {embed_model_path}...")
    embed_model = YOLO(embed_model_path)
    embed_model.to(device)

    # --- 2. 物件追蹤 & 3. 特徵提取 ---
    print(f"Starting tracking and feature extraction on {video_path}...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    track_features = defaultdict(list)
    track_class = {}

    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        if frame_count % 5 == 0: # 每 5 幀處理一次以加速
             print(f"Processing frame {frame_count}...")

        # 使用 BoT-SORT 追蹤器
        results = track_model.track(frame, persist=True, tracker="botsort.yaml", verbose=False)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            
            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                # 裁切物件
                x1, y1, x2, y2 = map(int, box)
                crop = frame[y1:y2, x1:x2]
                
                if crop.size > 0:
                    # 使用 embedding 模型提取特徵
                    embedding_results = embed_model.predict(crop, verbose=False)
                    
                    feature_vector = None
                    # 優先使用分類模型的 .probs 輸出
                    if embedding_results[0].probs is not None:
                        feature_vector = embedding_results[0].probs.data.cpu().numpy()
                    # 如果是偵測模型，回退到使用 BBox 的歸一化座標作為特徵
                    elif embedding_results[0].boxes is not None and len(embedding_results[0].boxes) > 0:
                        # 使用第一個偵測到的框的 xywhn 作為特徵
                        feature_vector = embedding_results[0].boxes.xywhn[0].cpu().numpy()
                        if track_id not in track_class:
                             print(f"Warning: Embedding model is a detection model. Using BBox data as a fallback feature for track ID {track_id}.")
                    
                    if feature_vector is not None:
                        track_features[track_id].append(feature_vector.flatten())
                        if track_id not in track_class:
                            track_class[track_id] = track_model.names[class_id]
                    else:
                        print(f"Warning: Could not extract any features for track ID {track_id} from the cropped image.")

    cap.release()
    print("Tracking and feature extraction complete.")

    # --- 4. 計算代表性特徵 ---
    print("Calculating representative features for each track ID...")
    representative_features = {}
    for track_id, features in track_features.items():
        if features:
            representative_features[track_id] = np.mean(features, axis=0)

    # --- 5. 計算餘弦距離 ---
    print("Calculating cosine distances...")
    intra_class_distances = []
    inter_class_distances = []

    # 依類別分組 Track IDs
    class_to_ids = defaultdict(list)
    for track_id in representative_features:
        class_name = track_class.get(track_id)
        if class_name:
            class_to_ids[class_name].append(track_id)

    # 計算類內距離
    for class_name, ids in class_to_ids.items():
        if len(ids) > 1:
            for id1, id2 in combinations(ids, 2):
                feat1 = representative_features[id1].reshape(1, -1)
                # BUG FIX: feat2 should be representative_features[id2]
                feat2 = representative_features[id2].reshape(1, -1)
                dist = 1 - cosine_similarity(feat1, feat2)[0][0]
                intra_class_distances.append({'class': class_name, 'id1': id1, 'id2': id2, 'distance': dist})

    # 計算類間距離
    if len(class_to_ids) > 1:
        for (class1, ids1), (class2, ids2) in combinations(class_to_ids.items(), 2):
            for id1 in ids1:
                for id2 in ids2:
                    feat1 = representative_features[id1].reshape(1, -1)
                    feat2 = representative_features[id2].reshape(1, -1)
                    dist = 1 - cosine_similarity(feat1, feat2)[0][0]
                    inter_class_distances.append({'class1': class1, 'class2': class2, 'id1': id1, 'id2': id2, 'distance': dist})

    # --- 6. 產出結果 ---
    print("Saving results...")
    # 儲存 CSV
    # Ensure DataFrames are created with expected columns even if lists are empty to avoid KeyError later
    # Define expected columns for intra and inter class distances
    intra_cols = ['class', 'id1', 'id2', 'distance']
    inter_cols = ['class1', 'class2', 'id1', 'id2', 'distance']

    df_intra = pd.DataFrame(intra_class_distances, columns=intra_cols) if intra_class_distances else pd.DataFrame(columns=intra_cols)
    df_inter = pd.DataFrame(inter_class_distances, columns=inter_cols) if inter_class_distances else pd.DataFrame(columns=inter_cols)
    
    df_intra.to_csv(f"{output_dir}/intra_class_cosine_distances_botsort.csv", index=False)
    df_inter.to_csv(f"{output_dir}/inter_class_cosine_distances_botsort.csv", index=False)
    print(f"CSVs saved to {output_dir}")

    # 繪製圖表
    # Prepare data for plotting, ensuring columns exist and handling empty dataframes
    plot_data_list = []
    if not df_intra.empty and 'distance' in df_intra.columns:
        df_intra_plot = df_intra.copy()
        df_intra_plot['type'] = 'Intra-class'
        plot_data_list.append(df_intra_plot[['distance', 'type']])
    
    if not df_inter.empty and 'distance' in df_inter.columns:
        df_inter_plot = df_inter.copy()
        df_inter_plot['type'] = 'Inter-class'
        plot_data_list.append(df_inter_plot[['distance', 'type']])

    if plot_data_list: # Only plot if there's data to plot
        combined_df = pd.concat(plot_data_list, ignore_index=True)

        plt.figure(figsize=(12, 7))
        sns.histplot(data=combined_df, x='distance', hue='type', kde=True, common_norm=False)
        plt.title('Distribution of Intra-class and Inter-class Cosine Distances (BoT-SORT)')
        plt.xlabel('Cosine Distance (1 - Similarity)')
        plt.ylabel('Count')
        plt.grid(True)
        
        plot_path = f"{output_dir}/cosine_distance_distribution_botsort.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"Distribution plot saved to {plot_path}")
    else:
        print("No distances were calculated or data is missing required columns. Skipping plot generation.")

    print("Analysis finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="End-to-end analysis: tracking, embedding, and cosine similarity.")
    parser.add_argument('--video', required=True, help="Path to the input video file.")
    parser.add_argument('--track_model', required=True, help="Path to the YOLOv8 detection/tracking model (.pt).")
    parser.add_argument('--embed_model', required=True, help="Path to the YOLOv8 classification model for embedding (.pt).")
    parser.add_argument('--output', default='analysis_results', help="Directory to save the output files.")
    
    args = parser.parse_args()
    
    # 建立輸出目錄
    import os
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        
    run_analysis(args.video, args.track_model, args.embed_model, args.output)
