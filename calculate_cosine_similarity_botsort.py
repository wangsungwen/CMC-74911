# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import pandas as pd
import argparse
from ultralytics import YOLO
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_reid_model(model_path):
    """載入 Re-ID 模型"""
    print(f"Loading Re-ID model from {model_path}...")
    model = YOLO(model_path)
    model.to(DEVICE)
    return model

def extract_features(model, image_path):
    """從單張圖片中提取 Re-ID 特徵"""
    results = model(image_path, verbose=False)
    return results[0].obb.cls.cpu().numpy() if results[0].obb is not None else None

def parse_tracking_results(track_dir):
    """
    解析追蹤結果，從 crops/{class}/{video}_{frame}_{id}.jpg 的檔案結構中提取資訊。
    """
    print("Parsing tracking results from file names...")
    crops_dir = os.path.join(track_dir, 'crops')
    
    if not os.path.exists(crops_dir):
        print(f"Crops directory not found at {crops_dir}. Cannot perform Re-ID analysis.")
        return None

    # 使用 defaultdict 讓每個 track_id 對應一個包含 class 和 image 列表的字典
    id_info = defaultdict(lambda: {'class': None, 'images': []})

    class_folders = [d for d in os.listdir(crops_dir) if os.path.isdir(os.path.join(crops_dir, d))]
    if not class_folders:
        print("No class folders found in crops directory.")
        return None

    for class_name in class_folders:
        class_path = os.path.join(crops_dir, class_name)
        
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png'))]
        
        for filename in image_files:
            try:
                # 從檔名 'monkey_video_1_100.jpg' 中提取 track_id '100'
                base_name = os.path.splitext(filename)[0]
                track_id_str = base_name.split('_')[-1]
                track_id = int(track_id_str)
                
                # 儲存資訊
                id_info[track_id]['class'] = class_name
                id_info[track_id]['images'].append(os.path.join(class_path, filename))

            except (IndexError, ValueError):
                print(f"Could not parse track ID from filename: {filename}. Skipping.")
                continue
    
    # 將 id_info 轉換為 id_data 的格式: {track_id: [{'class': name, 'images': [...]}]}
    id_data = defaultdict(list)
    for track_id, info in id_info.items():
        if info['class'] and info['images']:
             id_data[track_id].append({
                 'class': info['class'],
                 'images': info['images']
             })

    if not id_data:
        print("Failed to parse any valid tracking data from the crops folder.")
        return None

    return id_data

def analyze_cosine_similarity(reid_model, id_data):
    """分析餘弦相似度"""
    print("Analyzing cosine similarity...")
    id_features = defaultdict(list)

    # 1. 為每個 ID 的每張圖片提取特徵
    print("Step 1: Extracting Re-ID features for each tracked object...")
    for track_id, data_list in id_data.items():
        for data in data_list:
            for img_path in data['images']:
                feature = extract_features(reid_model, img_path)
                if feature is not None:
                    id_features[track_id].append(feature.flatten())

    # 2. 計算 ID 內部（intra-ID）的餘弦相似度
    print("Step 2: Calculating intra-ID (self) cosine similarity...")
    intra_id_similarities = defaultdict(list)
    for track_id, features in id_features.items():
        if len(features) > 1:
            sim_matrix = cosine_similarity(features)
            # 取上三角矩陣（不含對角線）的平均值
            indices = np.triu_indices_from(sim_matrix, k=1)
            if len(indices[0]) > 0:
                avg_sim = np.mean(sim_matrix[indices])
                intra_id_similarities[track_id] = avg_sim

    # 3. 計算類別內部（intra-class）的餘弦相似度 (更細緻的方法)
    print("Step 3: Calculating intra-class (cross-ID) cosine similarity...")
    
    # 依據 class_name 將 track_id 分組
    class_to_ids = defaultdict(list)
    for track_id, data_list in id_data.items():
        if track_id in id_features:
            class_name = data_list[0]['class']
            class_to_ids[class_name].append(track_id)

    intra_class_distances = []
    # 遍歷每個類別
    for class_name, track_ids in class_to_ids.items():
        # 如果該類別有多於一個 track_id，才進行比較
        if len(track_ids) > 1:
            # 兩兩比較該類別中的 track_id
            from itertools import combinations
            for id_a, id_b in combinations(track_ids, 2):
                features_a = id_features[id_a]
                features_b = id_features[id_b]
                
                if features_a and features_b:
                    # 計算 A 的所有特徵與 B 的所有特徵之間的距離
                    dist_matrix = 1 - cosine_similarity(features_a, features_b)
                    
                    # 將所有距離值加入列表
                    for dist in dist_matrix.flatten():
                        intra_class_distances.append({'class': class_name, 'cosine_distance': dist})
    
    return intra_id_similarities, intra_class_distances

def save_results(output_dir, intra_class_distances):
    """儲存結果到 CSV 並繪圖"""
    print("Saving results...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 儲存類內餘弦距離
    csv_path = os.path.join(output_dir, 'intra_class_cosine_distances.csv')
    df = pd.DataFrame(intra_class_distances)
    df.to_csv(csv_path, index=False)
    print(f"Intra-class cosine distances saved to {csv_path}")

    # 繪製分佈圖
    if not df.empty:
        plt.figure(figsize=(12, 7))
        sns.histplot(data=df, x='cosine_distance', hue='class', kde=True, multiple="stack")
        plt.title('Distribution of Intra-Class Cosine Distances (Different IDs, Same Class)')
        plt.xlabel('Cosine Distance (1 - Similarity)')
        plt.ylabel('Count')
        plot_path = os.path.join(output_dir, 'cosine_distance_distribution.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Distribution plot saved to {plot_path}")
    else:
        print("Skipping plot generation: No intra-class distance data to plot.")
        print("This usually means that each class only had one tracked object, so no cross-ID distances could be calculated.")

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="Calculate Cosine Similarity for Re-ID features.")
    parser.add_argument('--track_dir', type=str, required=True, help='Path to the tracking results directory.')
    parser.add_argument('--model', type=str, required=True, help='Path to the Re-ID model.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save the analysis results.')
    args = parser.parse_args()

    if not os.path.exists(args.track_dir):
        print(f"Error: Tracking directory '{args.track_dir}' not found.")
        return

    reid_model = load_reid_model(args.model)
    id_data = parse_tracking_results(args.track_dir)
    
    if id_data:
        intra_id_sim, intra_class_dist = analyze_cosine_similarity(reid_model, id_data)
        save_results(args.output_dir, intra_class_dist)
        print("Analysis complete.")
    else:
        print("Could not parse tracking data. Aborting analysis.")

if __name__ == '__main__':
    main()
