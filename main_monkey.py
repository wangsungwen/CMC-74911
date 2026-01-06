import os
import fiftyone as fo
import fiftyone.zoo as foz
import time

# 定義類別列表
classes = [
    'Person', # 人 - 0
    'Dog', # 狗 - 7
    'Monkey', # 猴子 - 340
]

# 构建类别索引映射
class_to_index = {cls: idx for idx, cls in enumerate(classes)}

def update_txt_file_class_indices(class_name):
    """
    更新 YOLO 格式標籤檔中的類別索引，確保與 classes 列表一致
    """
    # 注意：路徑設定需根據您的實際環境調整
    labels_dir = os.path.join("yolov5/open-images-v7", class_name, 'labels/val')
    dataset_yaml = os.path.join("yolov5/open-images-v7", class_name, 'dataset.yaml')
    
    # 刪除自動生成的 dataset.yaml
    if os.path.exists(dataset_yaml):
        os.remove(dataset_yaml)
        
    if os.path.exists(labels_dir):
        for filename in os.listdir(labels_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(labels_dir, filename)

                # 读取并处理.txt文件
                with open(filepath, 'r') as file:
                    lines = file.readlines()

                # 更新类别索引
                updated_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5: # 假设每行至少有5个元素 (class index在第0个位置)
                        class_idx_str = parts[0] # 原类别索引 (可能是名称也可能是数字)
                        try:
                            # 嘗試從映射表中取得正確索引
                            updated_class_idx = str(class_to_index[class_name])
                        except ValueError: # 原始截圖為 ValueError，但建議注意 KeyError
                            # 如果转换失败，则按类别名称补全
                            updated_class_idx = class_name
                            # 這裡原截圖邏輯似乎是想處理異常，將名稱填回
                        
                        parts[0] = updated_class_idx
                        updated_lines.append(' '.join(parts) + '\n')

                # 写回更新后的行
                with open(filepath, 'w') as file:
                    file.writelines(updated_lines)
        print(f"{class_name}类别的所有.txt文件的索引已更新。")

# 主程式迴圈
for class_name in classes:
    # 嘗試刪除舊的暫存資料集
    try:
        fo.delete_dataset('open-images-v7-train-2000')
    except:
        pass
    print(f"正在处理{class_name}类别...")

    # 下載並載入 Open Images v7 資料集
    max_retries = 3
    for attempt in range(max_retries):
        try:
            dataset = foz.load_zoo_dataset(
                "open-images-v7",
                split="train", # train validation test
                label_types=["detections"], # "detections", "segmentations", "points"
                classes=[class_name],
                max_samples=2000, # 筛选前2000个样本
                shuffle=True,
                only_matching=True, # 指定仅下载符合条件的图片
                num_workers=1, # Reduced to 1 for stability
                seed=42, # 指定随机种子
                dataset_name="open-images-v7-train-2000",
            )
            break # Success, exit retry loop
        except Exception as e:
            print(f"Error processing {class_name} (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print("Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print("Max retries reached. Raising exception.")
                raise

    # 创建一个新数据集来保存筛选后的样本
    filtered_dataset = fo.Dataset()
    
    # 遍历原数据集
    for sample in dataset:
        # 匹配的检测框 (只保留當前處理類別的框)
        filtered_detections = [d for d in sample.ground_truth.detections if d.label == class_name]

        # 如果样本中还有匹配的检测框，创建样本副本并添加到新数据集中
        if filtered_detections:
            new_sample = sample.copy()
            new_sample.ground_truth.detections = filtered_detections
            filtered_dataset.add_sample(new_sample)

    if filtered_dataset.count() == 0:
        print(f"No samples found for class: {class_name}")
        continue
    
    print(f"{class_name}类别的样本数量为: {filtered_dataset.count()}")
    
    # 导出这个筛选后的数据集為 YOLOv5 格式
    result = filtered_dataset.export(
        export_dir=f"/yolov5/open-images-v7/{class_name}/", # 注意：這裡是絕對路徑
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="ground_truth",
    )
    
    # 呼叫函式修正標籤索引
    update_txt_file_class_indices(class_name)