import os
import time
import fiftyone as fo
import fiftyone.zoo as foz

# 定義類別列表
classes = [
    'Person',  # 人 - 0
    'Dog',     # 狗 - 7
    'Monkey',  # 猴子 - 340 (OpenImages 索引)
]

# 建構類別索引映射
class_to_index = {cls: idx for idx, cls in enumerate(classes)}

def update_txt_file_class_indices(class_name):
    """
    更新 YOLO 格式標籤檔中的類別索引，確保與 classes 列表一致
    """
    # 設定相對路徑 (移除原本的 / 開頭，改為當前目錄下的 yolov5 資料夾)
    base_dir = os.path.join("yolov5", "open-images-v7")
    labels_dir = os.path.join(base_dir, class_name, 'labels', 'val')
    dataset_yaml = os.path.join(base_dir, class_name, 'dataset.yaml')
    
    # 刪除自動生成的 dataset.yaml (因為這是單一類別的 yaml，合併訓練時通常不需要)
    if os.path.exists(dataset_yaml):
        os.remove(dataset_yaml)
        
    if os.path.exists(labels_dir):
        for filename in os.listdir(labels_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(labels_dir, filename)

                # 讀取並處理 .txt 文件
                with open(filepath, 'r') as file:
                    lines = file.readlines()

                # 更新類別索引
                updated_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5: # 假設每行至少有 5 個元素
                        # parts[0] 是類別索引
                        try:
                            # 嘗試從映射表中取得正確索引
                            updated_class_idx = str(class_to_index[class_name])
                        except KeyError: 
                            # 如果在映射表中找不到該類別，則保留原名稱或進行其他處理
                            print(f"Warning: Class '{class_name}' not found in index map.")
                            updated_class_idx = class_name
                        
                        parts[0] = updated_class_idx
                        updated_lines.append(' '.join(parts) + '\n')

                # 寫回更新後的行
                with open(filepath, 'w') as file:
                    file.writelines(updated_lines)
        print(f"[{class_name}] 類別的所有 .txt 標籤索引已更新完畢。")

# 主程式迴圈
for class_name in classes:
    # 每次處理前，先清除 FiftyOne 內部的暫存資料集名稱，避免衝突
    try:
        if "open-images-v7-train-2000" in fo.list_datasets():
            fo.delete_dataset("open-images-v7-train-2000")
    except Exception as e:
        print(f"Cleanup warning: {e}")

    print(f"\n正在處理 {class_name} 類別...")

    # 下載並載入 Open Images v7 資料集 (加入重試機制)
    dataset = None
    max_retries = 3
    for attempt in range(max_retries):
        try:
            dataset = foz.load_zoo_dataset(
                "open-images-v7",
                split="train", # train, validation, test
                label_types=["detections"],
                classes=[class_name],
                max_samples=2000, # 每個類別篩選前 2000 個樣本
                shuffle=True,
                only_matching=True, # 僅下載符合條件的圖片
                num_workers=1, # 設為 1 以提高穩定性 (避免多執行緒下載錯誤)
                seed=42,
                dataset_name="open-images-v7-train-2000",
                # 指定下載暫存目錄 (可選)
                # dataset_dir=os.path.join("fiftyone_download", class_name) 
            )
            break # 成功則跳出重試迴圈
        except Exception as e:
            print(f"Error processing {class_name} (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print("Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print("Max retries reached. Skipping this class.")
    
    if dataset is None:
        continue

    # 建立一個新資料集來保存篩選後的樣本
    filtered_dataset = fo.Dataset()
    
    # 過濾標籤：只保留當前關注的類別 (例如下載 'Person'圖片時，只保留 'Person' 的框，去掉背景雜訊)
    for sample in dataset:
        if sample.ground_truth:
            filtered_detections = [d for d in sample.ground_truth.detections if d.label == class_name]
            
            if filtered_detections:
                new_sample = sample.copy()
                new_sample.ground_truth.detections = filtered_detections
                filtered_dataset.add_sample(new_sample)

    count = filtered_dataset.count()
    if count == 0:
        print(f"No samples found for class: {class_name}")
        continue
    
    print(f"{class_name} 類別的有效樣本數量為: {count}")
    
    # 定義匯出路徑 (相對路徑)
    export_path = os.path.join("yolov5", "open-images-v7", class_name)
    
    # 匯出為 YOLOv5 格式
    filtered_dataset.export(
        export_dir=export_path,
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="ground_truth",
    )
    
    # 修正標籤索引 (將 OpenImages 的字串標籤轉為我們自定義的 0, 1, 2...)
    update_txt_file_class_indices(class_name)

print("\n所有類別處理完成！")