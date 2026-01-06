import os
from pathlib import Path
import re # Import the re module for regular expressions

# --- 設定你的標籤資料夾路徑 ---
# 根據你的 wildlife.yaml, 訓練集包含 'train' 和 'train_pseudo'
label_dirs = [
    Path(r"C:\Users\wangs\monkeyv7\datasets\wildlife\labels\train"),
    Path(r"C:\Users\wangs\monkeyv7\datasets\wildlife\labels\train_pseudo"),
    Path(r"C:\Users\wangs\monkeyv7\datasets\wildlife\labels\val") # Added val directory here for completeness
]
# -----------------------------

bad_files = []
valid_class_id = 0  # 你的 wildlife.yaml 只有 '0: monkey'

for label_dir in label_dirs:
    if not label_dir.exists():
        print(f"⚠️ 警告：找不到資料夾 {label_dir}，跳過掃描。")
        continue

    print(f"--- 正在掃描 {label_dir} ---")
    
    # 找出所有 .txt 檔案
    txt_files = list(label_dir.glob("*.txt"))
    if not txt_files:
        print(f"   (在 {label_dir.name} 中未找到任何 .txt 檔案)")
        continue
        
    for txt_file in txt_files:
        if txt_file.stat().st_size == 0:
            continue  # 跳過空檔案 (背景圖)
        
        try:
            # Read in binary mode to explicitly handle BOM
            with open(txt_file, 'rb') as f_bytes:
                content_bytes = f_bytes.read()
            
            # Remove UTF-8 BOM if present (b'\xef\xbb\xbf')
            if content_bytes.startswith(b'\xef\xbb\xbf'):
                content_bytes = content_bytes[3:]
            
            # Decode to string (now guaranteed to be without BOM bytes)
            content_str = content_bytes.decode('utf-8', errors='ignore') # Ignore decoding errors
            lines = content_str.splitlines(keepends=True)
            for i, line in enumerate(lines):
                # Ensure Unicode BOM is stripped if it persists in the string
                line = line.lstrip('\ufeff')
                
                parts = line.strip().split()
                if not parts:
                    continue # Skip empty lines

                # 1. 檢查 Class ID
                # Isolate digits from parts[0] to handle persistent BOM ('ï»¿0')
                class_id_str = ''.join(filter(str.isdigit, parts[0]))
                if not class_id_str:
                    error_msg = f"❌ 格式錯誤! 檔案: {txt_file.name} (在 {label_dir.name} 中), 行: {i+1}, 錯誤: '在第一個元素中找不到數字'"
                    print(error_msg)
                    if error_msg not in bad_files:
                        bad_files.append(error_msg)
                    continue

                class_id = int(class_id_str)
                
                if class_id != valid_class_id:
                    error_msg = f"❌ 類別 ID 錯誤! 檔案: {txt_file.name} (在 {label_dir.name} 中), 行: {i+1}, ID: {class_id}"
                    print(error_msg)
                    if error_msg not in bad_files:
                        bad_files.append(error_msg)
                    continue 

                # 2. 檢查座標數量
                coords = [float(p) for p in parts[1:]] # Use parts[1:] as parts[0] is clean class_id
                if not len(coords) == 4:
                    error_msg = f"❌ 座標數量錯誤! 檔案: {txt_file.name} (在 {label_dir.name} 中), 行: {i+1}"
                    print(error_msg)
                    if error_msg not in bad_files:
                        bad_files.append(error_msg)
                    continue

                # 3. 檢查座標範圍
                for coord in coords:
                    if not (0.0 <= coord <= 1.0):
                        error_msg = f"❌ 座標超出範圍! 檔案: {txt_file.name} (在 {label_dir.name} 中), 行: {i+1}, 值: {coord}"
                        print(error_msg)
                        if error_msg not in bad_files:
                            bad_files.append(error_msg)
                        break # 這行有問題，跳到下一行
                        
        except Exception as e:
            error_msg = f"❌ 格式錯誤! 檔案: {txt_file.name} (在 {label_dir.name} 中), 行: {i+1}, 錯誤: {e}"
            print(error_msg)
            if error_msg not in bad_files:
                bad_files.append(error_msg)

if not bad_files:
    print("\n✅ 掃描完成，所有檢查的資料夾中未發現明顯錯誤。")
else:
    print(f"\n掃描完成，總共發現 {len(bad_files)} 個問題。請修正它們。")
