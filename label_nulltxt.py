# 腳本功能說明：
# 自動對應：程式會讀取圖片的檔名（例如 forest_01.jpg），並生成同名的文字檔（forest_01.txt）。
# 安全機制：程式碼中加入了一段判斷 if txt_path.exists() and txt_path.stat().st_size > 0:。如果該圖片已經有標註資料（檔案大小大於 0），程式會跳過該檔案，避免您不小心覆蓋掉已標註的資料。
# 自動建立資料夾：如果輸出的 labels 資料夾不存在，程式會自動建立。

import os
from pathlib import Path

def generate_empty_labels(images_dir, labels_dir):
    """
    讀取 images_dir 中的圖片，並在 labels_dir 生成對應的空白 txt 檔。
    """
    # 定義支援的圖片格式
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif'}
    
    # 確保路徑是 Path 物件
    img_path = Path(images_dir)
    lbl_path = Path(labels_dir)

    # 如果標籤資料夾不存在，則自動建立
    if not lbl_path.exists():
        lbl_path.mkdir(parents=True, exist_ok=True)
        print(f"已建立標籤資料夾: {lbl_path}")

    generated_count = 0
    skipped_count = 0

    print(f"正在掃描資料夾: {img_path}...")

    # 遍歷資料夾內所有檔案
    for file in img_path.iterdir():
        if file.suffix.lower() in valid_extensions:
            # 定義對應的 txt 檔名
            txt_filename = file.stem + ".txt"
            txt_path = lbl_path / txt_filename

            # 安全檢查：如果 txt 檔案已經存在且不為空，跳過以防覆蓋重要資料
            if txt_path.exists() and txt_path.stat().st_size > 0:
                print(f"跳過 (檔案已存在且不為空): {txt_filename}")
                skipped_count += 1
                continue

            # 建立空白檔案
            with open(txt_path, 'w') as f:
                pass  # 什麼都不寫，產生空文件
            
            generated_count += 1

    print("-" * 30)
    print(f"處理完成。")
    print(f"成功生成空白標籤: {generated_count} 個")
    print(f"跳過現有非空標籤: {skipped_count} 個")

# ==========================================
# 請在下方修改您的資料夾路徑
# ==========================================
if __name__ == "__main__":
    # 輸入：放置背景圖片的資料夾路徑
    background_images_path = "dataset/images/train_backgrounds" 
    
    # 輸出：希望生成 txt 檔的資料夾路徑
    background_labels_path = "dataset/labels/train_backgrounds"
    
    # 執行函式
    if os.path.exists(background_images_path):
        generate_empty_labels(background_images_path, background_labels_path)
    else:
        print(f"錯誤：找不到圖片資料夾 {background_images_path}")