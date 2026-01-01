import cv2
import numpy as np
import os
import random
import glob
from tqdm import tqdm

# ================= 設定區 =================
# 1. 輸入背景圖片路徑 (使用您原本的訓練集圖片)
INPUT_BG_DIR = 'datasets/wildlife/images/train' 

# 2. 輸出合成數據的路徑
OUTPUT_IMG_DIR = 'datasets/wildlife_laser/train/images'
OUTPUT_LBL_DIR = 'datasets/wildlife_laser/train/labels'

# 3. 設定參數
NUM_IMAGES_TO_GENERATE = 2000  # 要生成幾張圖
LASER_CLASS_ID = 1             # 雷射的類別 ID (0是猴子, 1是雷射)
LASER_COLOR = (255, 0, 0)      # 紅色 (BGR)
LASER_RADIUS_MIN = 2           # 光點最小半徑
LASER_RADIUS_MAX = 6           # 光點最大半徑
MOTION_BLUR_PROB = 0.5         # 50% 機率產生拖影 (模擬快速移動)
# ==========================================

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def yolo_format(img_w, img_h, box):
    # box = [xmin, ymin, xmax, ymax]
    # Convert to [x_center, y_center, width, height] normalized
    dw = 1. / img_w
    dh = 1. / img_h
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h

def add_laser_spot(img):
    h, w = img.shape[:2]
    
    # 隨機決定位置
    x1 = random.randint(10, w - 10)
    y1 = random.randint(10, h - 10)
    
    # 創建一個遮罩層繪製雷射
    overlay = img.copy()
    
    is_motion = random.random() < MOTION_BLUR_PROB
    radius = random.randint(LASER_RADIUS_MIN, LASER_RADIUS_MAX)
    
    xmin, ymin, xmax, ymax = 0, 0, 0, 0
    
    if is_motion:
        # 模擬移動拖影 (畫線)
        x2 = x1 + random.randint(-40, 40)
        y2 = y1 + random.randint(-40, 40)
        
        # 確保不出界
        x2 = np.clip(x2, 5, w-5)
        y2 = np.clip(y2, 5, h-5)
        
        cv2.line(overlay, (x1, y1), (x2, y2), LASER_COLOR, thickness=radius*2)
        
        # 計算 Bounding Box
        xmin = min(x1, x2) - radius
        ymin = min(y1, y2) - radius
        xmax = max(x1, x2) + radius
        ymax = max(y1, y2) + radius
        
    else:
        # 靜止光點 (畫圓)
        # 多畫幾層讓中心更亮
        cv2.circle(overlay, (x1, y1), radius, LASER_COLOR, -1)
        cv2.circle(overlay, (x1, y1), radius - 2, (150, 255, 150), -1) # 中心泛白
        
        xmin = x1 - radius
        ymin = y1 - radius
        xmax = x1 + radius
        ymax = y1 + radius

    # 應用高斯模糊模擬光暈 (Glow Effect)
    overlay = cv2.GaussianBlur(overlay, (15, 15), 0)
    
    # 混合圖片 (雷射疊加到背景)
    # alpha 調整雷射亮度
    alpha = random.uniform(0.7, 0.95)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # 確保 box 不出界
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(w, xmax)
    ymax = min(h, ymax)
    
    return img, [xmin, ymin, xmax, ymax]

def main():
    create_folder(OUTPUT_IMG_DIR)
    create_folder(OUTPUT_LBL_DIR)
    
    # 取得所有背景圖片
    bg_images = glob.glob(os.path.join(INPUT_BG_DIR, '*.jpg')) + \
                glob.glob(os.path.join(INPUT_BG_DIR, '*.png'))
                
    if not bg_images:
        print(f"錯誤：在 {INPUT_BG_DIR} 找不到背景圖片！")
        return

    print(f"開始生成 {NUM_IMAGES_TO_GENERATE} 張合成數據...")
    
    for i in tqdm(range(NUM_IMAGES_TO_GENERATE)):
        # 隨機選一張背景
        bg_path = random.choice(bg_images)
        img = cv2.imread(bg_path)
        if img is None: continue
        
        h, w = img.shape[:2]
        
        # 添加雷射
        img_aug, box = add_laser_spot(img)
        
        # 存檔名稱
        filename = f"synth_laser_{i:05d}"
        
        # 1. 儲存圖片
        cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, filename + ".jpg"), img_aug)
        
        # 2. 儲存標籤 (YOLO format)
        # 注意：這裡我們只存雷射的標籤 (class 1)。
        # 如果背景圖原本有猴子，為了訓練簡單，這張合成圖我們可以只專注教模型認雷射。
        # 或者，您可以選擇讀取原背景的 label txt 把猴子也加進去 (更進階)。
        
        xc, yc, bw, bh = yolo_format(w, h, box)
        label_str = f"{LASER_CLASS_ID} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"
        
        with open(os.path.join(OUTPUT_LBL_DIR, filename + ".txt"), "w") as f:
            f.write(label_str)
            
    print("生成完畢！")
    print(f"圖片儲存於: {OUTPUT_IMG_DIR}")
    print(f"標籤儲存於: {OUTPUT_LBL_DIR}")

if __name__ == "__main__":
    main()