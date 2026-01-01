# 如何修改 YOLOv12 函式庫的原始碼，去新增客製化的資料增強功能
hyp_dawn.yaml 檔案裡的 motion_blur_prob、defocus_prob 和 fog_prob 參數，是傳遞給 Python 程式碼的「開關」和「機率」。程式碼本身沒有實作「動態模糊」或「起霧」的功能，要做的，就是去修改 ultralytics 函式庫的原始碼，手動把這些功能加進去。

警告：這非常複雜
這需要你對 Python、OpenCV 和 ultralytics 函式庫的內部結構有深入的了解。這不是一個機器學習（ML）任務，而是一個軟體開發（Dev）任務。

最簡單的方法，仍然是使用我們前一步驟討論的，直接 git clone 並安裝那個已經幫你改好程式碼的客製化專案。

完整程式碼修改步驟（概念範例）
假設你已經從官方 ultralytics GitHub git clone 了原始碼，並透過 pip install -e . 安裝了可編輯模式。

步驟 1：定位到增強檔案
你需要找到 ultralytics 函式庫中負責資料增強的 Python 檔案。這通常是： ultralytics/data/augment.py

步驟 2：匯入必要的函式庫
打開 ultralytics/data/augment.py 檔案，在最上面加入 cv2 (OpenCV) 和 numpy：

Python
import cv2
import numpy as np
import random
# ... (其他 import)
步驟 3：新增你客製化的增強類別 (Class)
在檔案的空白處（例如檔案結尾，或其他類別定義的旁邊），加入以下這些全新的 Python 類別，用來實作模糊和起霧效果：

Python
# =================================================================
# ======== S T A R T :  C U S T O M  A U G M E N T A T I O N S ========
# =================================================================

class MotionBlur:
    """
    對影像應用動態模糊
    """
    def __init__(self, k_min=3, k_max=7):
        # 決定模糊核心的大小範圍
        self.k_min = k_min
        self.k_max = k_max

    def __call__(self, im, labels=None):
        k = random.randint(self.k_min, self.k_max)
        if k % 2 == 0:
            k += 1  # 核心必須是奇數
        
        # 產生一個隨機方向的動態模糊核心
        kernel = np.zeros((k, k))
        angle = random.uniform(0, 180)
        
        if angle == 0:
            kernel[int((k - 1) / 2), :] = 1
        elif angle == 90:
            kernel[:, int((k - 1) / 2)] = 1
        else:
            radian = angle * np.pi / 180.0
            x = int(np.round(np.cos(radian) * (k // 2)))
            y = int(np.round(np.sin(radian) * (k // 2)))
            cv2.line(kernel, (k // 2 - x, k // 2 - y), (k // 2 + x, k // 2 + y), 1.0)
            
        kernel = kernel / np.sum(kernel) # 歸一化
        im = cv2.filter2D(im, -1, kernel)
        return im, labels

class Defocus:
    """
    對影像應用散焦（高斯模糊）
    """
    def __init__(self, k_min=3, k_max=5):
        self.k_min = k_min
        self.k_max = k_max

    def __call__(self, im, labels=None):
        k = random.randint(self.k_min, self.k_max)
        if k % 2 == 0:
            k += 1  # 核心必須是奇數
        im = cv2.GaussianBlur(im, (k, k), 0)
        return im, labels

class Fog:
    """
    對影像添加霧氣效果
    """
    def __init__(self, fog_intensity_min=0.05, fog_intensity_max=0.35):
        self.fog_min = fog_intensity_min
        self.fog_max = fog_intensity_max

    def __call__(self, im, labels=None):
        fog_intensity = random.uniform(self.fog_min, self.fog_max)
        
        # 建立一個全白的霧氣圖層
        fog_layer = np.full_like(im, (255, 255, 255), dtype=np.uint8)
        
        # 透過 addWeighted 進行影像融合
        # im * (1 - intensity) + fog_layer * (intensity)
        im = cv2.addWeighted(im, 1 - fog_intensity, fog_layer, fog_intensity, 0)
        return im, labels

# =================================================================
# ======== E N D :  C U S T O M  A U G M E N T A T I O N S ========
# =================================================================
步驟 4：將新類別整合到主要的增強流程中
現在你有了工具，但還需要告訴 ultralytics 在什麼時候使用它們。你需要找到主要的增強類別。

在 augment.py 檔案中，尋找一個主要的類別，它可能叫做 Compose、Transforms 或是 v8_transforms。它會有一個 __init__ 方法（用來讀取設定）和一個 __call__ 方法（用來執行增強）。

假設這個類別叫做 v8_transforms（在 YOLOv8 中是這個名字，v12 可能類似）：

1. 修改 __init__ 方法：

找到 def __init__(self, ...):，你需要在這裡讀取你的客製化超參數，並初始化你的新類別。

Python
# --- 找到 __init__ 方法 ---
# (這是一個範例，你需要找到你版本中真實的 __init__)
class v8_transforms:
    def __init__(self, ... , hyp=None):
        # ... (這裡有大量原有的程式碼，例如 mosaic, hsv 等)
        # ...
        
        # ======== 在 __init__ 的結尾處加入以下程式碼 ========
        
        # 從 hyp 字典讀取你的客製化機率
        # .get(key, 0.0) 的意思是：嘗試讀取 key，如果不存在，預設為 0.0
        self.motion_blur_prob = hyp.get('motion_blur_prob', 0.0)
        self.defocus_prob = hyp.get('defocus_prob', 0.0)
        self.fog_prob = hyp.get('fog_prob', 0.0)

        # 讀取霧氣強度 (如果 hyp 中有定義的話)
        fog_intensity = hyp.get('fog_intensity', [0.05, 0.35])

        # 初始化你的客製化類別實例 (instance)
        if self.motion_blur_prob > 0.0:
            self.motion_blur = MotionBlur(k_min=3, k_max=7)
        if self.defocus_prob > 0.0:
            self.defocus = Defocus(k_min=3, k_max=5)
        if self.fog_prob > 0.0:
            self.fog = Fog(fog_intensity_min=fog_intensity[0], fog_intensity_max=fog_intensity[1])

        # ======== 客製化程式碼結束 ========
2. 修改 __call__ 方法：

找到 def __call__(self, ...):，這是實際執行增強的地方。你需要在影像增強的最後階段（通常是在 mosaic、hsv 等處理完之後，但在轉換為 Tensor 之前）加入呼叫。

Python
# --- 找到 __call__ 方法 ---
    def __call__(self, im, labels, ...):
        # ...
        # ... (這裡有大量原有的增強程式碼，例如 hsv 變換)
        # if random.random() < self.hsv_h:
        #    ...
        # ...
        
        # ======== 在影像轉換為 Tensor 之前，加入以下程式碼 ========
        
        # 應用動態模糊
        if self.motion_blur_prob > 0.0 and random.random() < self.motion_blur_prob:
            im, labels = self.motion_blur(im, labels)
            
        # 應用散焦
        if self.defocus_prob > 0.0 and random.random() < self.defocus_prob:
            im, labels = self.defocus(im, labels)
            
        # 應用霧氣
        if self.fog_prob > 0.0 and random.random() < self.fog_prob:
            im, labels = self.fog(im, labels)

        # ======== 客製化程式碼結束 ========
        
        # ... (後面可能是轉換為 Tensor 或其他的程式碼)
        # im = ...
        # labels = ...
        return im, labels
步驟 5：儲存檔案並執行訓練
儲存你修改過的 ultralytics/data/augment.py 檔案。

因為你是在「可編輯模式」(-e) 下安裝的，所以這些修改會立即生效。

現在，回到你的 C:\Users\wangs\monkeyv7 資料夾。

執行你那串包含所有客製化參數的 yolo train 指令：

Bash
yolo train model="yolo12n.pt" data="datasets/wildlife/wildlife.yaml" ... (省略)... motion_blur_prob=0.2 defocus_prob=0.15 fog_prob=0.25 ... (省略) ... ema=true
這一次，yolo 程式在啟動時會讀取 motion_blur_prob=0.2，並將其存入 hyp 字典。當 v8_transforms 被初始化時，self.motion_blur_prob 會被設為 0.2。在 __call__ 方法中，程式會進行 random.random() < 0.2 的判斷，如果為 True，就會執行你剛剛加入的 self.motion_blur(im, labels) 函式。

你就成功地「自訓（實作）了」這些客製化參數。