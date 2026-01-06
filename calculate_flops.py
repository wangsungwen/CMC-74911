import torch
from ultralytics import YOLO
from thop import profile
import logging

# --- 1. 參數設定 ---

# 設定您的 YOLO12n 模型權重檔案路徑
# *** 這是您必須修改的唯一地方 ***
MODEL_PATH = "yolo12n.pt"  # 請替換成您訓練好的.pt 檔案路徑

# 設定模型推論時的輸入影像尺寸
# *** 這必須與您在論文中報告 mAP 和 FPS 時所用的尺寸一致 ***
# (例如：640x640)
INPUT_SIZE = 640

# --- 2. 載入模型 ---

print(f"正在載入模型： {MODEL_PATH}")
try:
    # 載入 Ultralytics YOLO 模型
    model = YOLO(MODEL_PATH)
    print("模型載入成功。")

    # 關鍵步驟：
    # thop 需要一個純粹的 PyTorch nn.Module 來進行分析。
    # Ultralytics 的 YOLO 物件是一個包裝器 (wrapper)，
    # 我們需要提取其內部的實際模型架構，通常是.model
    pytorch_model = model.model
    
    # 將模型設置為評估模式 (evaluation mode)
    pytorch_model.eval()

    # 檢查模型是否成功提取
    if not isinstance(pytorch_model, torch.nn.Module):
        print("錯誤：未能成功提取 PyTorch nn.Module。請檢查模型結構。")
        exit()

except Exception as e:
    print(f"載入模型時發生錯誤: {e}")
    exit()

# --- 3. 準備虛擬輸入張量 (Dummy Input) ---

print(f"正在準備輸入尺寸為 (1, 3, {INPUT_SIZE}, {INPUT_SIZE}) 的虛擬輸入...")
# 建立一個符合模型輸入維度的虛擬張量
# (Batch Size=1, Channels=3, Height=INPUT_SIZE, Width=INPUT_SIZE)
try:
    dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    # 如果您的模型在 GPU 上，也請將輸入張量移至 GPU
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # pytorch_model = pytorch_model.to(device)
    # dummy_input = dummy_input.to(device)
    
    print("虛擬輸入準備完成。")
except Exception as e:
    print(f"準備虛擬輸入時發生錯誤: {e}")
    exit()

# --- 4. 執行 THOP 進行模型分析 ---

print("="*30)
print(f"開始使用 thop 分析模型 (輸入尺寸: {INPUT_SIZE}x{INPUT_SIZE})...")
logging.getLogger().setLevel(logging.ERROR) # 隱藏 thop 的詳細層級輸出

try:
    # 'profile' 函數會計算 MACs (Multiply-Accumulate Operations)
    # verbose=False 關閉逐層報告
    macs, params = profile(pytorch_model, inputs=(dummy_input, ), verbose=False)
    
    print("模型分析完成。")
    print("="*30)

    # --- 5. 解讀與計算 GFLOPs ---
    
    # thop 計算的是 MACs（乘積累加運算）
    # 理論上，1 個 MAC 約等於 2 個 FLOPs (浮點運算)
    # GFLOPs = (MACs * 2) / 1,000,000,000
    
    gflops = (macs * 2) / 1e9
    
    # 參數數量 (Params)
    # GParams = Params / 1,000,000,000 (十億)
    # MParams = Params / 1,000,000 (百萬)
    m_params = params / 1e6

    print(f"--- 分析結果 ({MODEL_PATH}) ---")
    print(f"輸入影像尺寸: {INPUT_SIZE}x{INPUT_SIZE}")
    print(f"總參數數量 (Params): {params:,.0f} (約 {m_params:.2f} M)")
    print(f"總乘積累加運算 (MACs): {macs:,.0f} (G-MACs: {macs/1e9:.2f})")
    print("\n")
    print("--- 理論複雜度 (GFLOPs) ---")
    print("GFLOPs (Giga-FLOPs) 的計算公式為 (MACs * 2) / 1e9")
    print(f"理論 GFLOPs: {gflops:.2f}")
    
    print("\n")
    print(f"您可以將 {gflops:.2f} 這個數值填入您的「表 4.1」中。")

except Exception as e:
    print(f"使用 thop 進行分析時發生錯誤: {e}")
    print("這通常發生在模型架構包含 thop 不支援的自定義操作。")
    print("請確保您的 ultralytics 和 thop 套件均為最新版本。")

logging.getLogger().setLevel(logging.INFO) # 恢復日誌級別