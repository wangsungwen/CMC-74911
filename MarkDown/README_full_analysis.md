# 端對端整合性分析流程 (BoT-SORT)

本文件說明如何使用 `start_full_analysis.ps1` 腳本，執行一個從影片追蹤、特徵提取到餘弦距離分析的完整流程。

## 功能概覽

這個新的整合性流程取代了先前分離的多個步驟，提供了一個更簡單、更直接的分析方法。其核心功能包括：

1.  **物件追蹤**：使用 BoT-SORT 追蹤器處理指定的影片檔案，獲取物件的軌跡。
2.  **特徵提取**：對追蹤到的每一個物件，即時提取其 Re-ID 特徵向量。
3.  **代表性特徵計算**：將同一個物件在其軌跡中出現的所有特徵向量平均，形成一個最能代表該物件的特徵。
4.  **餘弦距離分析**：
    - **類內 (Intra-class)**：計算同一類別中，不同物件之間的餘弦距離。
    - **類間 (Inter-class)**：計算不同類別之間，所有物件的餘弦距離。
5.  **自動化產出**：
    - 兩個詳細的 CSV 報告 (`intra_class...` 和 `inter_class...`)。
    - 一張結合了兩種距離分佈的視覺化圖表。

## 使用步驟

### 1. 準備工作

- 確保 `yolo12_env` Python 虛擬環境已建立。
- 確保您的影片檔案（例如 `wildlife.mp4`）已放置在專案的根目錄下。
- 確保您有兩個模型檔案：
  - 一個用於**物件追蹤**的偵測模型 (例如 `best.pt`)。
  - 一個用於**特徵提取**的模型。**強烈建議**使用分類模型 (例如 `yolo12n-cls.pt`) 以獲得最佳的 Re-ID 效果。

### 2. 執行分析

打開 PowerShell 終端機，並執行以下指令：

```powershell
.\start_full_analysis.ps1
```

腳本會使用預設參數開始執行。

### 3. 查看結果

分析完成後，所有結果將會被儲存在一個名為 `full_analysis_results` 的新目錄中。

- `intra_class_cosine_distances_botsort.csv`: 類內距離數據。
- `inter_class_cosine_distances_botsort.csv`: 類間距離數據。
- `cosine_distance_distribution_botsort.png`: 結合兩種距離分佈的直方圖。

## 自訂參數

您可以直接在 PowerShell 執行時傳遞參數來修改預設行為：

- `-VIDEO`: 指定要分析的影片檔案 (預設: `wildlife.mp4`)。
- `-TRACK_MODEL`: 指定用於物件追蹤的**偵測模型** (預設: `best.pt`)。
- `-EMBED_MODEL`: 指定用於特徵提取的**分類模型** (預設: `yolo12n-cls.pt`)。
- `-OUTPUT`: 指定儲存結果的目錄名稱 (預設: `full_analysis_results`)。

**範例**：
```powershell
.\start_full_analysis.ps1 -VIDEO "monkey_video_1.mp4" -TRACK_MODEL "yolo12n.pt" -EMBED_MODEL "yolov8n-cls.pt" -OUTPUT "monkey_analysis"
```

## 特徵提取說明

本腳本的特徵提取邏輯如下：
1.  **優先使用分類模型**：如果提供的 `-EMBED_MODEL` 是一個**分類模型**，腳本會提取其最後一層的機率輸出 (`.probs`) 作為高品質的物件外觀特徵。
2.  **回退使用偵測模型**：如果您為 `-EMBED_MODEL` 提供了一個**偵測模型**（例如，與 `-TRACK_MODEL` 相同），腳本會發出警告，並回退到使用在裁切圖中偵測到的物件的**邊界框位置與大小 (`.boxes.xywhn`)** 作為特徵。這是一個功能性的替代方案，但其 Re-ID 效果不如真正的外觀特徵。

## 檔案結構

- `start_full_analysis.ps1`: **您需要執行的主要啟動腳本**。
- `run_full_analysis_botsort.py`: 包含了所有核心分析邏輯的 Python 程式碼。
