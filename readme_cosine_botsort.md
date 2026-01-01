# BoT-SORT 追蹤結果的 Re-ID 特徵餘弦距離分析

本文件說明如何使用 `run_cosine_analysis_botsort.ps1` 腳本，對 `run_realtime_track_analyze_dashboard_botsort.ps1` 產生的追蹤結果進行 Re-ID 特徵的餘弦距離分析。

## 功能

- **特徵提取**：從 BoT-SORT 追蹤過程中儲存的 `crops` 影像中，提取每個追蹤物件的 Re-ID 特徵向量。
- **相似度計算**：
  - **類內相似度 (Intra-class)**：計算同一類別下，不同追蹤 ID 之間的平均餘弦距離。這有助於評估 Re-ID 模型區分不同個體的能力。
  - **ID 內相似度 (Intra-ID)**：計算同一個追蹤 ID 在不同時間點（影格）的特徵之間的平均餘弦距離。這反映了模型對於同一個體的特徵一致性。
- **結果產出**：
  - 生成包含詳細距離數據的 CSV 檔案。
  - 繪製餘弦距離分佈的直方圖，以視覺化方式呈現分析結果。

## 使用步驟

### 1. 產生追蹤結果 (包含 Crops)

在進行餘弦分析之前，您必須先執行一次追蹤來產生必要的資料。**請確保您使用的是最新版本的 `run_realtime_track_analyze_dashboard_botsort.ps1`，因為它現在會自動儲存 Re-ID 所需的 `crops` 影像。**

如果您之前執行的追蹤沒有產生 `crops` 目錄，請刪除舊的 `runs/predict/track_*` 目錄，然後重新執行一次追蹤。

請執行 `run_realtime_track_analyze_dashboard_botsort.ps1` 腳本：
```powershell
.\run_realtime_track_analyze_dashboard_botsort.ps1 -VIDEO "path/to/your/video.mp4"
```

### 2. 執行餘弦距離分析

追蹤完成後，執行 `run_cosine_analysis_botsort.ps1` 腳本來開始分析。

```powershell
.\run_cosine_analysis_botsort.ps1
```

腳本會自動尋找 `runs/predict` 下最新的 `track_*` 目錄作為分析來源。

### 3. 查看結果

分析完成後，所有的報告和圖表將會儲存在一個新的 `runs/analyze/cosine_*` 目錄中。

- `intra_class_cosine_distances.csv`：包含類內不同 ID 間的餘弦距離數據。
- `cosine_distance_distribution.png`：類內餘弦距離的分佈直方圖。

## 參數說明

- `-TRACK_DIR` (可選): 手動指定要分析的追蹤結果目錄路徑。如果希望分析某個特定的歷史追蹤結果，可以使用此參數。

  **範例**：
  ```powershell
  .\run_cosine_analysis_botsort.ps1 -TRACK_DIR "runs\predict\track_20251013_200000"
  ```

- `-MODEL` (可選): 指定用於特徵提取的 Re-ID 模型。預設為 `best.pt`。

## 檔案結構與執行方式說明

- `run_cosine_analysis_botsort.ps1`: **這是您應該執行的主要腳本**。它負責準備環境、尋找最新的追蹤結果，並呼叫 Python 腳本進行計算。
- `calculate_cosine_similarity_botsort.py`: 這是執行實際計算的 Python 程式碼。**請不要直接執行此 Python 腳本**，因为它需要由 PowerShell 腳本提供必要的路徑參數才能正常工作。
