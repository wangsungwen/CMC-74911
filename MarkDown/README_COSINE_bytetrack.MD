# YOLOv12 餘弦相似度分析說明

本文件說明如何執行 YOLOv12 物件追蹤的後續分析，以計算偵測物件之間的餘弦距離，並將結果視覺化。

**重要提示：** 請直接執行 `calculate_cosine_similarity.py` 腳本來進行分析。
專案中另一個名為 `run_cosine_analysis.ps1` 的 PowerShell 腳本是一個早期的、已棄用的嘗試，由於其設計上的根本缺陷（無法提取特徵向量），請勿使用。

---

## 目的

此分析旨在評估模型提取特徵（Embeddings）的品質。理想情況下：
- **類內距離 (Intra-class Distance)**：同一個類別的不同物件之間，其特徵向量的餘弦距離應該要很小（彼此相似）。
- **類間距離 (Inter-class Distance)**：不同類別的物件之間，其特徵向量的餘弦距離應該要很大（彼此不相似）。

透過視覺化這兩種距離的分佈，我們可以直觀地判斷模型的特徵區分能力。

## 執行步驟

分析過程透過一個獨立的 Python 腳本 `calculate_cosine_similarity.py` 來完成，其內部執行流程如下：

1.  **載入模型**：載入指定的 `.pt` 模型檔案。
2.  **第一階段 (物件追蹤)**：
    - 逐幀讀取指定的影片檔案。
    - 使用 `YOLO.track()` 方法進行物件追蹤，以獲取每個物件的邊界框 (BBox) 和追蹤 ID (Track ID)。
    - 將偵測到的每個物件從原始畫面中裁切出來。
3.  **第二階段 (特徵提取)**：
    - 針對上一步裁切出的所有物件圖片。
    - 使用 `YOLO.embed()` 方法為每一個物件圖片提取其特徵向量 (Embedding)。
4.  **計算代表性特徵**：
    - 對於同一個追蹤 ID 的所有特徵向量，計算其平均值，以作為該物件的代表性特徵。
5.  **計算餘弦距離**：
    - 計算所有**類內**物件配對的餘弦距離。
    - 計算所有**類間**物件配對的餘弦距離。
6.  **產出結果**：
    - 將計算出的距離分別儲存為 `intra_class_cosine_distances_bytetrack.csv`、`inter_class_cosine_distances_bytetrack.csv` 和 `intra_id_cosine_similarities_bytetrack.csv`。
    - 繪製一張結合了類內與類間距離分佈的圖表 `cosine_distance_distribution_bytetrack.png`。

## 執行指令

請在專案根目錄下，使用 `yolo12_env` 虛擬環境來執行以下指令：

```bash
yolo12_env\Scripts\python.exe calculate_cosine_similarity_bytetrack.py
```

### 可選參數

您也可以自訂模型、影片來源、資料集設定檔與輸出目錄：

```bash
yolo12_env\Scripts\python.exe calculate_cosine_similarity.py --model "path/to/your/model.pt" --video "path/to/your/video.mp4" --yaml "path/to/your/dataset.yaml" --output_dir "path/to/your/output"
```

- `--model`: 指定要使用的模型檔案路徑 (預設: `best.pt`)。
- `--video`: 指定要分析的影片檔案路徑 (預設: `wildlife.mp4`)。
- `--yaml`: 指定資料集的 YAML 設定檔路徑 (預設: `datasets/wildlife/wildlife.yaml`)。
- `--output_dir`: 指定儲存結果的目錄 (預設: `runs/analyze/cosine_analysis`)。
