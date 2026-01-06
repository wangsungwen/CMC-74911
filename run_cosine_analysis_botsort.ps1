<#
針對指定的追蹤結果（預設為 BoT-SORT），進行 Re-ID 特徵的餘弦距離分析。
功能：
1. 提取每個追蹤 ID 在其生命週期中出現的所有 Re-ID 特徵向量。
2. 計算同一類別內，不同追蹤 ID 之間的平均餘弦距離。
3. 計算同一追蹤 ID 內部，不同幀之間的平均餘弦距離（自我相似度）。
4. 生成類內/ID內餘弦距離的 CSV 報告。
5. 繪製餘弦距離分佈的直方圖。
#>

param(
  [string]$TRACK_DIR = "", # 指定 predict/track_* 的路徑
  [string]$MODEL = "best.pt" # Re-ID 模型
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$BASE = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RUNS = Join-Path $BASE "runs"
$ANALYZE = Join-Path $RUNS "analyze"

$PYTHON = Join-Path $BASE "yolo12_env\Scripts\python.exe"
if (-not (Test-Path $PYTHON)) { Write-Host "ERROR: python.exe not found."; exit 1 }

# 如果未指定 -TRACK_DIR，自動尋找最新的 BoT-SORT 追蹤結果目錄
if (-not $TRACK_DIR) {
    Write-Host "INFO: -TRACK_DIR not specified, searching for the latest BoT-SORT tracking result..."
    $latestTrack = Get-ChildItem -Path (Join-Path $RUNS "predict") -Directory -Filter "track_*" |
                   Where-Object {
                       # 簡易判斷是否為 BoT-SORT，可根據實際情況調整
                       # 這裡假設 BoT-SORT 的目錄下會有特定的標記，或者我們可以信任使用者在正確的流程後執行
                       # 為簡化，我們先選擇最新的 track 目錄
                       $_.Name -match ".*"
                   } |
                   Sort-Object CreationTime -Descending |
                   Select-Object -First 1

    if ($latestTrack) {
        $TRACK_DIR = $latestTrack.FullName
        Write-Host "INFO: Found latest tracking directory: $TRACK_DIR"
    } else {
        Write-Host "ERROR: No tracking directory found in runs/predict."
        exit 1
    }
}

if (-not (Test-Path $TRACK_DIR)) { Write-Host "ERROR: Track directory not found: $TRACK_DIR"; exit 1 }

$ts = (Get-Date).ToString("yyyyMMdd_HHmmss")
$SESSION = "cosine_$ts"
$SESSION_DIR = Join-Path $ANALYZE $SESSION
New-Item -ItemType Directory -Force -Path $SESSION_DIR | Out-Null

$ANALYZER_PY = Join-Path $BASE "calculate_cosine_similarity_botsort.py"
if (-not (Test-Path $ANALYZER_PY)) { Write-Host "ERROR: Analyzer script not found: $ANALYZER_PY"; exit 1 }

Write-Host "`n[1/3] Checking/installing Python dependencies..."
& $PYTHON -m pip install --quiet --upgrade pip | Out-Null
& $PYTHON -m pip install --quiet torch torchvision torchaudio ultralytics numpy pandas matplotlib seaborn scikit-learn | Out-Null

Write-Host "[2/3] Running Cosine Similarity Analysis..."
& $PYTHON $ANALYZER_PY --track_dir "$TRACK_DIR" --model "$MODEL" --output_dir "$SESSION_DIR"

Write-Host "`n[3/3] Done."
Write-Host "Analysis report output: $SESSION_DIR"
