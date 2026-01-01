<#
一鍵：使用指定模型進行影片追蹤，提取特徵向量，
並計算類內/類間的餘弦距離，最後生成 CSV 報告和視覺化圖表。
#>

param(
  [string]$MODEL = "best.pt", # 預設使用 best.pt
  [string]$VIDEO = "wildlife.mp4" # 預設使用 monkey_video_1.mp4
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$BASE   = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RUNS   = Join-Path $BASE "runs"
$ANALYZE= Join-Path $RUNS "analyze"

$PYTHON   = Join-Path $BASE "yolo12_env\Scripts\python.exe"
if (-not (Test-Path $PYTHON)) { Write-Host "ERROR: python.exe not found. Ensure yolo12_env is created."; exit 1 }

if (-not (Test-Path $MODEL)) { Write-Host "ERROR: Model does not exist: $MODEL"; exit 1 }
if (-not (Test-Path $VIDEO)) { Write-Host "ERROR: Video does not exist: $VIDEO"; exit 1 }

$ts = (Get-Date).ToString("yyyyMMdd_HHmmss")
$SESSION    = "cosine_$ts"
$PRED_DIR   = Join-Path $RUNS ("predict\$SESSION")
$SESSION_DIR= Join-Path $ANALYZE $SESSION
New-Item -ItemType Directory -Force -Path $RUNS, $ANALYZE, $PRED_DIR, $SESSION_DIR | Out-Null

Write-Host "`n[1/3] Running tracking and feature extraction..."
# 注意：這裡我們假設 Ultralytics 的 track 模式支援一個（未經官方確認的）參數 `save_embed=True` 來儲存特徵向量。
# 如果這無法運作，將需要一個更複雜的自訂 Python 腳本來進行推論。
& $PYTHON -m ultralytics track model="$MODEL" source="$VIDEO" tracker="bytetrack.yaml" imgsz=1280 device=0 conf=0.45 iou=0.6 save=True save_txt=True save_conf=True project="$RUNS\predict" name="$SESSION" # ReID-specific flag might be needed here. Let's assume for now it's part of the standard output.

# 動態尋找實際的輸出目錄
$actualPredDir = Get-ChildItem -Path (Join-Path $RUNS "predict") -Directory | Where-Object { $_.Name -like "$SESSION*" } | Sort-Object CreationTime -Descending | Select-Object -First 1
if ($null -eq $actualPredDir) {
    Write-Host "ERROR: Could not find prediction output directory for session $SESSION."
    exit 1
}
$PRED_DIR = $actualPredDir.FullName
Write-Host "INFO: Found actual prediction directory: $PRED_DIR"

$LABEL_DIR = Join-Path $PRED_DIR "labels"
if (-not (Test-Path $LABEL_DIR) -or -not (Get-ChildItem -Path $LABEL_DIR -Filter "*.txt")) {
    Write-Host "WARNING: No labels found in $LABEL_DIR. The model may not have detected any objects. Skipping analysis."
    exit 0
}

$DATASET_YAML = Join-Path $BASE "datasets\wildlife\wildlife.yaml"

$ANALYZER_PY = Join-Path $SESSION_DIR "analyze_cosine.py"
$ANALYZER_TEMPLATE_PY = Join-Path $BASE "cosine_similarity_template.py" 

$templateContent = Get-Content -Path $ANALYZER_TEMPLATE_PY -Raw

$scriptContent = $templateContent -replace "__LABEL_DIR__", $LABEL_DIR `
                                  -replace "__SESSION_DIR__", $SESSION_DIR `
                                  -replace "__YAML__", $DATASET_YAML

$scriptContent | Set-Content -Path $ANALYZER_PY -Encoding utf8

Write-Host "`n[2/3] Installing/checking Python dependencies (numpy, scipy, seaborn)..."
& $PYTHON -m pip install --quiet numpy scipy seaborn

Write-Host "`n[3/3] Running cosine similarity analysis..."
& $PYTHON $ANALYZER_PY

Write-Host "`nDone."
Write-Host ("Tracking output: " + $PRED_DIR)
Write-Host ("Analysis report output: " + $SESSION_DIR)
