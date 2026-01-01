<#
一鍵：使用指定模型進行影片追蹤，提取特徵向量，
並計算類內/類間的餘弦距離，最後生成 CSV 報告和視覺化圖表。
#>

param(
  [string]$MODEL = "best.pt", # 預設使用 best.pt
  [string]$VIDEO = "video\monkey_video_1.mp4", # 預設使用 video\wildlife.mp4
  [string]$YAML = "datasets\wildlife\wildlife.yaml" # 預設使用 datasets\wildlife\wildlife.yaml
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
if (-not (Test-Path $YAML)) { Write-Host "ERROR: YAML file does not exist: $YAML"; exit 1 }

$ts = (Get-Date).ToString("yyyyMMdd_HHmmss")
$SESSION    = "cosine_$ts"
$SESSION_DIR= Join-Path $ANALYZE $SESSION
New-Item -ItemType Directory -Force -Path $RUNS, $ANALYZE, $SESSION_DIR | Out-Null

Write-Host "`n[1/3] Running tracking, feature extraction, and analysis..."

# Define the path to the Python analysis script
$ANALYSIS_SCRIPT_PATH = Join-Path $BASE "calculate_cosine_similarity_bytetrack.py"
if (-not (Test-Path $ANALYSIS_SCRIPT_PATH)) {
    Write-Host "ERROR: Analysis script not found at '$ANALYSIS_SCRIPT_PATH'."
    exit 1
}

# Execute the Python script directly
# The script handles tracking, feature extraction, and analysis internally.
& $PYTHON $ANALYSIS_SCRIPT_PATH --model "$MODEL" --video "$VIDEO" --yaml "$YAML" --output_dir "$SESSION_DIR"

Write-Host "`n[2/3] Python script executed."

Write-Host "`n[3/3] Done."
Write-Host ("Analysis report output: " + $SESSION_DIR)
