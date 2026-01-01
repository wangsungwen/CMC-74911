<#
一鍵啟動完整的 BoT-SORT 追蹤、特徵提取與餘弦距離分析流程。
#>

param(
  [string]$VIDEO = "wildlife.mp4", # 要分析的影片
  [string]$TRACK_MODEL = "best.pt",      # 使用的追蹤模型
  [string]$EMBED_MODEL = "yolo12n-cls.pt", # 使用的 Embedding 模型
  [string]$OUTPUT = "full_analysis_results" # 輸出目錄
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$BASE = Split-Path -Parent $MyInvocation.MyCommand.Definition
$PYTHON = Join-Path $BASE "yolo12_env\Scripts\python.exe"
$ANALYZER_PY = Join-Path $BASE "run_full_analysis_botsort.py"

if (-not (Test-Path $PYTHON)) { Write-Host "ERROR: python.exe not found."; exit 1 }
if (-not (Test-Path $ANALYZER_PY)) { Write-Host "ERROR: Analyzer script not found: $ANALYZER_PY"; exit 1 }
if (-not (Test-Path $VIDEO)) { Write-Host "ERROR: Video file not found: $VIDEO"; exit 1 }
if (-not (Test-Path $TRACK_MODEL)) { Write-Host "ERROR: Tracking model file not found: $TRACK_MODEL"; exit 1 }
if (-not (Test-Path $EMBED_MODEL)) { Write-Host "ERROR: Embedding model file not found: $EMBED_MODEL"; exit 1 }

Write-Host "Starting full analysis..."
Write-Host " - Video: $VIDEO"
Write-Host " - Tracking Model: $TRACK_MODEL"
Write-Host " - Embedding Model: $EMBED_MODEL"
Write-Host " - Output: $OUTPUT"

& $PYTHON $ANALYZER_PY --video "$VIDEO" --track_model "$TRACK_MODEL" --embed_model "$EMBED_MODEL" --output "$OUTPUT"

Write-Host "`nAnalysis complete. Results are in the '$OUTPUT' directory."
