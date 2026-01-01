<#
一鍵：使用 best.pt 進行影片偵測＋追蹤（ByteTrack/BoT-SORT），
並自動生成統計報表（CSV/JSON/PNG）＋啟動 Flask 儀表板（Plotly 互動圖表）。
保持「純 PyTorch」版本，不含 TensorRT。
#>

param(
  [string]$MODEL = "",
  [string]$VIDEO = "",
  [string]$TRACKER = "bytetrack.yaml",
  [int]$PORT = 5050
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$BASE   = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RUNS   = Join-Path $BASE "runs"
$ANALYZE= Join-Path $RUNS "analyze"

$YOLO_EXE = Join-Path $BASE "yolo12_env\Scripts\yolo.exe"
$PYTHON   = Join-Path $BASE "yolo12_env\Scripts\python.exe"
$USE_PY = $false
if (-not (Test-Path $YOLO_EXE)) {
  if (-not (Test-Path $PYTHON)) { Write-Host "ERROR: yolo.exe and python.exe not found. Ensure yolo12_env is created."; exit 1 }
  $USE_PY = $true
  Write-Host "INFO: yolo.exe not found, using python -m ultralytics instead."
}

if (-not $MODEL -or $MODEL -eq "") { $MODEL = Join-Path $RUNS "train\dawn_semi_round3\weights\best.pt" }
if (-not $VIDEO -or $VIDEO -eq "") { $VIDEO = Join-Path $BASE "wildlife.mp4" }

if (-not (Test-Path $MODEL)) { Write-Host "ERROR: Model does not exist: $MODEL"; exit 1 }
if (-not (Test-Path $VIDEO)) { Write-Host "ERROR: Video does not exist: $VIDEO"; exit 1 }

$ts = (Get-Date).ToString("yyyyMMdd_HHmmss")
$SESSION    = "track_$ts"
$SESSION_DIR= Join-Path $ANALYZE $SESSION
New-Item -ItemType Directory -Force -Path $RUNS, $ANALYZE, $SESSION_DIR | Out-Null

Write-Host "`n[1/4] Tracking inference: $TRACKER ..."
if ($USE_PY) {
  $argumentList = "-m ultralytics track model=`"$MODEL`" source=`"$VIDEO`" tracker=`"$TRACKER`" imgsz=1280 device=0 conf=0.45 iou=0.6 save=True save_txt=True save_conf=True project=`"$RUNS\predict`" name=`"$SESSION`""
  Start-Process -FilePath $PYTHON -ArgumentList $argumentList -Wait -NoNewWindow
} else {
  $argumentList = "track model=`"$MODEL`" source=`"$VIDEO`" tracker=`"$TRACKER`" imgsz=1280 device=0 conf=0.45 iou=0.6 save=True save_txt=True save_conf=True project=`"$RUNS\predict`" name=`"$SESSION`""
  Start-Process -FilePath $YOLO_EXE -ArgumentList $argumentList -Wait -NoNewWindow
}

# Find the actual prediction directory created by YOLO, which may have a numeric suffix.
$actualPredDir = Get-ChildItem -Path (Join-Path $RUNS "predict") -Directory | `
                 Where-Object { $_.Name -like "$SESSION*" } | `
                 Sort-Object CreationTime -Descending | `
                 Select-Object -First 1

if (-not $actualPredDir) { Write-Host "ERROR: Could not find prediction output directory for session $SESSION"; exit 1 }

$PRED_DIR = $actualPredDir.FullName
$LABEL_DIR = Join-Path $PRED_DIR "labels"
if (-not (Test-Path $LABEL_DIR)) { Write-Host "ERROR: No labels found after inference: $LABEL_DIR"; exit 1 }

$DATASET_CAND = @(
  (Join-Path $BASE "datasets\wildlife\wildlife.yaml"),
  (Join-Path $BASE "dataset\wildlife\wildlife.yaml")
)
$YAML = $null
foreach ($c in $DATASET_CAND) { if (Test-Path $c) { $YAML = $c; break } }

$ANALYZER_PY = Join-Path $SESSION_DIR "analyze_and_serve.py"
$ANALYZER_TEMPLATE_PY = Join-Path $BASE "analyze_and_serve_template.py"

# Read the raw Python script template from the file
$rawPythonScriptTemplate = Get-Content -Path $ANALYZER_TEMPLATE_PY -Raw

# Perform replacements to inject PowerShell variables
$pythonScriptContent = $rawPythonScriptTemplate -replace "__LABEL_DIR__", $LABEL_DIR `
                                                -replace "__VIDEO__", $VIDEO `
                                                -replace "__SESSION_DIR__", $SESSION_DIR `
                                                -replace "__YAML__", $YAML

# Write the constructed Python script content to the final file
$pythonScriptContent | Set-Content -Path $ANALYZER_PY -Encoding UTF8

Write-Host "`n[2/4] Checking/installing Python dependencies (pandas, matplotlib, pyyaml, opencv-python, flask, plotly)..."
& $PYTHON -m pip install --quiet --upgrade pip wheel setuptools
& $PYTHON -m pip install --quiet pandas matplotlib pyyaml opencv-python flask plotly

Write-Host "[3/4] Generating statistics and starting dashboard..."
& $PYTHON $ANALYZER_PY --port $PORT | Out-Null

$URL = "http://127.0.0.1:$PORT/"
Write-Host "-> Dashboard URL: $URL"
try { Start-Process $URL | Out-Null } catch { Write-Host "Please open manually: $URL" }

Write-Host "`n[4/4] Done."
Write-Host ("Inference and tracking output: " + $PRED_DIR)
Write-Host ("Analysis report output:   " + $SESSION_DIR)
