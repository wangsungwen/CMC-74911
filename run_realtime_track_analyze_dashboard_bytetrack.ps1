<#
一鍵：使用 best.pt 進行影片偵測＋追蹤（ByteTrack/BoT-SORT），
並自動生成統計報表（CSV/JSON/PNG）＋啟動 Flask 儀表板（Plotly 互動圖表）。
支援：本地影片檔案、RTSP/HTTP 串流、網路影片 URL 等作為輸入 source。
推論時可即時彈出視窗觀看追蹤結果，統計分析在追蹤結束後生成。
#>

param(
  [string]$MODEL = "",
  [string]$VIDEO = "", # 影片來源：可以是本地檔案路徑、RTSP/HTTP URL、網路影片 URL
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
if (-not $VIDEO -or $VIDEO -eq "") { Write-Host "ERROR: Video/Stream source must be provided (e.g., -VIDEO 'path/to/video.mp4' or -VIDEO 'rtsp://...')."; exit 1 }
if (-not (Test-Path $MODEL)) { Write-Host "ERROR: Model does not exist: $MODEL"; exit 1 }

$ts = (Get-Date).ToString("yyyyMMdd_HHmmss")
$SESSION    = "track_$ts"
$PRED_DIR   = Join-Path $RUNS ("predict\$SESSION")
$SESSION_DIR= Join-Path $ANALYZE $SESSION
New-Item -ItemType Directory -Force -Path $RUNS, $ANALYZE, $PRED_DIR, $SESSION_DIR | Out-Null

Write-Host "`n[1/4] Tracking inference: $TRACKER on source: $VIDEO ..."
# Ultralytics track 支援即時推論，彈出視窗同步觀看
if ($USE_PY) {
  & $PYTHON -m ultralytics track model="$MODEL" source="$VIDEO" tracker="$TRACKER" imgsz=1280 device=0 conf=0.45 iou=0.6 save=True save_txt=True save_conf=True project="$RUNS\predict" name="$SESSION"
} else {
  & $YOLO_EXE track model="$MODEL" source="$VIDEO" tracker="$TRACKER" imgsz=1280 device=0 conf=0.45 iou=0.6 save=True save_txt=True save_conf=True project="$RUNS\predict" name="$SESSION"
}
# 對於即時串流，需手動 Ctrl+C 停止，腳本才會繼續執行

# 動態尋找 Ultralytics 實際使用的輸出目錄（因為它可能在目錄已存在時添加後綴，例如 track_xyz -> track_xyz2）
$actualPredDir = Get-ChildItem -Path (Join-Path $RUNS "predict") -Directory | Where-Object { $_.Name -like "$SESSION*" } | Sort-Object CreationTime -Descending | Select-Object -First 1
if ($null -eq $actualPredDir) {
    Write-Host "ERROR: Could not find prediction output directory for session $SESSION."
    exit 1
}
$PRED_DIR = $actualPredDir.FullName
Write-Host "INFO: Found actual prediction directory: $PRED_DIR"

$LABEL_DIR = Join-Path $PRED_DIR "labels"
if (-not (Test-Path $LABEL_DIR) -or -not (Get-ChildItem -Path $LABEL_DIR -Filter "*.txt")) {
    Write-Host "WARNING: No labels found in $LABEL_DIR. The model may not have detected any objects. Skipping analysis and dashboard."
    exit 0 # 正常退出，因為這不是一個腳本錯誤
}

# 檢查是否有本地儲存的影片檔案供分析使用 (針對 URL/串流)
$analysisVideoSource = $VIDEO
if ($VIDEO -like "http*") {
    # 嘗試尋找 Ultralytics 儲存的影片檔案
    $videoFile = Get-ChildItem -Path $PRED_DIR -Include "*.mp4", "*.avi" -Recurse | Select-Object -First 1
    if ($videoFile) {
        $analysisVideoSource = $videoFile.FullName
        Write-Host "INFO: Using downloaded video for analysis: $analysisVideoSource"
    } else {
        Write-Host "WARNING: Could not find downloaded video file in $PRED_DIR. Analysis will use the URL/Stream source for metadata."
    }
}

$DATASET_CAND = @(
  (Join-Path $BASE "datasets\wildlife\wildlife.yaml"),
  (Join-Path $BASE "dataset\wildlife\wildlife.yaml")
)
$YAML = $null
foreach ($c in $DATASET_CAND) { if (Test-Path $c) { $YAML = $c; break } }

$ANALYZER_PY = Join-Path $SESSION_DIR "analyze_and_realtime.py"
$ANALYZER_TEMPLATE_PY = Join-Path $BASE "analyze_and_realtime_template.py" 

# Read the raw Python script template from the file
$rawPythonScriptTemplate = Get-Content -Path $ANALYZER_TEMPLATE_PY -Raw

# Perform replacements to inject PowerShell variables
$pythonScriptContent = $rawPythonScriptTemplate -replace "__LABEL_DIR__", $LABEL_DIR `
                                                -replace "__VIDEO__", $analysisVideoSource `
                                                -replace "__SESSION_DIR__", $SESSION_DIR `
                                                -replace "__YAML__", $YAML

# Write the constructed Python script content to the final file
$pythonScriptContent | Set-Content -Path $ANALYZER_PY -Encoding utf8

Write-Host "`n[2/4] Checking/installing Python dependencies (pandas, matplotlib, pyyaml, opencv-python, flask, plotly)..."
& $PYTHON -m pip install --quiet --upgrade pip wheel setuptools | Out-Null
& $PYTHON -m pip install --quiet pandas matplotlib pyyaml opencv-python flask plotly | Out-Null

Write-Host "[3/4] Generating statistics and starting dashboard..."
& $PYTHON $ANALYZER_PY --port $PORT | Out-Null
# 使用 Start-Job 在背景啟動 Flask 儀表板，這樣 PowerShell 腳本可以繼續執行並顯示最終訊息
Start-Job -ScriptBlock {
    param($py, $analyzer, $p)
    & $py $analyzer --port $p
} -ArgumentList $PYTHON, $ANALYZER_PY, $PORT | Out-Null

# 等待幾秒鐘，確保 Flask 伺服器有足夠的時間啟動
Start-Sleep -Seconds 3

$URL = "http://127.0.0.1:$PORT/"
Write-Host "-> Dashboard URL: $URL"
try { Start-Process $URL | Out-Null } catch { Write-Host "Please open manually: $URL" }

Write-Host "`n[4/4] Done."
Write-Host ("Inference and tracking output: " + $PRED_DIR)
Write-Host ("Analysis report output:   " + $SESSION_DIR)
