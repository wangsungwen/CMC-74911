<#
 YOLOv12 Multi-Round Semi-Supervised Training Pipeline
 (pseudo-label -> filter -> retrain) with robust checks
#>

# Script Version: 1.1 (Fixed ForEach-Object issue)
Write-Host "Executing run_yolo12_semi_loop.ps1 - Version 1.1"

# Set console to UTF-8
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$ErrorActionPreference = "Stop"

# --- Basic Setup ---
$BASE = Split-Path -Parent $MyInvocation.MyCommand.Definition
$YOLO_FULL_PATH = Join-Path $BASE "yolo12_env\Scripts\yolo.exe"

if (-not (Test-Path $YOLO_FULL_PATH)) {
    Write-Host "ERROR: YOLO executable not found at: $YOLO_FULL_PATH"
    Write-Host "Please ensure the yolo12_env virtual environment is set up and ultralytics is installed."
    exit 1
}

# --- Dataset Path Resolution ---
$DATASET_DEFAULT = Join-Path $BASE "dataset\wildlife"
$DATASET_ALT = Join-Path $BASE "datasets\wildlife"
if (Test-Path $DATASET_ALT) {
    $DATASET = $DATASET_ALT
}
elseif (Test-Path $DATASET_DEFAULT) {
    $DATASET = $DATASET_DEFAULT
}
else {
    Write-Host "ERROR: Dataset folder not found."
    Write-Host "Please ensure that either '$DATASET_DEFAULT' or '$DATASET_ALT' exists."
    exit 1
}

# --- Paths & Config ---
$RUNS = Join-Path $BASE "runs"
$MODEL_PATH = Join-Path $RUNS "train\dawn_supervised_v1\weights\best.pt"
$MODEL = if (Test-Path $MODEL_PATH) { $MODEL_PATH } else { "yolo12n.pt" }
if ($MODEL -eq "yolo12n.pt") {
    Write-Host "WARNING: Pre-trained model not found at $MODEL_PATH. Using 'yolo12n.pt' as a fallback."
}

$DEVICE = "0"
$CONF_PREDICT = 0.50     # threshold used when generating pseudo labels
$CONF_FILTER_R1 = 0.80   # round-1 keep threshold
$CONF_FILTER_R2 = 0.75   # from round-2
$IOU_PREDICT = 0.60
$EPOCHS = 100
$BATCH  = 2
$IMGSZ  = 416
$ROUNDS = 3              # total rounds

# --- I/O helpers ---
function Get-FirstExistingImagePath([string]$dir, [string]$stem) {
    $cands = @(
        (Join-Path $dir ($stem + ".jpg")),
        (Join-Path $dir ($stem + ".jpeg")),
        (Join-Path $dir ($stem + ".png")),
        (Join-Path $dir ($stem + ".bmp"))
    )
    foreach ($p in $cands) { if (Test-Path $p) { return $p } }
    return $null
}

# Ensure folders
$UNLABELED_IMG_DIR = Join-Path $DATASET "images\unlabeled"
$DEST_LABELS = Join-Path $DATASET "labels\train_pseudo"
$DEST_IMAGES = Join-Path $DATASET "images\train_pseudo"
New-Item -ItemType Directory -Force -Path $RUNS, $UNLABELED_IMG_DIR, $DEST_LABELS, $DEST_IMAGES | Out-Null

# --- Write/Update dataset YAML (absolute path for robustness) ---
$absPath = ($DATASET -replace '\\', '/')
$yamlPath = Join-Path $DATASET "wildlife.yaml"
$yamlContent = @"
# YOLOv12 dataset
path: $absPath
train: [images/train, images/train_pseudo]
test: images/test
val: images/val
names:
  0: monkey
"@
$yamlContent | Set-Content -Path $yamlPath -Encoding utf8

# --- Check unlabeled availability ---
$imgFiles = Get-ChildItem -Path $UNLABELED_IMG_DIR -Include *.jpg, *.jpeg, *.png, *.bmp -File -Recurse
if ($imgFiles.Count -eq 0) {
    Write-Host "WARNING: No images found in '$UNLABELED_IMG_DIR'. The loop will still run but no new pseudo-labels will be created."
}

# ---------------- ROUNDS LOOP ----------------
for ($r = 1; $r -le $ROUNDS; $r++) {

    $teacher = if ($r -eq 1) { $MODEL } else { Join-Path $RUNS "train\dawn_semi_round$($r-1)\weights\best.pt" }
    if (-not (Test-Path $teacher)) {
        Write-Host "ERROR: Teacher weights not found: $teacher"
        exit 1
    }

    $CONF_FILTER = if ($r -ge 2) { $CONF_FILTER_R2 } else { $CONF_FILTER_R1 }

    # Step 1: predict -> pseudo labels
    Write-Host ""
    Write-Host "========== [Round $r/ $ROUNDS] Pseudo-labeling =========="
    if ($imgFiles.Count -gt 0) {
        & $YOLO_FULL_PATH predict model=$teacher source=$UNLABELED_IMG_DIR imgsz=$IMGSZ conf=$CONF_PREDICT iou=$IOU_PREDICT save=True save_txt=True save_conf=True project="$RUNS\pseudo" name=("round{0}" -f $r) device=$DEVICE
    } else {
        Write-Host "Skip pseudo-labeling (no unlabeled images)."
    }

    # Step 2: filter pseudo txts and copy images
    Write-Host "========== [Round $r] Filtering (conf >= $CONF_FILTER) =========="
    $SRC_LABELS = Join-Path $RUNS ("pseudo\round{0}\labels" -f $r)
    New-Item -ItemType Directory -Force -Path $DEST_LABELS, $DEST_IMAGES | Out-Null

    $kept = 0
    if (Test-Path $SRC_LABELS) {
        # Changed from ForEach-Object to a foreach loop for potentially better error handling.
        # Also added a check to ensure $txtFiles is not null or empty.
        $txtFiles = Get-ChildItem -Path $SRC_LABELS -Filter *.txt -Recurse
        if ($txtFiles) {
            foreach ($file in $txtFiles) {
                $filteredLines = Get-Content $file.FullName | Where-Object {
                    $parts = $_ -split '\s+'
                    $parts.Count -ge 6 -and [double]$parts[5] -ge $CONF_FILTER
                }
                if ($filteredLines) {
                    # write 5-col YOLO lines
                    $outputContent = $filteredLines | ForEach-Object { ($_.Split(' '))[0..4] -join ' ' }
                    $outputPath = Join-Path $DEST_LABELS $file.Name
                    $outputContent | Set-Content -Path $outputPath -Encoding utf8

                    # copy matched image
                    $imgPath = Get-FirstExistingImagePath -dir $UNLABELED_IMG_DIR -stem $file.BaseName
                    if ($imgPath) { Copy-Item -Path $imgPath -Destination $DEST_IMAGES -Force }
                    $kept++
                }
            }
        } else {
            Write-Host "No .txt files found in $SRC_LABELS."
        }
    } else {
        Write-Host "WARNING: No source pseudo-labels found at: $SRC_LABELS"
    }
    Write-Host ("Kept pseudo-labeled samples: {0}" -f $kept)

    # Step 3: retrain student
    Write-Host "========== [Round $r] Retraining student =========="
    $STUDENT_NAME = "dawn_semi_round$($r)"
    & $YOLO_FULL_PATH train model=$teacher data=$yamlPath imgsz=$IMGSZ epochs=$EPOCHS batch=$BATCH device=$DEVICE project="$RUNS\train" name=$STUDENT_NAME

    $BEST_PT = Join-Path $RUNS "train\$STUDENT_NAME\weights\best.pt"
    if (-not (Test-Path $BEST_PT)) {
        Write-Host "ERROR: best.pt not found: $BEST_PT"
        exit 1
    }

    Write-Host ("✅ Round {0} finished. New teacher: {1}" -f $r, $BEST_PT)
}

Write-Host ""
Write-Host "🎯 ALL ROUNDS FINISHED."
