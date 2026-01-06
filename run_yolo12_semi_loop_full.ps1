<#
 YOLOv12 Multi-Round Semi-Supervised Pipeline (FULL)
 - pseudo-labeling
 - filtering
 - retraining
 - validation -> metrics.json
 - video prediction (optional)
 - ONNX export per round
#>

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
$EXPORTS = Join-Path $BASE "exports"
$MODEL_PATH = Join-Path $RUNS "train\dawn_supervised_v1\weights\best.pt"
$MODEL = if (Test-Path $MODEL_PATH) { $MODEL_PATH } else { "yolo12n.pt" }
if ($MODEL -eq "yolo12n.pt") {
    Write-Host "WARNING: Pre-trained model not found at $MODEL_PATH. Using 'yolo12n.pt' as a fallback."
}

$DEVICE = "0"
$CONF_PREDICT = 0.50
$CONF_FILTER_R1 = 0.80
$CONF_FILTER_R2 = 0.75
$IOU_PREDICT = 0.60
$EPOCHS = 80
$BATCH  = 2
$IMGSZ  = 416
$ROUNDS = 3

# Optional demo video (file or folder). If not found, prediction will be skipped.
$EVAL_VIDEO = Join-Path $BASE "wildlife.mp4"

# --- helpers ---
function Get-FirstExistingImagePath([string]$dir, [string]$stem) {
    $cands = @(
        Join-Path $dir ($stem + ".jpg"),
        Join-Path $dir ($stem + ".jpeg"),
        Join-Path $dir ($stem + ".png"),
        Join-Path $dir ($stem + ".bmp")
    )
    foreach ($p in $cands) { if (Test-Path $p) { return $p } }
    return $null
}

# ensure dirs
$UNLABELED_IMG_DIR = Join-Path $DATASET "images\unlabeled"
$DEST_LABELS = Join-Path $DATASET "labels\train_pseudo"
$DEST_IMAGES = Join-Path $DATASET "images\train_pseudo"
New-Item -ItemType Directory -Force -Path $RUNS, $EXPORTS, $UNLABELED_IMG_DIR, $DEST_LABELS, $DEST_IMAGES | Out-Null

# dataset yaml (abs path)
$absPath = ($DATASET -replace '\\', '/')
$yamlPath = Join-Path $DATASET "wildlife.yaml"
@"
# YOLOv12 dataset
path: $absPath
train: [images/train, images/train_pseudo]
test: images/test
val: images/val
names:
  0: monkey
"@ | Set-Content -Path $yamlPath -Encoding utf8

$imgFiles = Get-ChildItem -Path $UNLABELED_IMG_DIR -Include *.jpg, *.jpeg, *.png, *.bmp -File -Recurse

# --------------- ROUNDS ---------------
for ($r = 1; $r -le $ROUNDS; $r++) {

    $teacher = if ($r -eq 1) { $MODEL } else { Join-Path $RUNS "train\dawn_semi_round$($r-1)\weights\best.pt" }
    if (-not (Test-Path $teacher)) {
        Write-Host "ERROR: Teacher weights not found: $teacher"
        exit 1
    }

    $CONF_FILTER = if ($r -ge 2) { $CONF_FILTER_R2 } else { $CONF_FILTER_R1 }

    # 1) pseudo
    Write-Host ""
    Write-Host "==== [Round $r/$ROUNDS] Pseudo-labeling ===="
    if ($imgFiles.Count -gt 0) {
        & $YOLO_FULL_PATH predict model=$teacher source=$UNLABELED_IMG_DIR imgsz=$IMGSZ conf=$CONF_PREDICT iou=$IOU_PREDICT save=True save_txt=True save_conf=True project="$RUNS\pseudo" name=("round{0}" -f $r) device=$DEVICE
    } else {
        Write-Host "Skip pseudo-labeling (no unlabeled images)."
    }

    # 2) filter
    Write-Host "==== [Round $r] Filtering (conf >= $CONF_FILTER) ===="
    $SRC_LABELS = Join-Path $RUNS ("pseudo\round{0}\labels" -f $r)
    New-Item -ItemType Directory -Force -Path $DEST_LABELS, $DEST_IMAGES | Out-Null

    $kept = 0
    if (Test-Path $SRC_LABELS) {
        Get-ChildItem -Path $SRC_LABELS -Filter *.txt -Recurse | ForEach-Object {
            $file = $_
            $filtered = Get-Content $file.FullName | Where-Object {
                $p = ($_ -split '\s+'); $p.Count -ge 6 -and [double]$p[5] -ge $CONF_FILTER
            }
            if ($filtered) {
                $outTxt = Join-Path $DEST_LABELS $file.Name
                $filtered | ForEach-Object { ($_.Split(' '))[0..4] -join ' ' } | Set-Content -Path $outTxt -Encoding utf8

                $imgPath = Get-FirstExistingImagePath -dir $UNLABELED_IMG_DIR -stem $file.BaseName
                if ($imgPath) { Copy-Item -Path $imgPath -Destination $DEST_IMAGES -Force }
                $kept++
            }
        }
    } else {
        Write-Host "WARNING: Pseudo labels directory not found: $SRC_LABELS"
    }
    Write-Host ("Kept pseudo-labeled samples: {0}" -f $kept)

    # 3) train
    $STUDENT_NAME = "dawn_semi_round$($r)"
    $RUN_OUT = Join-Path $RUNS "train\$STUDENT_NAME"
    Write-Host "==== [Round $r] Training -> $STUDENT_NAME ===="
    & $YOLO_FULL_PATH train model=$teacher data=$yamlPath imgsz=$IMGSZ epochs=$EPOCHS batch=$BATCH device=$DEVICE project="$RUNS\train" name=$STUDENT_NAME

    $BEST_PT = Join-Path $RUN_OUT "weights\best.pt"
    if (-not (Test-Path $BEST_PT)) {
        Write-Host "ERROR: best.pt not found: $BEST_PT"
        exit 1
    }

    # 4) val -> metrics.json
    Write-Host "==== [Round $r] Validation (metrics.json) ===="
    $VAL_PROJ = Join-Path $RUNS "val"
    $VAL_NAME = "round$($r)"
    & $YOLO_FULL_PATH val model=$BEST_PT data=$yamlPath imgsz=$IMGSZ device=$DEVICE project=$VAL_PROJ name=$VAL_NAME

    $VAL_DIR = Join-Path $VAL_PROJ $VAL_NAME
    $MET_JSON_DST = Join-Path $RUN_OUT "metrics.json"
    $RES_JSON = Join-Path $VAL_DIR "results.json"
    $RES_CSV  = Join-Path $VAL_DIR "results.csv"
    if (Test-Path $RES_JSON) {
        Copy-Item $RES_JSON $MET_JSON_DST -Force
    } elseif (Test-Path $RES_CSV) {
        (Import-Csv $RES_CSV | ConvertTo-Json -Depth 5) | Set-Content -Encoding UTF8 $MET_JSON_DST
    } else {
        Write-Host "WARNING: No results.json/csv found at $VAL_DIR"
    }

    # 5) export onnx
    Write-Host "==== [Round $r] Export ONNX ===="
    & $YOLO_FULL_PATH export model=$BEST_PT format=onnx imgsz=$IMGSZ device=$DEVICE
    $ONNX_SRC = Join-Path $RUN_OUT "weights\best.onnx"
    $ROUND_EXP = Join-Path $EXPORTS "round$($r)"
    New-Item -ItemType Directory -Force -Path $ROUND_EXP | Out-Null
    if (Test-Path $ONNX_SRC) {
        Copy-Item $ONNX_SRC (Join-Path $ROUND_EXP "best.onnx") -Force
    } else {
        Write-Host "WARNING: ONNX not found at $ONNX_SRC"
    }

    # 6) video predict (optional)
    if (Test-Path $EVAL_VIDEO) {
        Write-Host "==== [Round $r] Video prediction ===="
        & $YOLO_FULL_PATH predict model=$BEST_PT source=$EVAL_VIDEO imgsz=$IMGSZ conf=0.45 iou=$IOU_PREDICT device=$DEVICE save=True project="$RUNS\predict" name=("round{0}" -f $r)
    } else {
        Write-Host "==== [Round $r] Video prediction skipped (no input) ===="
    }

    Write-Host ("✅ Round {0} done. Student: {1}" -f $r, $BEST_PT)
}

Write-Host ""
Write-Host "🎯 ALL ROUNDS FINISHED."
Write-Host ("Final model:   " + (Join-Path $RUNS "train\dawn_semi_round$ROUNDS\weights\best.pt"))
Write-Host ("Final metrics: " + (Join-Path $RUNS "train\dawn_semi_round$ROUNDS\metrics.json"))
Write-Host ("Final ONNX:    " + (Join-Path $EXPORTS "round$ROUNDS\best.onnx"))
