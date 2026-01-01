<#
 YOLOv12 Single-Round Semi-Supervised Training Pipeline
#>

# Set console to UTF-8
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

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

# --- Configuration ---
$RUNS = Join-Path $BASE "runs"
$MODEL_PATH = Join-Path $RUNS "train\dawn_supervised_v1\weights\best.pt"
$MODEL = if (Test-Path $MODEL_PATH) { $MODEL_PATH } else { "yolo12n.pt" }
if ($MODEL -eq "yolo12n.pt") {
    Write-Host "WARNING: Pre-trained model not found at $MODEL_PATH. Using 'yolo12n.pt' as a fallback."
}

$DEVICE = "0"
$CONF_PREDICT = 0.5
$CONF_FILTER = 0.8
$EPOCHS = 100
$BATCH = 2
$IMGSZ = 416

# --- Step 1: Generate Pseudo-Labels ---
Write-Host "`n[1/4] Generating pseudo-labels..."
$UNLABELED_IMG_DIR = Join-Path $DATASET "images\unlabeled"
if (-not (Test-Path $UNLABELED_IMG_DIR)) {
    New-Item -ItemType Directory -Force -Path $UNLABELED_IMG_DIR | Out-Null
    Write-Host "INFO: 'unlabeled' directory created. Place images here for pseudo-labeling."
}

$imgFiles = Get-ChildItem -Path $UNLABELED_IMG_DIR -Include *.jpg, *.jpeg, *.png -File -Recurse
if ($imgFiles.Count -gt 0) {
    & $YOLO_FULL_PATH predict model=$MODEL source=$UNLABELED_IMG_DIR imgsz=$IMGSZ conf=$CONF_PREDICT iou=0.6 save=True save_txt=True save_conf=True project="$RUNS\pseudo" name="round1" device=$DEVICE
} else {
    Write-Host "WARNING: No images found in the 'unlabeled' directory. Skipping pseudo-label generation."
}

# --- Step 2: Filter High-Confidence Pseudo-Labels ---
Write-Host "`n[2/4] Filtering pseudo-labels with confidence >= $CONF_FILTER..."
$SRC_LABELS = Join-Path $RUNS "pseudo\round1\labels"
$DEST_LABELS = Join-Path $DATASET "labels\train_pseudo"
$DEST_IMAGES = Join-Path $DATASET "images\train_pseudo"
New-Item -ItemType Directory -Force -Path $DEST_LABELS, $DEST_IMAGES | Out-Null

if (-not (Test-Path $SRC_LABELS)) {
    Write-Host "WARNING: Source labels directory not found at $SRC_LABELS. Skipping filtering step."
} else {
    Get-ChildItem -Path $SRC_LABELS -Filter *.txt -Recurse | ForEach-Object {
        $file = $_
        $filteredLines = Get-Content $file.FullName | Where-Object {
            $parts = $_ -split '\s+'
            $parts.Count -ge 6 -and [double]$parts[5] -ge $CONF_FILTER
        }

        if ($filteredLines) {
            $outputContent = $filteredLines | ForEach-Object { ($_.Split(' '))[0..4] -join ' ' }
            $outputPath = Join-Path $DEST_LABELS $file.Name
            $outputContent | Set-Content -Path $outputPath -Encoding utf8

            $imageName = "$($file.BaseName).jpg"
            $srcImagePath = Join-Path $UNLABELED_IMG_DIR $imageName
            if (Test-Path $srcImagePath) {
                Copy-Item -Path $srcImagePath -Destination $DEST_IMAGES -Force
            }
        }
    }
}

# --- Step 3: Update Dataset YAML ---
Write-Host "`n[3/4] Updating dataset YAML file..."
$absPath = $DATASET -replace '\\', '/'
$yamlContent = @"
path: $absPath
train:
  - images/train
  - images/train_pseudo
val: images/val
test: images/test
names:
  0: monkey
"@
$yamlPath = Join-Path $DATASET "wildlife.yaml"
Set-Content -Path $yamlPath -Value $yamlContent -Encoding utf8

# --- Step 4: Start Semi-Supervised Retraining ---
Write-Host "`n[4/4] Starting semi-supervised retraining..."
& $YOLO_FULL_PATH train model=$MODEL data=$yamlPath imgsz=$IMGSZ epochs=$EPOCHS batch=$BATCH device=$DEVICE project="$RUNS\train" name="dawn_semi_round1"

Write-Host "`n[OK] Semi-supervised training complete. Model saved in '$RUNS\train\dawn_semi_round1\weights\best.pt'"


