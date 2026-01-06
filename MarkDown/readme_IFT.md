第一階段：監督式首輪訓練與背景負樣本優化（建立教師模型）
在啟動半監督學習流程之前，必須先利用您已標註的獼猴數據（以及其他目標物種）訓練一個穩健的「教師模型」（Teacher Model）。

步驟 1.1：準備資料與負樣本策略
為確保模型能精準偵測「獼猴」並有效忽略「非目標對象」（人、貓、狗），您應採用背景負樣本優化策略（策略 B）：
1. 目標物種標註： 在用於首輪監督式訓練的資料集（例如 images/train）中，僅標註您的有害目標物種（例如獼猴、山羌、松鼠）。
2. 非目標物種處理： 影像中偶爾出現的非目標物種（人、貓、狗）應刻意不予標註。此舉會訓練模型將這些對象視為隱性背景樣本，從而在損失函數優化過程中學習抑制對它們的反應，從根本上降低對無害目標的誤報率（False Positives）。
3. 純背景樣本： 確保訓練集中包含約 10% 的純背景影像（不含任何標註物件），以增強模型對環境噪聲的抗干擾能力。
4. 未標註資料準備： 將包含非目標對象或其他場景的大量未標註影像放置於 images/unlabeled 資料夾。

步驟 1.2：執行首輪監督式訓練（Teacher Model）
使用您的標註資料對 YOLOv12 模型進行訓練，通常是從 COCO 等通用資料集上預訓練的權重開始進行遷移學習，以加速收斂並提升準確度。
# 假設您已進入 YOLOv12 環境，並設置好專案路徑 $PROJ
# 使用 yolo train 指令
yolo train `
  model="yolo12n.pt" ` # 預訓練權重
  data="datasets/wildlife/wildlife.yaml" ` # 包含 macaque 類別的 YAML
  hyp="configs/hyp_dawn.yaml" ` # 使用專門為低光環境設計的增強策略
  imgsz=1280 epochs=100 batch=16 workers=8 `
  device=0 project="runs/train" name="dawn_supervised_v1"
指令：
yolo train model="yolo12n.pt" data="datasets/wildlife/wildlife.yaml" hyp="datasets/wildlife/hyp_dawn.yaml" imgsz=1280 epochs=100  batch=16 workers=8 device=0 project="runs/train" name="dawn_supervised_v1"

yolo train model="yolo12n.pt" data="datasets/wildlife/wildlife.yaml" imgsz=1280 epochs=100 batch=16 workers=8 device=0 project="runs/train" name="dawn_supervised_v1" degrees=0.0 translate=0.08 scale=0.3 shear=0.0 perspective=0.0 mosaic=1.0 copy_paste=0.35 mixup=0.1 hsv_h=0.015 hsv_s=0.6 hsv_v=0.45 brightness=0.25 contrast=0.25 gamma_min=0.6 gamma_max=1.6 motion_blur_prob=0.2 defocus_prob=0.15 fog_prob=0.25 optimizer=AdamW lr0=0.001 lrf=0.01 momentum=0.9 weight_decay=0.05 warmup_epochs=3 ema=true

[best]:
yolo train model="yolo12n.pt" data="datasets/wildlife/wildlife.yaml" imgsz=640 epochs=100 batch=8 workers=8 device=0 project="runs/train" name="dawn_supervised_v1" degrees=0.0 translate=0.08 scale=0.3 shear=0.0 perspective=0.0 mosaic=1.0 copy_paste=0.35 mixup=0.1 hsv_h=0.015 hsv_s=0.6 hsv_v=0.45 optimizer="AdamW" lr0=0.001 lrf=0.01 momentum=0.9 weight_decay=0.05 warmup_epochs=3
# 訓練完成後，教師模型權重將在 runs\train\dawn_supervised_v1\weights\best.pt

# 進入您的 YOLOv12 環境
# 執行新的監督式訓練（使用 yolo12s.pt）
yolo train \
    model="yolo12s.pt" \
    data="datasets/wildlife/wildlife.yaml" \
    imgsz=1280 \
    epochs=100 \
    batch=16 \
    workers=8 \
    device=0 \
    project="runs/train" \
    name="dawn_supervised_s_v1" \
    degrees=0.0 \
    translate=0.08 \
    scale=0.3 \
    shear=0.0 \
    perspective=0.0 \
    mosaic=1.0 \
    copy_paste=0.35 \
    mixup=0.1 \
    hsv_h=0.015 \
    hsv_s=0.6 \
    hsv_v=0.45 \
    optimizer="AdamW" \
    lr0=0.001 \
    lrf=0.01 \
    momentum=0.9 \
    weight_decay=0.05 \
    warmup_epochs=3

# [指令]
yolo train model="yolo12l.pt" data="datasets/wildlife/wildlife.yaml" imgsz=640 epochs=100 batch=16 workers=8 device=0 project="runs/train" name="dawn_supervised_s_v1" degrees=0.0 translate=0.08 scale=0.3 shear=0.0 perspective=0.0 mosaic=1.0 copy_paste=0.35 mixup=0.1 hsv_h=0.015 hsv_s=0.6 hsv_v=0.45 optimizer="AdamW" lr0=0.001 lrf=0.01 momentum=0.9 weight_decay=0.05 warmup_epochs=3

# yolo12l [指令]
yolo train model="yolo12l.pt" data="datasets/wildlife/wildlife.yaml" imgsz=640 epochs=100 batch=32 workers=16 device=0 project="runs/train" name="dawn_supervised_l_v1" degrees=15.0 translate=0.1 scale=0.4 shear=2.0 perspective=0.0 mosaic=1.0 copy_paste=0.35 mixup=0.15 hsv_h=0.015 hsv_s=0.6 hsv_v=0.45 optimizer="AdamW" lr0=0.001 lrf=0.01 momentum=0.9 weight_decay=0.05 warmup_epochs=3

--------------------------------------------------------------------------------
第二階段：半監督式偽標籤訓練（Student Model）
利用大量未標註資料通過偽標籤程序（Pseudo-labeling）進一步強化模型，特別是鞏固其對非目標物種（人、貓、狗）的忽略能力。

步驟 2.1：產生偽標籤（Inference）
使用第一階段訓練好的「教師模型」對 images/unlabeled 中的大量未標註影像進行推論。
關鍵操作： 推論時必須確保儲存每個偵測框的置信度（conf），這對於後續篩選至關重要。
# 產⽣偽標籤：使用 Teacher 權重對未標註資料推論
yolo predict `
  model="runs/train/dawn_supervised_v1/weights/best.pt" ` 
  source="datasets/wildlife/images/unlabeled" `
  imgsz=1280 conf=0.5 iou=0.6 `
  save=True save_txt=True save_conf=True ` # 確保儲存 .txt 標籤檔與置信度
  project="runs/pseudo" name="round1"
# 指令 (Round1)：
yolo predict model="runs/train/dawn_supervised_s_v1/weights/best.pt" source="datasets/wildlife/images/unlabeled" imgsz=640 conf=0.6 iou=0.6 save=True save_txt=True save_conf=True project="runs/pseudo" name="round1"
# 指令 (Round2)：
yolo predict model="runs/train/dawn_semi_round1/weights/best.pt" source="datasets/wildlife/images/unlabeled" imgsz=1280 conf=0.5 iou=0.6 save=True save_txt=True save_conf=True project="runs/pseudo" name="round2"
# 指令 (Round3)：
yolo predict model="runs/train/dawn_semi_round2/weights/best.pt" source="datasets/wildlife/images/unlabeled" imgsz=1280 conf=0.5 iou=0.6 save=True save_txt=True save_conf=True project="runs/pseudo" name="round3"
# 產⽣的偽標籤檔案會存放在 runs\pseudo\round1\labels\*.txt (格式為：cls xc yc w h conf)]

步驟 2.2：篩選高信心偽標籤與淨化（Filtering）
這是確保 SSL 質量的最關鍵步驟。嚴格的篩選可確保只有模型對目標物種有極高信心的預測才被保留。由於模型在第一階段已被訓練將非目標物種視為背景，因此它們對這些對象的預測置信度通常會很低，在這一階段會被自動濾除，維持了背景負樣本優化的原則。
1. 設定高置信度閾值： 設定一個嚴格的閾值，通常建議 CONF≥0.80。
2. 過濾： 篩選掉置信度低於 0.80 的偵測框。
3. （進階）面積過濾： 額外篩除面積小於 64 像素² 的偽標籤，以移除雜訊。
4. 格式清洗： 將合格的 6 欄偽標籤（cls xc yc w h conf）轉換為標準 YOLO 格式的 5 欄（cls xc yc w h），移除置信度欄位。
5. 儲存： 將清洗後的標註檔和對應的影像拷貝到新的子訓練資料夾 (images/train_pseudo 和 labels/train_pseudo)。
（您可以參考來源文件中提供的 PowerShell 腳本來執行自動篩選與拷貝動作。）
# 範例 1：(基本執行) 只過濾 Round 1，信心 0.80 (預設值)。
PowerShell.exe -ExecutionPolicy Bypass -File ".\filter_pseudo_labels_1.ps1" -Round 1 -ConfidenceThreshold 0.90
# 範例 2：(進階執行) 過濾 Round 2，信心 0.75，並啟用面積過濾 (最小 64 像素)，假設 predict 時 imgsz=640。
PowerShell.exe -ExecutionPolicy Bypass -File ".\filter_pseudo_labels_2.ps1" -Round 2 -ConfidenceThreshold 0.85
PowerShell.exe -ExecutionPolicy Bypass -File ".\filter_pseudo_labels_3.ps1" -Round 3 -ConfidenceThreshold 0.80

步驟 2.3：半監督再訓練（Student Model）
修改資料集設定檔，合併原始人工標註與高信心偽標籤，然後使用教師模型權重作為起點進行再訓練。
# 1. 更新 wildlife.yaml，合併人工標註集和偽標籤集
# train: [images/train, images/train_pseudo]

# 2. 訓練 Student Model (以 Teacher 的 best.pt 作為起點)
yolo train `
  model="runs/train/dawn_supervised_v1/weights/best.pt" ` 
  data="datasets/wildlife/wildlife.yaml" `
  imgsz=1280 epochs=80 batch=16 workers=8 `
  device=0 project="runs/train" name="dawn_semi_round1"

# 指令 (Round1)：
yolo train model="runs/train/dawn_supervised_v1/weights/best.pt" data="datasets/wildlife/wildlife.yaml" imgsz=1280 epochs=80 batch=16 workers=8 device=0 project="runs/train" name="dawn_semi_round1"
# 指令 (Round2)：
yolo train model="runs/train/dawn_semi_round1/weights/best.pt" data="datasets/wildlife/wildlife.yaml" imgsz=1280 epochs=80 batch=16 workers=8 device=0 project="runs/train" name="dawn_semi_round2"
# 指令 (Round3)：
yolo train model="runs/train/dawn_semi_round2/weights/best.pt" data="datasets/wildlife/wildlife.yaml" imgsz=1280 epochs=80 batch=16 workers=8 device=0 project="runs/train" name="dawn_semi_round3"
# 訓練結束後，產生 Student 模型權重 runs\train\dawn_semi_round1\weights\best.pt

# 進入您的 YOLOv12 環境
# 執行學生模型訓練（使用合併後的資料集）

# YOLO12s對照組：
yolo train model="yolo12s.pt" data="datasets/wildlife/wildlife.yaml" imgsz=640 epochs=100 batch=8 workers=8 device=0  project="runs/train" name="dawn_student_ssl_v1" degrees=0.0 translate=0.08 scale=0.3 shear=0.0 perspective=0.0 mosaic=1.0 copy_paste=0.35 mixup=0.1 hsv_h=0.015 hsv_s=0.6 hsv_v=0.45 optimizer="AdamW" lr0=0.001 lrf=0.01 momentum=0.9 weight_decay=0.05 warmup_epochs=3

# 所有的 .txt 標籤檔案（包括 train 和 val 資料夾中所有的檔案）強制轉換為 UTF-8 (無 BOM) 編碼
# 1. 設定您的 labels 資料夾路徑

# 在 monkeyv7 專案根目錄下，開啟 PowerShell，然後執行以下腳本。這會一勞永逸地修復所有標籤檔案的編碼問題
$trainDir = "C:\Users\wangs\monkeyv7\datasets\wildlife\labels\train_pseudo"
$valDir   = "C:\Users\wangs\monkeyv7\datasets\wildlife\labels\val"

# 建立一個無 BOM 的 UTF-8 編碼器
$utf8NoBOM = New-Object System.Text.UTF8Encoding $false  # $false 代表 "不要 BOM"

# 2. 處理 Train 資料夾
$trainFiles = Get-ChildItem -Path $trainDir -Filter "*.txt" -Recurse
Write-Host "--- 正在處理 Train 資料夾 ($($trainFiles.Count) 個檔案) ---"

foreach ($file in $trainFiles) {
    try {
        # [System.IO.File] 類別能更精確地控制編碼
        $text = [System.IO.File]::ReadAllText($file.FullName)
        [System.IO.File]::WriteAllText($file.FullName, $text, $utf8NoBOM)
    } catch {
        Write-Error "  - 轉換失敗: $($file.Name) - $($_.Exception.Message)"
    }
}

# 3. 處理 Val 資料夾
$valFiles = Get-ChildItem -Path $valDir -Filter "*.txt" -Recurse
Write-Host "--- L.r (正在處理 Val 資料夾 ($($valFiles.Count) 個檔案) ---"

foreach ($file in $valFiles) {
    try {
        $text = [System.IO.File]::ReadAllText($file.FullName)
        [System.IO.File]::WriteAllText($file.FullName, $text, $utf8NoBOM)
    } catch {
        Write-Error "  - 轉換失敗: $($file.Name) - $($_.Exception.Message)"
    }
}

Write-Host "---"
Write-Host "✅ [編碼轉換完成] ✅"
Write-Host "所有 .txt 標籤檔案均已轉換為 UTF-8 (無 BOM) 格式。"
Write-Host "您現在可以重新執行 yolo train 指令了。"

--------------------------------------------------------------------------------
第三階段：利用增量式微調（IFT）整合「雷射光點」類別
完成半監督訓練後，模型（Student Model）已經穩健地學會偵測獼猴並忽略非目標對象。現在，您將利用增量式微調（IFT）避免災難性遺忘（Catastrophic Forgetting）。
IFT 核心優勢： IFT 可以將訓練時間降低約 68%，並在訓練週期內維持舊有類別 ≥95% 的精度，同時成功學習新類別。
步驟 3.1：標籤空間擴增與資料集準備
1. 準備新類別資料： 準備包含「雷射光點」標註（Bounding Box）的新資料集。
2. 新增新類別資料集配置： 將新的「雷射光點」資料集存放在"datasets\wildlife_laser"。修改 YOLO 設定檔（例如 wildlife_plus_laser.yaml），擴充標籤空間，新增 laser_spot 類別。
類別 ID 名稱
0 macaque
1 spot
3. 執行 modify_labels.py 將光點的標記編號 由 0 改為 1

步驟 3.2：選擇性凍結層與低學習率微調（核心步驟）
為避免災難性遺忘，必須遵循層級式微調原理：凍結神經網路中負責提取通用、低層次特徵的層，僅訓練負責最終決策和新類別分類的層。
1. 選擇性凍結： 凍結 YOLOv12 的骨幹網路（Backbone），通常為前 10 層（freeze=10），以保留模型透過遷移學習和半監督學習學到的舊有視覺特徵（如邊緣、紋理、動物外觀）。
2. 訓練目標： 僅開放偵測頭（Detection Head）與分類層參與訓練，使其適應新的「雷射光點」類別。
3. 低學習率： 採用極低的初始學習率（例如 lr0=0.0005）與短週期（例如 epochs=40）來進行微調，以最小化對舊權重的破壞。

步驟 3.3：執行增量式微調（IFT Command）
使用第二階段訓練出的 Student Model (dawn_semi_round1/weights/best.pt) 作為起點，執行 IFT。
# 執行增量式微調指令
yolo train model="runs/train/dawn_semi_round3/weights/best.pt" `
  data=datasets/wildlife/wildlife_plus_laser.yaml `
  epochs=50 `
  optimizer=AdamW lr0=0.0002 `  # 關鍵：強制使用低學習率
  batch=8 `                    # 批次 8 (如果 640 仍 OOM，請改為 4)
  imgsz=640 `                  # 關鍵：降低解析度以適應 4GB VRAM
  freeze=0 `
  name=wildlife_laser_finetune_v2 `
  device=0
# 指令：
yolo train model="runs/train/dawn_semi_round3/weights/best.pt" data=datasets/wildlife/wildlife_plus_laser.yaml epochs=50  optimizer=AdamW lr0=0.0002 batch=16 imgsz=416 freeze=0 name=wildlife_laser_finetune_v2 device=0

步驟 3.4：結果與知識保留驗證
完成 IFT 後，根據步驟 3.1 的設定，您的標籤空間已明確定義為 0: macaque 和 1: spot。因此，完成 IFT 後，模型將輸出一個具備兩個類別（獼猴與雷射光點）偵測能力的權重檔案

# 指令：
test1:
yolo train model="runs/train/dawn_semi_round3/weights/best.pt" data=datasets/wildlife/wildlife_plus_laser.yaml epochs=50  optimizer=AdamW lr0=0.0002 batch=8 imgsz=1280 freeze=0 name=wildlife_laser_finetune_v2 device=0

test2:
yolo train model="runs/train/dawn_semi_round3/weights/best.pt" data=datasets/wildlife/wildlife_plus_laser.yaml epochs=75  optimizer=AdamW lr0=0.0002 batch=8 imgsz=640 freeze=0 copy_paste=0.1 name=wildlife_laser_finetune_v4_copypaste device=0

test3:
yolo train model="runs/detect/wildlife_laser_finetune_v2_1141022/weights/best.pt" data=datasets/wildlife/wildlife_plus_laser.yaml epochs=100 batch=8 imgsz=640 name=wildlife_laser_finetune_v4_copypaste device=0

test4:
yolo train model="runs/train/dawn_semi_round3/weights/best.pt" data=datasets/wildlife/wildlife_plus_laser.yaml epochs=75  optimizer=AdamW lr0=0.0002 batch=8 imgsz=416 freeze=0 copy_paste=0.1 scale=0.1 name=wildlife_laser_finetune_v5_final_fix device=0

val4:
yolo val model="runs/detect/wildlife_laser_finetune_v5_final_fix_1141023_1/weights/best.pt" data=datasets/wildlife/wildlife_plus_laser.yaml conf=0.4 imgsz=640 split=val name=v5_validation_conf0_4

predict4:
yolo predict model="runs/detect/wildlife_laser_finetune_v5_final_fix/weights/best.pt" source="您的影片路徑.mp4" conf=0.4 device=0

[Best]:
yolo train model="runs/train/dawn_semi_round1/weights/best.pt" data=datasets/wildlife/wildlife_plus_laser.yaml epochs=100  optimizer=AdamW lr0=0.0002 batch=16 imgsz=640 freeze=0 copy_paste=0.1 scale=0.1 name=wildlife_laser_finetune_v5_final_fix device=0

yolo train model="runs/detect/wildlife_laser_finetune_v5_final_fix/weights/best.pt" data=datasets/wildlife/wildlife_plus_laser.yaml epochs=75  optimizer=AdamW lr0=0.0005 batch=16 imgsz=416 freeze=0 copy_paste=0.1 scale=0.1 name=wildlife_laser_finetune_v5_final_fix device=0

# yolov12s
yolo train model="runs/train/dawn_semi_round3/weights/best.pt" ^
data="datasets/wildlife/wildlife_plus_laser.yaml" ^
epochs=50 ^
imgsz=1280 ^
batch=16 ^
device=0 ^
optimizer="AdamW" ^
lr0=0.005 ^
freeze=10 ^
name="wildlife_laser_finetune_v2" ^
cos_lr=True ^
warmup_epochs=3

指令:
yolo train model="runs/train/dawn_semi_round3/weights/best.pt" data="datasets/wildlife/wildlife_plus_laser.yaml" epochs=50 imgsz=1280 batch=16 device=0 optimizer="AdamW" lr0=0.005 freeze=10 name="wildlife_laser_finetune_v1" cos_lr=True warmup_epochs=3

[Resume]:
yolo train resume=True model="runs/detect/wildlife_laser_finetune_v5_final_fix/weights/last.pt" amp=False

驗證conf值
yolo val model="runs/detect/wildlife_laser_finetune_v5_final_fix_1141024_r3/weights/best.pt" data=datasets/wildlife/wildlife_plus_laser.yaml conf=0.4 imgsz=416 split=val name=v5_final_fix_416_validation_conf0_4

實證推論(影片)
yolo predict model="runs/detect/wildlife_laser_finetune_v5_final_fix_1141024_r3/weights/best.pt" source="monkey_video_1.mp4" conf=0.5 save=True name=predict_conf_0_5