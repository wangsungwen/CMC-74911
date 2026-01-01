新策略：主動式「困難負樣本挖掘」(Active Hard Negative Mining)
這個策略的目標不再是「增加猴子樣本」，而是「找出模型最容易混淆的背景」，並強制它學會「這些不是猴子」。

這將能進一步降低我們 5% 的誤報率 (FP)，並可能間接提升 mAP@.50-.95（因為模型更專注於真正的目標）。

實作步驟：
步驟 1：找出「困難負樣本」(Hard Negatives)
我們需要使用您最強的教師模型 (dawn_supervised_s_v1 - best.pt) 來找出它「最常犯錯」的影像。
# 進入您的 YOLOv12 環境
# 步驟 1：使用 s@640 教師模型，以「中低信心」進行預測

yolo predict \
    model="runs/train/dawn_supervised_s_v1/weights/best.pt" \
    source="datasets/wildlife/images/unlabeled" \
    imgsz=640 \
    conf=0.30 \
    iou=0.6 \
    save=True \
    save_txt=False \
    save_conf=True \
    project="runs/hard_negative_mining" \
    name="find_conf_30_70"

# 指令：
yolo predict model="runs/train/dawn_supervised_s_v1_640x8/weights/best.pt" source="datasets/wildlife/images/unlabeled" imgsz=640 conf=0.30 iou=0.6 save=True save_txt=False save_conf=True project="runs/hard_negative_mining" name="find_conf_30_70"

關鍵參數說明：

conf=0.30: 我們故意使用一個中低的信心閾值。

save_txt=False: 我們不要標籤，因為我們要手動審查。

save=True: 這會在 runs/hard_negative_mining/find_conf_30_70/ 資料夾中儲存所有「被標記」的影像。

步驟 2：手動審查與篩選 (最關鍵的一步)
打開 runs/hard_negative_mining/find_conf_30_70/ 資料夾。

您會在這裡看到所有模型認為「有 30%~100% 機率是猴子」的影像。

手動審查這些影像，並找出所有錯誤的預測 (False Positives)。例如：

dog_image.jpg (模型在狗身上畫了個 45% 信心的框)

human_image.jpg (模型在人身上畫了個 60% 信心的框)

blurry_rock.jpg (模型在石頭上畫了個 35% 信心的框)

這，就是「困難負樣本」。這些是模型最容易混淆的對象。

步驟 3：建立「HNM 強化資料集」
複製影像： 將您在步驟 2 中篩選出的所有「困難負樣本」影像（dog_image.jpg, human_image.jpg 等）複製到您原始的 datasets/wildlife/images/train 資料夾中。

[重要] 不要建立標籤： 千萬不要為這些影像建立 .txt 標籤檔。YOLO 在訓練時，如果發現一張影像沒有對應的 .txt 標籤檔，就會自動將其視為「純背景負樣本」。

完成： 您的 train 資料夾現在包含了： (A) 人工標註的獼猴 + (B) 原有的純背景 + (C) 我們新加入的「困難負樣本」（模型最愛誤判的狗、人、石頭）。

步驟 4：重新訓練「強化版」教師模型
現在，我們使用這個「強化資料集」從頭開始重新訓練 yolo12s 模型。我們期望這個新模型的 FP 率能低於 5%。

# 步驟 4：使用強化後的資料集，從頭訓練新模型
yolo train \
    model="yolo12s.pt" \
    data="datasets/wildlife/wildlife.yaml" \
    imgsz=640 \
    epochs=100 \
    batch=8 \
    workers=8 \
    device=0 \
    project="runs/train" \
    name="dawn_supervised_s_v2_HNM" \
    ... (其餘參數與 s@640 相同)

策略,您的舊 SSL (失敗),新 HNM 策略 (推薦)
目標,挖掘正樣本 (更多猴子),挖掘負樣本 (困難背景)
方法,predict + conf > 0.8 過濾,predict + conf > 0.3 過濾
處理,保留「高信心」標籤，加入訓練,手動審查，只保留「錯誤預測」的影像，不帶標籤加入訓練
結果,毒害訓練集，FP率上升 (5% -> 9%),強化訓練集，FP率有望下降 (5% -> 3%?)

這個「困難負樣本挖掘」(HNM) 策略，才是您目前真正需要的、能有效降低誤報率的「SSL 負樣本策略」。
