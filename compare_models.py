import cv2
import numpy as np
from ultralytics import YOLO

# ================= 設定區 =================
# 設定兩個模型的路徑
MODEL_A_PATH = 'runs/train/dawn_supervised_s_v1/weights/best.pt'  # 基準線
MODEL_B_PATH = 'runs/train/dawn_semi_round3/weights/best.pt'      # 半監督 Round 3

# 測試影片路徑 (若為 0 則使用 Webcam)
VIDEO_SOURCE = 'video/20251022.mp4' 
# 如果沒有影片，想用 Webcam 測試，請改為: VIDEO_SOURCE = 0

# 設定信心閾值 (建議設為 0.35 過濾雜訊)
CONF_THRESHOLD = 0.35
# ==========================================

def main():
    # 1. 載入模型
    print(f"正在載入模型 A: {MODEL_A_PATH}...")
    model_a = YOLO(MODEL_A_PATH)
    print(f"正在載入模型 B: {MODEL_B_PATH}...")
    model_b = YOLO(MODEL_B_PATH)

    # 2. 開啟影片來源
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"錯誤：無法開啟影片來源 {VIDEO_SOURCE}")
        return

    # 取得影片資訊
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 設定視窗
    window_name = "Left: Baseline (Supervised) | Right: Semi-SL Round 3"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("開始推論...按 'q' 鍵離開")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 3. 模型推論
        # stream=True 可節省記憶體，verbose=False 減少 Log
        results_a = model_a.predict(frame, conf=CONF_THRESHOLD, verbose=False)
        results_b = model_b.predict(frame, conf=CONF_THRESHOLD, verbose=False)

        # 4. 繪製結果
        # plot() 會回傳標註好的 BGR 圖片
        annotated_frame_a = results_a[0].plot()
        annotated_frame_b = results_b[0].plot()

        # 在畫面上加上標籤文字
        cv2.putText(annotated_frame_a, "Baseline (Supervised)", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame_b, "Semi-SL Round 3", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # 5. 左右拼接 (Horizontal Concatenation)
        combined_frame = np.hstack((annotated_frame_a, annotated_frame_b))

        # 顯示
        cv2.imshow(window_name, combined_frame)

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()