# 影片轉圖片常見加參數（可視需求加上）

控制 JPEG 品質（數字越小越好）：-q:v 2
從第 N 張開始編號：-start_number 0
先縮圖（例如長邊 1280，等比例）：-vf "scale=1280:-1"
只取片段（例如前 2 分鐘）：-t 00:02:00

範例（高品質、縮圖、從 0 開始）：
```powershell
ffmpeg -y -i .\PXL_20251122_220318891.mp4 -t 00:02:00 -r 3 -vf "scale=1280:-1" -q:v 2 .\frame_laser1_%05d.jpg
ffmpeg -y -i .\PXL_20251122_220349532.mp4 -t 00:02:00 -r 3 -vf "scale=1280:-1" -q:v 2 .\frame_laser2_%05d.jpg
ffmpeg -y -i .\PXL_20251122_220502867.mp4 -t 00:02:00 -r 3 -vf "scale=1280:-1" -q:v 2 .\frame_laser3_%05d.jpg
ffmpeg -y -i .\PXL_20251122_220540611.mp4 -t 00:02:00 -r 3 -vf "scale=1280:-1" -q:v 2 .\frame_laser4_%05d.jpg
ffmpeg -y -i .\PXL_20251122_220625042.mp4 -t 00:02:00 -r 3 -vf "scale=1280:-1" -q:v 2 .\frame_laser5_%05d.jpg
```
# YOLOv12 半監督自動化管線

本工具包包含三支 PowerShell 腳本：
1. run_yolo12_semi_train.ps1 - 單輪半監督訓練
2. run_yolo12_semi_loop.ps1 - 多輪自動訓練
3. run_yolo12_semi_loop_full.ps1 - 多輪 + 驗證 + 匯出

## 使用方法
```powershell
cd E:\Coding\monkeyv7
py -3.11 -m venv yolo12_env
.\yolo12_env\Scripts\Activate.ps1
pip install torch==2.3.1+cu121 torchaudio==2.3.1+cu121 torchvision==0.18.1+cu121 --index-url https://download.pytorch.org/whl/cu121
powershell -ExecutionPolicy Bypass -File .\run_yolo12_semi_train.ps1
powershell -ExecutionPolicy Bypass -File .\run_yolo12_semi_loop.ps1
powershell -ExecutionPolicy Bypass -File .\run_yolo12_semi_loop_full.ps1
```

訓練完成後輸出包含：
- best.pt / best.onnx
- results.json (metrics)
- predict/roundX/ (影片預測結果)


✅ YOLO TRAIN部署ven環境一次完全修復方式（最快也最乾淨）
🚀 Step 1：刪除舊壞環境

# 確保沒有在使用中：
```powershell
deactivate
```

# 然後刪除壞掉的環境：
```powershell
Remove-Item -Recurse -Force E:\Coding\monkeyv7\yolo12_env
```

🧱 Step 2：重新建立新的乾淨虛擬環境
仍然在 E:\Coding\monkeyv7 目錄內執行：

```powershell
python -m venv yolo12_env
```

# 啟用：
```powershell
powershell -command "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser"
.\yolo12_env\Scripts\Activate.ps1
```

# 檢查：
```powershell
where python
```

應該顯示：
E:\Coding\monkeyv7\yolo12_env\Scripts\python.exe

🧩 Step 3：安裝核心套件（正確的）

```powershell
pip install --upgrade pip wheel setuptools
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install ultralytics --upgrade
pip install flask opencv-python torch torchvision ultralytics

# 安裝 yt-dlp (這是推薦的 YouTube 解析器)
# 1. 確保環境已啟動 (已完成)
.\yolo12_env\Scripts\Activate.ps1

# 2. 徹底移除舊的 pafy (必須先清除官方版)
pip uninstall -y pafy

# 3. 從 GitHub 安裝包含 'set_backend' 方法的修復版 pafy。
# 這個版本是社群為了解決 YouTube 錯誤而維護的。
pip install git+https://github.com/mps-youtube/pafy.git@develop

# 4. 確保 yt-dlp 依然存在 (這是 pafy 新版本要使用的後端)
pip install yt-dlp
```

✅ Step 4：驗證 YOLOv12 指令
```powershell
C:\Users\wangs\AppData\Roaming\Python\Python311\Scripts\yolo help
```

若顯示：
Arguments received: ['yolo', 'help']. Ultralytics 'yolo' commands use the following syntax:
yolo TASK MODE ARGS
→ 完全修復成功 🎉

💡 Step 5（可選）快速測試 GPU 推論
```powershell
C:\Users\wangs\AppData\Roaming\Python\Python311\Scripts\yolo predict model=yolov8n.pt source="https://ultralytics.com/images/bus.jpg" device=0
```

你應該會看到：
Ultralytics YOLOv8  🚀
Model summary: 225 layers, 7,031,463 parameters
Results saved to runs\predict\predict

🧰 Step 6：備份這個環境（建議）
```powershell
pip freeze > requirements_yolo12.txt
```

下次可以直接：
```powershell
pip install -r requirements_yolo12.txt
```
