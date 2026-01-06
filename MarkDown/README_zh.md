# YOLOv12 影片偵測＋追蹤＋分析（純 PyTorch 版）— 中文教學包

## 內容
- `run_infer_track_analyze_dashboard.ps1`：一鍵推論（追蹤）＋統計分析＋啟動儀表板
- `app_dashboard.py`：多 Session 常駐儀表板（Plotly 互動圖）
- `datasets/wildlife/`：資料夾骨架＋範例 `wildlife.yaml`、`hyp_dawn.yaml`
- `runs/`：預留推論與分析輸出資料夾

## 使用
```powershell
cd C:\Users\wangs\monkeyv7
.\yolo12_env\Scripts\Activate.ps1

# 一鍵推論＋追蹤＋分析＋自動開啟儀表板
powershell -ExecutionPolicy Bypass -File .\run_infer_track_analyze_dashboard.ps1 -VIDEO ".\wildlife.mp4"
powershell -ExecutionPolicy Bypass -File .\run_realtime_track_analyze_dashboard.ps1 -VIDEO "https://www.youtube.com/watch?v=UXhA0tRaCBs&pp=ygUN542854y0IOaenOWckg%3D%3D"

# 多 Session 儀表板
python app_dashboard.py --base runs/analyze --port 5050
```

# 即時同步推論
```powershell
python.exe live_stream_server.py
```