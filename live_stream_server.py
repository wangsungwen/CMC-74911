# live_stream_server.py (最終修復版本 - 移除 pafy，直接使用 yt-dlp)
import cv2
import time
import traceback
import argparse
import numpy as np
from flask import Flask, Response, render_template_string
from ultralytics import YOLO

# ========== NEW: Direct yt-dlp Import and Helper Function ==========
try:
    import yt_dlp
    print("INFO: yt-dlp library successfully imported.")
except ImportError:
    # 這是發生致命錯誤時的提示，提醒用戶安裝 yt-dlp
    print("FATAL ERROR: yt-dlp library not found. YouTube streaming requires: pip install yt-dlp")

def get_youtube_stream_url(video_url):
    """使用 yt-dlp API 提取最佳的直接影片串流 URL。"""
    if 'youtube.com' not in video_url and 'youtu.be' not in video_url:
        return video_url # 非 YouTube URL，直接返回

    print(f"INFO: 正在提取 YouTube 影片的直接串流 URL: {video_url}...")
    try:
        ydl_opts = {
            # 選擇最佳的 mp4 格式，包含影片和音訊
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': '%(title)s.%(ext)s',
            'quiet': True,
            'skip_download': True,
            'logtostderr': False,
            'no_warnings': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # 提取影片資訊，不下載
            info = ydl.extract_info(video_url, download=False)
            
            # 尋找直接的串流 URL
            formats = info.get('formats', [])
            
            # 尋找最佳的 mp4 串流 URL
            best_url = None
            for f in formats:
                # 尋找帶有 vcodec 且 ext 為 mp4 的格式
                if f.get('ext') == 'mp4' and f.get('vcodec') != 'none' and f.get('url'):
                    best_url = f['url']
                    break # 找到第一個符合條件的就使用

            if best_url:
                return best_url

            # 如果找不到 mp4 格式，回退到 general URL
            return info.get('url')
            
    except Exception as e:
        print(f"FATAL ERROR: yt-dlp 提取失敗: {e}")
        traceback.print_exc()
        return None
# ===================================================================


# 預設路徑 (如果命令行未提供)
DEFAULT_MODEL_PATH = r"C:\Users\wangs\monkeyv7\best.pt"
DEFAULT_VIDEO_SOURCE = "20251022.mp4" 
GLOBAL_MODEL_PATH = DEFAULT_MODEL_PATH
GLOBAL_VIDEO_SOURCE = DEFAULT_VIDEO_SOURCE

# --- Flask App 設定 ---
app = Flask(__name__)

# LIVE_HTML (Flask 模板變數)
LIVE_HTML = """
<!DOCTYPE html>
<html>
<head>
<title>YOLOv12 即時追蹤串流</title>
<style>body{font-family:Segoe UI,Arial;text-align:center;} img{max-width: 90%;}</style>
</head>
<body>
<h1>即時推論與追蹤 (Web Stream)</h1>
<p>來源: <b>{{ video_source }}</b></p>
<img id="video-stream" src="{{ url_for('video_feed') }}" width="100%">
<p>
    {% if 'youtube.com' in video_source or 'youtu.be' in video_source %}
    * 網路串流運行中，請在終端機按 Ctrl+C 停止。
    {% else %}
    * 檔案影片運行中，播放完畢後，串流將自動結束。
    {% endif %}
</p>
<hr>
<h2><a href="http://127.0.0.1:5050/">點擊前往批次分析儀表板 (若已執行 run_infer_track_analyze_dashboard.ps1)</a></h2>
</body>
</html>
"""

def generate_frames():
    """使用 YOLOv12 進行實時追蹤，並將結果編碼為 MJPEG 串流。"""
    
    current_source = GLOBAL_VIDEO_SOURCE
    
    # NEW: 嘗試獲取 YouTube 的原始串流 URL
    if 'youtube.com' in GLOBAL_VIDEO_SOURCE or 'youtu.be' in GLOBAL_VIDEO_SOURCE:
        stream_url = get_youtube_stream_url(GLOBAL_VIDEO_SOURCE)
        if stream_url:
            current_source = stream_url
            print(f"INFO: 成功提取到直接串流 URL。")
        else:
            print("FATAL: 無法提取 YouTube 串流 URL。請檢查 yt-dlp 錯誤信息。")
            current_source = None # 設置為 None 立即退出
            
    if not current_source:
        # 如果無法獲取串流，生成一個錯誤圖像並退出
        img = np.zeros((480, 640, 3), dtype="uint8")
        cv2.putText(img, "STREAM EXTRACTION FAILED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', img)
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        print(f"✅ 追蹤已結束。來源: {GLOBAL_VIDEO_SOURCE}")
        return # 退出生成器

    print(f"INFO: Starting tracking on source: {current_source}")
    
    try:
        model = YOLO(GLOBAL_MODEL_PATH) 
        # 將直接的串流 URL 傳遞給 model.track()
        results = model.track(source=current_source, stream=True, show=False, 
                              tracker="bytetrack.yaml", imgsz=1280, conf=0.45)

        for r in results:
            if r is None:
                continue
                
            frame = r.plot() 
            ret, buffer = cv2.imencode('.jpg', frame)
            
            if not ret: 
                print("⚠️ 幀編碼失敗，跳過。")
                continue
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
    except Exception as e:
        error_msg = f"❌ 在 generate_frames 中發生嚴重錯誤: {e}"
        print(error_msg)
        traceback.print_exc()
        
        img = np.zeros((480, 640, 3), dtype="uint8") 
        cv2.putText(img, "STREAM ERROR", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, str(e)[:60], (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        ret, buffer = cv2.imencode('.jpg', img)
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
    print(f"✅ 追蹤已結束。來源: {GLOBAL_VIDEO_SOURCE}")
    return

@app.route('/video_feed')
def video_feed():
    """MJPEG 串流路由"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """主頁面路由"""
    return render_template_string(LIVE_HTML, video_source=GLOBAL_VIDEO_SOURCE)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLOv12 Live Stream Server.")
    parser.add_argument("--video", type=str, default=DEFAULT_VIDEO_SOURCE, help="影片來源 (檔案路徑, RTSP, URL, 0 for webcam)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="模型路徑")
    parser.add_argument("--port", type=int, default=5000, help="服務器端口")
    args = parser.parse_args()
    
    GLOBAL_VIDEO_SOURCE = args.video
    GLOBAL_MODEL_PATH = args.model
    
    print(f"🚀 Live Stream Dashboard on http://127.0.0.1:{args.port}/")
    app.run(host='0.0.0.0', port=args.port, debug=False)