# live_stream_server.py (打包專用最終版)
import cv2
import time
import traceback
import argparse
import numpy as np
import sys  # <-- 1. 新增
import os   # <-- 1. 新增
from flask import Flask, Response, render_template_string, request, redirect, url_for
from ultralytics import YOLO

# ========== 2. MODIFIED: 新增 resource_path 函數並取代舊的路徑定義 ==========
def resource_path(relative_path):
    """ 獲取資源的絕對路徑，適用於開發和 PyInstaller 打包 """
    try:
        # PyInstaller 創建一個臨時資料夾並將路徑存在 _MEIPASS
        base_path = sys._MEIPASS
        print(f"INFO: 正在從 PyInstaller 臨時目錄加載資源: {base_path}")
    except Exception:
        # 開發環境中，_MEIPASS 不存在，使用腳本所在目錄
        base_path = os.path.abspath(".")
        print(f"INFO: 正在從開發目錄加載資源: {base_path}")

    return os.path.join(base_path, relative_path)

# 預設路徑 (使用 resource_path 函數來確保 .exe 能找到模型)
DEFAULT_MODEL_PATH = resource_path("best.pt")
DEFAULT_VIDEO_SOURCE = "20251022.mp4" 
GLOBAL_MODEL_PATH = DEFAULT_MODEL_PATH
GLOBAL_VIDEO_SOURCE = DEFAULT_VIDEO_SOURCE
# ===================================================================

# Direct yt-dlp Import and Helper Function
try:
    import yt_dlp
    print("INFO: yt-dlp library successfully imported.")
except ImportError:
    print("FATAL ERROR: yt-dlp library not found. YouTube streaming requires: pip install yt-dlp")

def get_youtube_stream_url(video_url):
    """使用 yt-dlp API 提取最佳的直接影片串流 URL。"""
    if 'youtube.com' not in video_url and 'youtu.be' not in video_url:
        return video_url # 非 YouTube URL，直接返回

    print(f"INFO: 正在提取 YouTube 影片的直接串流 URL: {video_url}...")
    try:
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': '%(title)s.%(ext)s',
            'quiet': True,
            'skip_download': True,
            'logtostderr': False,
            'no_warnings': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            formats = info.get('formats', [])
            
            best_url = None
            for f in formats:
                if f.get('ext') == 'mp4' and f.get('vcodec') != 'none' and f.get('url'):
                    best_url = f['url']
                    break 

            if best_url:
                return best_url

            return info.get('url')
            
    except Exception as e:
        print(f"FATAL ERROR: yt-dlp 提取失敗: {e}")
        traceback.print_exc()
        return None

# --- Flask App 設定 ---
app = Flask(__name__)

# LIVE_HTML (Flask 模板變數)
LIVE_HTML = """
<!DOCTYPE html>
<html>
<head>
<title>YOLO 即時追蹤串流</title>
<style>
    body { font-family: Segoe UI, Arial; text-align: center; margin: 20px; }
    img { max-width: 90%; border: 1px solid #ccc; background: #000; }
    h1 { color: #333; }
    form { margin: 20px auto; padding: 15px; border: 1px solid #ddd; border-radius: 8px; max-width: 800px; }
    input[type="text"] { width: 70%; padding: 8px; font-size: 1em; }
    button { padding: 8px 15px; font-size: 1em; cursor: pointer; }
</style>
</head>
<body>
<h1>即時推論與追蹤 (Web Stream)</h1>

<form method="POST" action="/">
    <label for="video_source_input"><b>輸入新來源:</b></label>
    <br><br>
    <input type="text" id="video_source_input" name="video_source_input" 
           size="60" placeholder="輸入 MP4/RTSP/YouTube URL 或 0 (webcam)">
    <button type="submit">更新串流</button>
</form>
<hr>

<p>目前來源: <b>{{ video_source }}</b></p>
<img id="video-stream" src="{{ url_for('video_feed') }}" width="100%">
<p>
    {% if 'youtube.com' in video_source or 'youtu.be' in video_source or 'rtsp://' in video_source %}
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
    """使用 YOLO 進行實時追蹤，並將結果編碼為 MJPEG 串流。"""
    
    current_source_display = GLOBAL_VIDEO_SOURCE
    current_source_process = GLOBAL_VIDEO_SOURCE
    
    if 'youtube.com' in current_source_process or 'youtu.be' in current_source_process:
        stream_url = get_youtube_stream_url(current_source_process)
        if stream_url:
            current_source_process = stream_url
            print(f"INFO: 成功提取到直接串流 URL。")
        else:
            print("FATAL: 無法提取 YouTube 串流 URL。請檢查 yt-dlp 錯誤信息。")
            current_source_process = None
            
    if not current_source_process:
        img = np.zeros((480, 640, 3), dtype="uint8")
        cv2.putText(img, "STREAM EXTRACTION FAILED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, f"Source: {current_source_display}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        ret, buffer = cv2.imencode('.jpg', img)
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        print(f"✅ 追蹤已結束。來源: {current_source_display}")
        return

    print(f"INFO: Starting tracking on source: {current_source_process}")
    
    try:
        # 這裡會使用 resource_path 解析後的 GLOBAL_MODEL_PATH
        model = YOLO(GLOBAL_MODEL_PATH) 
        results = model.track(source=current_source_process, stream=True, show=False, 
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
        cv2.putText(img, f"Source: {current_source_display}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        ret, buffer = cv2.imencode('.jpg', img)
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
    print(f"✅ 追蹤已結束。來源: {current_source_display}")
    return

@app.route('/video_feed')
def video_feed():
    """MJPEG 串流路由"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/', methods=['GET', 'POST'])
def index():
    """主頁面路由，增加 POST 處理來更新來源"""
    global GLOBAL_VIDEO_SOURCE
    
    if request.method == 'POST':
        new_source = request.form.get('video_source_input', '').strip()
        
        if new_source:
            print(f"INFO: 收到新的影像來源: {new_source}")
            GLOBAL_VIDEO_SOURCE = new_source
        
        return redirect(url_for('index'))

    return render_template_string(LIVE_HTML, video_source=GLOBAL_VIDEO_SOURCE)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLO Live Stream Server.")
    parser.add_argument("--video", type=str, default=DEFAULT_VIDEO_SOURCE, help="影片來源 (檔案路徑, RTSP, URL, 0 for webcam)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="模型路徑")
    parser.add_argument("--port", type=int, default=5000, help="服務器端口")
    args = parser.parse_args()
    
    GLOBAL_VIDEO_SOURCE = args.video
    GLOBAL_MODEL_PATH = args.model
    
    print(f"🚀 Live Stream Dashboard on http://127.0.0.1:{args.port}/")
    print(f"INFO: 使用模型: {GLOBAL_MODEL_PATH}")
    app.run(host='0.0.0.0', port=args.port, debug=False)