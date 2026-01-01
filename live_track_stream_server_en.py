# live_stream_server.py (Final Fix - English Interface with Track History)
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
    # This remains as a fail-safe message for users
    print("FATAL ERROR: yt-dlp library not found. YouTube streaming requires: pip install yt-dlp")

def get_youtube_stream_url(video_url):
    """Uses yt-dlp API to extract the best direct video stream URL."""
    if 'youtube.com' not in video_url and 'youtu.be' not in video_url:
        return video_url # Not a YouTube URL, return as is

    print(f"INFO: Extracting direct stream URL for YouTube video: {video_url}...")
    try:
        ydl_opts = {
            # Select the best mp4 format with both video and audio
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': '%(title)s.%(ext)s',
            'quiet': True,
            'skip_download': True,
            'logtostderr': False,
            'no_warnings': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract info without downloading
            info = ydl.extract_info(video_url, download=False)
            
            formats = info.get('formats', [])
            
            # Find the best mp4 stream URL for OpenCV
            best_url = None
            for f in formats:
                if f.get('ext') == 'mp4' and f.get('vcodec') != 'none' and f.get('url'):
                    best_url = f['url']
                    break

            if best_url:
                return best_url

            # Fallback to the general URL
            return info.get('url')
            
    except Exception as e:
        print(f"FATAL ERROR: yt-dlp extraction failed: {e}")
        traceback.print_exc()
        return None
# ===================================================================


# Default paths (if no command line args are provided)
DEFAULT_MODEL_PATH = r"C:\Users\wangs\monkeyv7\runs\train\dawn_semi_round3\weights\best.pt"
DEFAULT_VIDEO_SOURCE = "wildlife.mp4" 
GLOBAL_MODEL_PATH = DEFAULT_MODEL_PATH
GLOBAL_VIDEO_SOURCE = DEFAULT_VIDEO_SOURCE

# --- Flask App Configuration ---
app = Flask(__name__)

# LIVE_HTML (Flask Template Variable)
LIVE_HTML = """
<!DOCTYPE html>
<html>
<head>
<title>YOLOv12 Real-time Tracking Stream</title>
<style>body{font-family:Segoe UI,Arial;text-align:center;} img{max-width: 90%;}</style>
</head>
<body>
<h1>Real-time Inference & Tracking (Web Stream)</h1>
<p>Source: <b>{{ video_source }}</b></p>
<img id="video-stream" src="{{ url_for('video_feed') }}" width="100%">
<p>
    {% if 'youtube.com' in video_source or 'youtu.be' in video_source %}
    * Network stream is running. Press CTRL+C in the terminal to stop.
    {% else %}
    * File video is running. The stream will end automatically when the video finishes.
    {% endif %}
</p>
<hr>
<h2><a href="http://127.0.0.1:5050/">Click here for Batch Analysis Dashboard (if run_infer_track_analyze_dashboard.ps1 was executed)</a></h2>
</body>
</html>
"""

def generate_frames():
    """Performs real-time tracking using YOLOv12 and encodes results for MJPEG stream."""
    
    current_source = GLOBAL_VIDEO_SOURCE
    
    # Attempt to get the raw stream URL for YouTube sources
    if 'youtube.com' in GLOBAL_VIDEO_SOURCE or 'youtu.be' in GLOBAL_VIDEO_SOURCE:
        stream_url = get_youtube_stream_url(GLOBAL_VIDEO_SOURCE)
        if stream_url:
            current_source = stream_url
            print(f"INFO: Successfully extracted direct stream URL.")
        else:
            print("FATAL: Failed to extract YouTube stream URL. Check yt-dlp logs.")
            current_source = None # Set to None to exit immediately
            
    if not current_source:
        # If stream acquisition fails, yield an error image and exit
        img = np.zeros((480, 640, 3), dtype="uint8")
        cv2.putText(img, "STREAM EXTRACTION FAILED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', img)
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        print(f"✅ Tracking finished. Source: {GLOBAL_VIDEO_SOURCE}")
        return # Exit the generator gracefully

    print(f"INFO: Starting tracking on source: {current_source}")
    
    # --- Track History Initialization ---
    track_history = {}
    MAX_HISTORY_POINTS = 30 # 歷史足跡點數
    TRACK_COLOR = (0, 255, 255) # 黃色 (BGR 格式)
    # ------------------------------------

    try:
        model = YOLO(GLOBAL_MODEL_PATH) 
        # Pass the direct stream URL to model.track()
        results = model.track(source=current_source, stream=True, show=False, 
                              tracker="bytetrack.yaml", imgsz=1280, conf=0.45)

        for r in results:
            if r is None:
                continue
            
            # --- 獲取推論繪圖結果 ---
            annotated_frame = r.plot()
            
            # --- 軌跡繪製邏輯 ---
            if r.boxes.id is not None:
                boxes = r.boxes.xyxy.cpu().numpy().astype(int)
                track_ids = r.boxes.id.cpu().numpy().astype(int)

                # 1. 更新歷史中心點
                current_track_ids = set()
                for box, track_id in zip(boxes, track_ids):
                    current_track_ids.add(track_id)
                    center_x, center_y = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
                    
                    if track_id not in track_history:
                        track_history[track_id] = []
                    
                    track_history[track_id].append((center_x, center_y))
                    
                    # 限制歷史長度為 MAX_HISTORY_POINTS
                    if len(track_history[track_id]) > MAX_HISTORY_POINTS:
                        track_history[track_id].pop(0)
                
                # 2. 移除當前幀未出現的軌跡（可選，但保持字典清潔）
                # for track_id in list(track_history.keys()):
                #     if track_id not in current_track_ids and len(track_history[track_id]) > 0:
                #         track_history[track_id].pop(0) # 讓它在幾幀內自然消失

                # 3. 繪製歷史軌跡
                for track_id, points in track_history.items():
                    if len(points) < 2:
                        continue
                        
                    # 繪製歷史足跡（深淺粗細線條）
                    for i in range(1, len(points)):
                        # 計算顏色和粗細，實現深淺粗細變化
                        alpha = i / len(points) # 越接近 1 (越近)，顏色越亮/粗
                        thickness = int(1 + alpha * 3) # 粗細變化: 1 到 4
                        
                        # 黃色漸變 (利用 alpha 調整亮度/深淺)
                        # BGR: (Blue, Green, Red)
                        # 顏色從暗黃 (遠) 漸變到亮黃 (近)
                        blue_value = int(TRACK_COLOR[0] * alpha)
                        green_value = int(TRACK_COLOR[1] * alpha)
                        red_value = int(TRACK_COLOR[2] * alpha)
                        
                        current_color = (blue_value, green_value, red_value)
                        
                        cv2.line(annotated_frame, points[i-1], points[i], current_color, thickness)

            # --- 輸出幀 ---
            frame_to_encode = annotated_frame
            ret, buffer = cv2.imencode('.jpg', frame_to_encode)
            # ---------------------
            
            if not ret: 
                print("⚠️ Frame encoding failed, skipping.")
                continue
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
    except Exception as e:
        error_msg = f"❌ Fatal Error in generate_frames: {e}"
        print(error_msg)
        traceback.print_exc()
        
        # Draw error message to screen
        img = np.zeros((480, 640, 3), dtype="uint8") 
        cv2.putText(img, "STREAM ERROR", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, str(e)[:60], (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        ret, buffer = cv2.imencode('.jpg', img)
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
    print(f"✅ Tracking finished. Source: {GLOBAL_VIDEO_SOURCE}")
    return

@app.route('/video_feed')
def video_feed():
    """MJPEG stream route"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Main dashboard route"""
    return render_template_string(LIVE_HTML, video_source=GLOBAL_VIDEO_SOURCE)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLOv12 Live Stream Server.")
    parser.add_argument("--video", type=str, default=DEFAULT_VIDEO_SOURCE, help="Video source (file path, RTSP, URL, 0 for webcam)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Model path")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    args = parser.parse_args()
    
    GLOBAL_VIDEO_SOURCE = args.video
    GLOBAL_MODEL_PATH = args.model
    
    print(f"🚀 Live Stream Dashboard on http://127.0.0.1:{args.port}/")
    app.run(host='0.0.0.0', port=args.port, debug=False)