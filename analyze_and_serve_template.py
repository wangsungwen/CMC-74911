import os, sys, json, csv
from pathlib import Path
import yaml
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from flask import Flask, send_from_directory, render_template_string
import plotly.graph_objs as go
from plotly.offline import plot

labels_dir = Path(r"__LABEL_DIR__")
# '__VIDEO__' 可能是本地路徑或 URL/串流
video_source = r"__VIDEO__"
out_dir    = Path(r"__SESSION_DIR__")
out_dir.mkdir(parents=True, exist_ok=True)

names = {}
yaml_path = Path(r"__YAML__") if r"__YAML__" else None
if yaml_path and yaml_path.exists():
    with open(yaml_path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    raw = d.get("names", {})
    if isinstance(raw, dict):
        names = {int(k): str(v) for k, v in raw.items()}
    elif isinstance(raw, list):
        names = {i: str(v) for i, v in enumerate(raw)}
else:
    names = {0:"cls0",1:"cls0",2:"cls0"} # 修正原始模板中的錯誤，確保至少有 cls0

# 嘗試使用 OpenCV 讀取 FPS 和總幀數
cap = cv2.VideoCapture(video_source)
# 如果無法開啟 (例如純 URL 或斷線的串流)，則使用預設值
fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() and cap.get(cv2.CAP_PROP_FPS) > 0 else 30.0
frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None
cap.release()

rows = []
for txt in sorted(labels_dir.glob("*.txt"), key=lambda p: p.name):
    stem = txt.stem
    try:
        frame_index = int(float(stem))
    except:
        frame_index = stem
    for ln in txt.read_text(encoding="utf-8").splitlines():
        p = ln.strip().split()
        if len(p) >= 7:
            cid, xc, yc, w, h, conf, tid = p[:7]
            try:
                cid_i = int(float(cid)); tid_i = int(float(tid)); conf_f = float(conf)
            except:
                continue
            rows.append({
                "frame": frame_index,
                "cls": cid_i,
                "cls_name": names.get(cid_i, f"cls{cid_i}"),
                "conf": conf_f,
                "track_id": tid_i
            })

if not rows:
    print("No tracking data parsed."); sys.exit(0)

import pandas as pd
df = pd.DataFrame(rows)

# 確保 'frame' 是數值類型，以便繪圖/計算
df['frame'] = pd.to_numeric(df['frame'], errors='coerce').fillna(0).astype(int)

frame_counts = df.groupby("frame")["track_id"].nunique().rename("objects").reset_index()

life = df.groupby("track_id").size().rename("frames").reset_index()
life["seconds"] = life["frames"] / (fps if fps else 30.0)

mode_cls = (df.groupby(["track_id","cls_name"]).size().rename("n").reset_index()
              .sort_values(["track_id","n"], ascending=[True,False])
              .drop_duplicates("track_id"))
life = life.merge(mode_cls[["track_id","cls_name"]], on="track_id", how="left")

class_totals = df["cls_name"].value_counts().rename_axis("class").reset_index(name="count")
avg_dwell = life.groupby("cls_name")["seconds"].mean().rename("avg_seconds").reset_index()

frame_counts.to_csv(out_dir / "objects_per_frame.csv", index=False)
life.to_csv(out_dir / "track_lifespans.csv", index=False)
class_totals.to_csv(out_dir / "class_totals.csv", index=False)
avg_dwell.to_csv(out_dir / "avg_dwell_by_class.csv", index=False)

plt.figure(figsize=(10,4)); plt.plot(frame_counts["frame"], frame_counts["objects"]); plt.title("Objects per Frame"); plt.tight_layout(); plt.savefig(out_dir/"objects_per_frame.png", dpi=150); plt.close()
plt.figure(figsize=(6,4)); plt.hist(life["seconds"], bins=20); plt.title("Track Lifespan (seconds)"); plt.tight_layout(); plt.savefig(out_dir/"track_lifespan_seconds.png", dpi=150); plt.close()
plt.figure(figsize=(7,4)); plt.bar(class_totals["class"], class_totals["count"]); plt.xticks(rotation=45, ha="right"); plt.title("Class Distribution"); plt.tight_layout(); plt.savefig(out_dir/"class_distribution.png", dpi=150); plt.close()
plt.figure(figsize=(7,4)); plt.bar(avg_dwell["cls_name"], avg_dwell["avg_seconds"]); plt.xticks(rotation=45, ha="right"); plt.title("Average Dwell by Class"); plt.tight_layout(); plt.savefig(out_dir/"avg_dwell_by_class.png", dpi=150); plt.close()

# 影片名稱處理：如果是本地檔案，則取名稱；如果是 URL/Stream 則顯示完整來源
video_display_name = video_source
try:
    path_obj = Path(video_source)
    # 判斷是否為本地檔案，包含有副檔名且不是常見的串流協議開頭
    if path_obj.suffix and not video_source.lower().startswith(("rtsp://", "http://", "https://")):
        video_display_name = path_obj.name
except:
    # URL/串流
    pass

summary = {
    "video": video_display_name,
    "fps": fps,
    "frames": frames_total if frames_total is not None else "N/A", # 可能是串流或無法讀取
    "tracks_total": int(life.shape[0]),
    "avg_lifespan_sec": float(life["seconds"].mean()),
    "objects_total": int(df.shape[0])
}
(out_dir/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

def fig_objects_over_time():
    return go.Figure([go.Scatter(x=frame_counts["frame"], y=frame_counts["objects"], mode="lines")])
def fig_class_distribution():
    return go.Figure([go.Bar(x=class_totals["class"], y=class_totals["count"])])
def fig_avg_dwell():
    return go.Figure([go.Bar(x=avg_dwell["cls_name"], y=avg_dwell["avg_seconds"])])

from flask import Flask, render_template_string, send_from_directory
from plotly.offline import plot

app = Flask(__name__, static_folder=str(out_dir))

INDEX = """
<!doctype html><html><head><meta charset="utf-8"><title>YOLOv12 追蹤分析儀表板</title>
<style>body{font-family:Segoe UI,Arial;margin:24px}.card{border:1px solid #ddd;border-radius:8px;padding:16px;margin:12px 0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.mono{font-family:Consolas,monospace}img{max-width:100%}</style>
</head><body>
<h1>YOLOv12 追蹤分析儀表板</h1>
<div class="card mono">
<b>Video:</b> {{ summary.video }} &nbsp; <b>FPS:</b> {{ summary.fps }} &nbsp; <b>Frames:</b> {{ summary.frames }} &nbsp; <b>Total tracks:</b> {{ summary.tracks_total }} &nbsp; <b>Avg lifespan (s):</b> {{ '%.2f'|format(summary.avg_lifespan_sec) }}
</div>
<div class="grid">
  <div class="card"><h2>Objects per Frame</h2><img src="objects_per_frame.png"></div>
  <div class="card"><h2>Track Lifespan (seconds)</h2><img src="track_lifespan_seconds.png"></div>
</div>
<div class="grid">
  <div class="card"><h2>Class Distribution</h2><img src="class_distribution.png"></div>
  <div class="card"><h2>Average Dwell by Class</h2><img src="avg_dwell_by_class.png"></div>
</div>
<div class="card"><h2>Interactive: Objects over Time</h2>{{ plot_objects|safe }}</div>
<div class="card"><h2>Interactive: Class Distribution</h2>{{ plot_classes|safe }}</div>
<div class="card"><h2>Interactive: Avg Dwell</h2>{{ plot_dwell|safe }}</div>
<div class="card"><h2>下載資料</h2>
<ul>
<li><a href="summary.json">summary.json</a></li>
<li><a href="objects_per_frame.csv">objects_per_frame.csv</a></li>
<li><a href="track_lifespans.csv">track_lifespans.csv</a></li>
<li><a href="class_totals.csv">class_totals.csv</a></li>
<li><a href="avg_dwell_by_class.csv">avg_dwell_by_class.csv</a></li>
</ul>
</div>
</body></html>
"""

@app.route("/")
def index():
    with open(out_dir/"summary.json","r",encoding="utf-8") as f:
        summary = json.load(f)
    po = plot(fig_objects_over_time(), include_plotlyjs="cdn", output_type="div")
    pc = plot(fig_class_distribution(), include_plotlyjs=False, output_type="div")
    pdw= plot(fig_avg_dwell(), include_plotlyjs=False, output_type="div")
    return render_template_string(INDEX, summary=summary, plot_objects=po, plot_classes=pc, plot_dwell=pdw)

@app.route("/<path:path>")
def static_proxy(path):
    return send_from_directory(str(out_dir), path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5050)
    args = parser.parse_args()
    print("Dashboard ready.")
    app.run(port=args.port)
