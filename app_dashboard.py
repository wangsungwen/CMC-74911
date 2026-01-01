# -*- coding: utf-8 -*-
"""
YOLOv12 追蹤分析 — 多 Session 常駐儀表板
使用方式：
    python app_dashboard.py --base runs/analyze --port 5050
"""
import os, json, argparse
from pathlib import Path
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import plot
from flask import Flask, render_template_string, send_from_directory, request

parser = argparse.ArgumentParser()
parser.add_argument("--base", default="runs/analyze", help="分析輸出根目錄（包含多個 track_YYYYMMDD_HHMMSS）")
parser.add_argument("--port", type=int, default=5050)
args = parser.parse_args()

BASE = Path(args.base)
if not BASE.exists():
    print(f"❌ 找不到目錄：{BASE.resolve()}")
    exit(1)

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>YOLOv12 Multi-Session Dashboard</title>
<style>
body { font-family: 'Segoe UI', Arial; margin: 24px; }
select { padding: 6px; font-size: 16px; }
.card { border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 12px 0; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
h1, h2 { margin-top: 0; }
img { max-width: 100%; border-radius: 6px; }
.mono { font-family: Consolas, monospace; }
</style>
</head>
<body>
<h1>YOLOv12 Tracking Analytics — Multi-Session</h1>
<form method="get" action="/">
<b>選擇 Session：</b>
<select name="session" onchange="this.form.submit()">
{% for s in sessions %}
  <option value="{{ s }}" {% if s==selected %}selected{% endif %}>{{ s }}</option>
{% endfor %}
</select>
</form>

{% if summary %}
<div class="card mono">
  <b>Video:</b> {{ summary.video }} |
  <b>FPS:</b> {{ summary.fps }} |
  <b>Frames:</b> {{ summary.frames }} |
  <b>Total tracks:</b> {{ summary.tracks_total }} |
  <b>Avg lifespan (s):</b> {{ "%.2f"|format(summary.avg_lifespan_sec) }}
</div>

<div class="grid">
  <div class="card"><h2>Objects per Frame</h2><img src="/static/{{ selected }}/objects_per_frame.png"></div>
  <div class="card"><h2>Track Lifespan (seconds)</h2><img src="/static/{{ selected }}/track_lifespan_seconds.png"></div>
</div>
<div class="grid">
  <div class="card"><h2>Class Distribution</h2><img src="/static/{{ selected }}/class_distribution.png"></div>
  <div class="card"><h2>Average Dwell by Class</h2><img src="/static/{{ selected }}/avg_dwell_by_class.png"></div>
</div>

<div class="card">
  <h2>Interactive: Objects over Time</h2>
  {{ plots.objects|safe }}
</div>
<div class="card">
  <h2>Interactive: Class Distribution</h2>
  {{ plots.classes|safe }}
</div>
<div class="card">
  <h2>Interactive: Average Dwell</h2>
  {{ plots.dwell|safe }}
</div>

<div class="card">
  <h2>下載資料</h2>
  <ul>
    <li><a href="/static/{{ selected }}/summary.json">summary.json</a></li>
    <li><a href="/static/{{ selected }}/objects_per_frame.csv">objects_per_frame.csv</a></li>
    <li><a href="/static/{{ selected }}/track_lifespans.csv">track_lifespans.csv</a></li>
    <li><a href="/static/{{ selected }}/class_totals.csv">class_totals.csv</a></li>
    <li><a href="/static/{{ selected }}/avg_dwell_by_class.csv">avg_dwell_by_class.csv</a></li>
  </ul>
</div>
{% endif %}
</body></html>
"""

@app.route("/")
def index():
    sessions = [p.name for p in BASE.iterdir() if p.is_dir() and (p / "summary.json").exists()]
    sessions.sort(reverse=True)
    selected = (request.args.get("session") or (sessions[0] if sessions else None))
    summary = None
    plots = {"objects":"", "classes":"", "dwell":""}

    if selected:
        sess = BASE / selected
        try:
            with open(sess / "summary.json", "r", encoding="utf-8") as f:
                summary = json.load(f)
        except Exception as e:
            summary = None

        try:
            df_frame = pd.read_csv(sess / "objects_per_frame.csv")
            df_class = pd.read_csv(sess / "class_totals.csv")
            df_dwell = pd.read_csv(sess / "avg_dwell_by_class.csv")
        except:
            df_frame = df_class = df_dwell = None

        if df_frame is not None and not df_frame.empty:
            fig = go.Figure([go.Scatter(x=df_frame["frame"], y=df_frame["objects"], mode="lines")])
            fig.update_layout(title="Objects per Frame", xaxis_title="Frame", yaxis_title="Objects")
            plots["objects"] = plot(fig, include_plotlyjs="cdn", output_type="div")
        if df_class is not None and not df_class.empty:
            fig = go.Figure([go.Bar(x=df_class["class"], y=df_class["count"])])
            fig.update_layout(title="Class Distribution", xaxis_title="Class", yaxis_title="Count")
            plots["classes"] = plot(fig, include_plotlyjs=False, output_type="div")
        if df_dwell is not None and not df_dwell.empty:
            fig = go.Figure([go.Bar(x=df_dwell["cls_name"], y=df_dwell["avg_seconds"])])
            fig.update_layout(title="Average Dwell by Class", xaxis_title="Class", yaxis_title="Seconds")
            plots["dwell"] = plot(fig, include_plotlyjs=False, output_type="div")

    return render_template_string(HTML, sessions=sessions, selected=selected, summary=summary, plots=plots)

@app.route("/static/<path:subpath>")
def static_files(subpath):
    target = BASE / subpath
    return send_from_directory(str(target.parent), target.name)

if __name__ == "__main__":
    print(f"🚀 Dashboard on http://127.0.0.1:{args.port}/")
    app.run(host="127.0.0.1", port=args.port, debug=False)
