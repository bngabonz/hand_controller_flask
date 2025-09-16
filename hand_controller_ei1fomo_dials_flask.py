#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Copyright 2025 Tria Technologies Inc.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

# Hand Controller (Edge Impulse FOMO) as a Flask streamer (QCS6490)

app_name = "hand_controller_ei1fomo_dials"

import os
import sys
import cv2
import time
import glob
import re
import getpass
import socket
import argparse
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from timeit import default_timer as timer
from collections import deque
from typing import Tuple, Optional

from flask import Flask, Response, render_template_string

# project paths (keep your layout)
sys.path.append(os.path.abspath('blaze_app_python/'))
sys.path.append(os.path.abspath('blaze_app_python/blaze_common/'))

from visualization import draw_stacked_bar_chart, stacked_bar_performance_colors
from visualization import tria_blue, tria_yellow, tria_pink, tria_aqua
from utils_linux import get_media_dev_by_name, get_video_dev_by_name

from edge_impulse_linux.image import ImageImpulseRunner

user = getpass.getuser()
host = socket.gethostname()
print("[INFO] user@hostname:", f"{user}@{host}")

# -------------------- Flask --------------------
app = Flask(__name__)

# Shared state for MJPEG
_latest_frame = None
_last_lock_ts = 0.0

# -------------------- Args ---------------------
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

ap = argparse.ArgumentParser()
ap.add_argument('-i','--input', type=str, default="", help="Video input device or file. Default: auto USB cam")
ap.add_argument('-m','--model', type=str, default="./ei_fomo_face_hands_float32.eim", help="Path to Edge Impulse FOMO .eim")
ap.add_argument('-v','--verbose', default=False, action='store_true', help="Verbose logging")
ap.add_argument('-z','--profilelog', default=False, action='store_true', help="Enable latency CSV logging")
ap.add_argument('-y','--profileview', default=False, action='store_true', help="Overlay profiling charts into the stream")
ap.add_argument('-f','--fps', default=False, action='store_true', help="Overlay FPS text")
ap.add_argument('--host', default="0.0.0.0", help="Flask host")
ap.add_argument('--port', type=int, default=5002, help="Flask port")
ap.add_argument('--jpeg-quality', type=int, default=85, help="JPEG quality for MJPEG stream")
ARGS, _ = ap.parse_known_args()

# -------------------- Flags --------------------
bShowFPS     = ARGS.fps
bVerbose     = ARGS.verbose
bProfileLog  = ARGS.profilelog
bProfileView = ARGS.profileview

print("[INFO] Options:",
      "\n  --input       :", ARGS.input,
      "\n  --model       :", ARGS.model,
      "\n  --verbose     :", bVerbose,
      "\n  --profilelog  :", bProfileLog,
      "\n  --profileview :", bProfileView,
      "\n  --fps         :", bShowFPS,
      "\n  --host/port   :", f"{ARGS.host}:{ARGS.port}")

# -------------------- Profiling CSV --------------------
output_dir = './captured-images'
os.makedirs(output_dir, exist_ok=True)

profile_csv = f"./{app_name}_profiling.csv"
if os.path.isfile(profile_csv):
    f_profile_csv = open(profile_csv, "a")
    print("[INFO] Appending to existing profiling results file:", profile_csv)
else:
    f_profile_csv = open(profile_csv, "w")
    print("[INFO] Creating new profiling results file:", profile_csv)
    f_profile_csv.write("time,user,hostname,pipeline,detection-qty,resize,fomo_pre,fomo_model,fomo_post,annotate,dials,total,fps\n")

pipeline = app_name

# -------------------- Colors for charts --------------------
stacked_bar_latency_colors = [
    tria_blue,    # resize
    tria_yellow,  # fomo_pre
    tria_pink,    # fomo_model
    tria_aqua,    # fomo_post
    tria_blue,    # annotate
    tria_yellow,  # dials
]

# -------------------- Helpers --------------------
def resize_pad(img, h_scale, w_scale):
    """Resize and pad to square (160x160) keeping aspect ratio. Returns (img, scale, pad)."""
    size0 = img.shape
    if size0[0] >= size0[1]:
        h1 = int(h_scale)
        w1 = int(w_scale * size0[1] // size0[0])
        padh = 0
        padw = int(w_scale - w1)
        scale = size0[1] / w1
    else:
        h1 = int(h_scale * size0[0] // size0[1])
        w1 = int(w_scale)
        padh = int(h_scale - h1)
        padw = 0
        scale = size0[0] / h1
    padh1 = padh // 2
    padh2 = padh // 2 + padh % 2
    padw1 = padw // 2
    padw2 = padw // 2 + padw % 2
    img = cv2.resize(img, (w1, h1))
    img = np.pad(img, ((padh1, padh2), (padw1, padw2), (0, 0)), mode='constant')
    pad = (int(padh1 * scale), int(padw1 * scale))
    return img, scale, pad

@dataclass
class HandData:
    handedness: str
    landmarks: np.ndarray        # Nx3
    center_perc: Tuple[float,float,float]

    def __init__(self, handedness, landmarks, image_width, image_height):
        self.handedness = handedness
        self.landmarks = landmarks.copy()
        self.landmarks[:,0] = self.landmarks[:,0] / image_width
        self.landmarks[:,1] = self.landmarks[:,1] / image_height
        n = landmarks.shape[0]
        x_avg = float(np.mean(self.landmarks[:,0]))
        y_avg = float(np.mean(self.landmarks[:,1]))
        z_avg = float(np.mean(self.landmarks[:,2]))
        self.center_perc = (x_avg, y_avg, z_avg)

def draw_control_overlay(img, lh_data: Optional[HandData]=None, rh_data: Optional[HandData]=None):
    H, W, _ = img.shape
    CV_DRAW_COLOR_PRIMARY = tria_aqua
    CONTROL_CIRCLE_DEADZONE_R = 50
    # left control (XY)
    center_xy = (int(W/4), int(H/2))
    cv2.circle(img, center_xy, CONTROL_CIRCLE_DEADZONE_R, CV_DRAW_COLOR_PRIMARY, 2)
    hand_xy_point = center_xy
    if lh_data:
        x_norm = min((lh_data.center_perc[0] - 0.25) * 4, 1.0)
        y_norm = min((lh_data.center_perc[1] - 0.5) * 2, 1.0)
        hand_xy_point = (int(x_norm * CONTROL_CIRCLE_DEADZONE_R) + center_xy[0],
                         int(y_norm * CONTROL_CIRCLE_DEADZONE_R) + center_xy[1])
        cv2.line(img, center_xy, hand_xy_point, CV_DRAW_COLOR_PRIMARY, 1)
        cv2.circle(img, hand_xy_point, 4, CV_DRAW_COLOR_PRIMARY, cv2.FILLED)
    delta_xy = tuple(c/CONTROL_CIRCLE_DEADZONE_R for c in np.subtract(center_xy, hand_xy_point))
    # right control (Z/aperture)
    center_z = (int(3*W/4), int(H/2))
    cv2.circle(img, center_z, CONTROL_CIRCLE_DEADZONE_R, CV_DRAW_COLOR_PRIMARY, 2)
    hand_z_point = center_z
    if rh_data:
        z_norm = min((rh_data.center_perc[1] - 0.50) * 2, 1.0)
        a_norm = min((rh_data.center_perc[0] - 0.75) * 4, 1.0)
        hand_z_point = (int(a_norm * CONTROL_CIRCLE_DEADZONE_R) + center_z[0],
                        int(z_norm * CONTROL_CIRCLE_DEADZONE_R) + center_z[1])
        cv2.line(img, center_z, hand_z_point, CV_DRAW_COLOR_PRIMARY, 1)
        cv2.circle(img, hand_z_point, 4, CV_DRAW_COLOR_PRIMARY, cv2.FILLED)
    delta_z = tuple(c/CONTROL_CIRCLE_DEADZONE_R for c in np.subtract(center_z, hand_z_point))
    # midline
    cv2.line(img, (int(W/2), 0), (int(W/2), H), CV_DRAW_COLOR_PRIMARY, 1)
    return delta_xy, delta_z

# -------------------- Input setup --------------------
def open_input():
    bInputImage = bInputVideo = False
    bInputCamera = True
    input_src = ARGS.input

    if os.path.exists(input_src):
        file_ext = os.path.splitext(input_src)[1].lower()
        if file_ext in (".jpg",".jpeg",".png",".tif",".tiff"):
            bInputImage, bInputVideo, bInputCamera = True, False, False
        elif file_ext in (".mov",".mp4",".avi",".mkv"):
            bInputImage, bInputVideo, bInputCamera = False, True, False

    if bInputCamera:
        print("[INFO] Searching for USB camera ...")
        dev_video = get_video_dev_by_name("uvcvideo")
        dev_media = get_media_dev_by_name("uvcvideo")
        print("[INFO] video node:", dev_video, " media node:", dev_media)
        if dev_video is None:
            input_video = 0
        elif input_src != "":
            input_video = input_src
        else:
            input_video = dev_video
        cap = cv2.VideoCapture(input_video, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        print(f"[INFO] input: camera {input_video} ({CAMERA_WIDTH},{CAMERA_HEIGHT})")
        return "camera", cap, None
    elif bInputVideo:
        cap = cv2.VideoCapture(input_src)
        w = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        h = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print(f"[INFO] input: video {input_src} ({w},{h})")
        return "video", cap, None
    else:
        img = cv2.imread(input_src)
        h,w,_ = img.shape
        print(f"[INFO] input: image {input_src} ({w},{h})")
        return "image", None, img

# -------------------- Processing loop --------------------
def processing_worker():
    global _latest_frame, _last_lock_ts
    # status/console banner
    print("="*64)
    print("Hand Controller (Edge Impulse) with Dials — Flask streamer")
    print("="*64)

    inp_kind, cap, img_static = open_input()

    # FOMO runner
    modelfile = ARGS.model
    if bVerbose:
        print("[INFO] model:", modelfile)

    # Counters
    frame_count = 0
    rt_fps_count = 0
    rt_fps_time = cv2.getTickCount()
    rt_fps_valid = False
    rt_fps = 0.0

    profile_latency_title = "Latency (sec)"
    profile_performance_title = "Performance (FPS)"

    with ImageImpulseRunner(modelfile) as runner:
        try:
            model_info = runner.init()
            if bVerbose:
                print('[INFO] Loaded runner for "%s / %s"' % (model_info['project']['owner'], model_info['project']['name']))
                print("[INFO] labels:", model_info['model_parameters']['labels'])

            while True:
                # --- Grab frame ---
                if inp_kind == "image":
                    frame = img_static.copy()
                else:
                    ok, frame = cap.read()
                    if not ok:
                        print("[WARN] cap.read() failed; retrying in 50ms")
                        time.sleep(0.05)
                        continue

                # mirror selfie-style (matching your interactive app)
                frame = cv2.flip(frame, 1)

                image = frame
                output = image.copy()

                # Profiling buckets (per-frame)
                profile_resize = 0.0
                profile_fomo_qty = 0
                profile_fomo_pre = 0.0
                profile_fomo_model = 0.0
                profile_fomo_post = 0.0
                profile_annotate = 0.0
                profile_dials = 0.0

                # real-time FPS (like original, every 10 frames)
                if rt_fps_count == 0:
                    rt_fps_time = cv2.getTickCount()

                # ---- FOMO pipeline (unchanged semantics) ----
                # resize & pad to 160x160 RGB for EI
                start = timer()
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_size = max(CAMERA_WIDTH, CAMERA_HEIGHT)
                cropped_size = 160
                img1, scale1, pad1 = resize_pad(rgb, cropped_size, cropped_size)
                profile_resize = timer() - start

                # pre: extract features from image for EI
                start = timer()
                features, cropped = runner.get_features_from_image(img1)  # <-- avoids SHM channel mismatch
                cropped = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
                profile_fomo_pre = timer() - start

                # model classify
                start = timer()
                res = runner.classify(features)
                profile_fomo_model = timer() - start

                # post: parse bboxes and build dials inputs
                start = timer()
                lh_data, rh_data = None, None
                if "result" in res and "bounding_boxes" in res["result"]:
                    bbs = res["result"]["bounding_boxes"]
                    profile_fomo_qty = len(bbs)
                    for bb in bbs:
                        label = bb.get('label', '')
                        if label == 'face':
                            continue
                        x = bb.get('x', 0)
                        y = bb.get('y', 0)
                        w = bb.get('width', 0)
                        h = bb.get('height', 0)

                        # visualize on the cropped 160x160 (debug-ish)
                        color = (0,255,0) if label == 'open' else (0,0,255)
                        cv2.rectangle(cropped, (x,y), (x+w,y+h), color, 2)

                        # map back to original coords using pad/scale
                        x1 = (((x) / cropped_size) * image_size) - pad1[1]
                        y1 = (((y) / cropped_size) * image_size) - pad1[0]
                        x2 = (((x + w) / cropped_size) * image_size) - pad1[1]
                        y2 = (((y + h) / cropped_size) * image_size) - pad1[0]
                        z1 = 0.0; z2 = 0.0
                        landmarks = np.asarray([[x1,y1,z1],[x2,y2,z2]], dtype=np.float32)
                        handedness = "Left"

                        if x1 < CAMERA_WIDTH/2:
                            lh_data = HandData(handedness, landmarks, CAMERA_WIDTH, CAMERA_HEIGHT)
                        else:
                            rh_data = HandData(handedness, landmarks, CAMERA_WIDTH, CAMERA_HEIGHT)

                        # draw on full frame
                        if label == 'open':
                            cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                        elif label == 'closed':
                            cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)

                profile_fomo_post = timer() - start

                # Dials overlay
                start = timer()
                if lh_data is not None:
                    cv2.circle(output, (int(lh_data.center_perc[0]*CAMERA_WIDTH),
                                        int(lh_data.center_perc[1]*CAMERA_HEIGHT)), 10, tria_pink, -1)
                if rh_data is not None:
                    cv2.circle(output, (int(rh_data.center_perc[0]*CAMERA_WIDTH),
                                        int(rh_data.center_perc[1]*CAMERA_HEIGHT)), 10, tria_pink, -1)
                delta_xy, delta_z = draw_control_overlay(output, lh_data, rh_data)
                profile_dials += timer() - start

                # FPS overlay (text)
                if rt_fps_valid and bShowFPS:
                    cv2.putText(output, f"FPS: {rt_fps:.2f}", (10, CAMERA_HEIGHT-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2, cv2.LINE_AA)

                # Profiling summary and optional CSV logging
                profile_fomo = profile_fomo_pre + profile_fomo_model + profile_fomo_post
                profile_total = profile_resize + profile_fomo + profile_annotate + profile_dials
                profile_fps = (1.0 / profile_total) if profile_total > 0 else 0.0

                if bProfileLog:
                    timestamp = datetime.now()
                    csv_str = ",".join([
                        str(timestamp), str(user), str(host), pipeline,
                        str(profile_fomo_qty),
                        str(profile_resize),
                        str(profile_fomo_pre),
                        str(profile_fomo_model),
                        str(profile_fomo_post),
                        str(profile_annotate),
                        str(profile_dials),
                        str(profile_total),
                        str(profile_fps)
                    ]) + "\n"
                    f_profile_csv.write(csv_str)

                # Optional: overlay latency/FPS charts into the frame (bottom-left)
                if bProfileView:
                    component_labels = ["resize","fomo[pre]","fomo[model]","fomo[post]","annotate","dials"]
                    pipeline_titles = [app_name]
                    component_values = [[profile_resize],
                                        [profile_fomo_pre],
                                        [profile_fomo_model],
                                        [profile_fomo_post],
                                        [profile_annotate],
                                        [profile_dials]]
                    chart = draw_stacked_bar_chart(
                        pipeline_titles=pipeline_titles,
                        component_labels=component_labels,
                        component_values=component_values,
                        component_colors=stacked_bar_latency_colors,
                        chart_name="Latency (sec)"
                    )
                    # shrink and blend
                    ch = 220
                    cw = int(chart.shape[1] * (ch / chart.shape[0]))
                    chart_small = cv2.resize(chart, (cw, ch), interpolation=cv2.INTER_AREA)
                    H, W = output.shape[:2]
                    x0, y0 = 10, H - ch - 10
                    x1, y1 = x0 + cw, y0 + ch
                    if x1 <= W and y1 <= H:
                        roi = output[y0:y1, x0:x1]
                        cv2.addWeighted(chart_small, 0.82, roi, 0.18, 0, dst=roi)

                # Update FPS (every 10 frames like original)
                rt_fps_count += 1
                if rt_fps_count == 10:
                    t = (cv2.getTickCount() - rt_fps_time)/cv2.getTickFrequency()
                    rt_fps_valid = True
                    rt_fps = 10.0/t
                    rt_fps_count = 0

                # Debug: print dials
                print(f"[INFO] DIALS XY={delta_xy[0]:+.3f}|{delta_xy[1]:+.3f}, Z={delta_z[0]:+.3f}|{delta_z[1]:+.3f}")

                # --- publish to MJPEG ---
                ok, buf = cv2.imencode(".jpg", output,
                                       [int(cv2.IMWRITE_JPEG_QUALITY), int(ARGS.jpeg_quality)])
                if ok:
                    _latest_frame = buf.tobytes()
                    _last_lock_ts = time.time()
                else:
                    # if encode fails, skip this frame
                    pass

        finally:
            try:
                if 'runner' in locals() and runner:
                    runner.stop()
            except Exception:
                pass

# -------------------- Flask endpoints --------------------
INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Hand Controller Stream (QCS6490)</title>
  <style>
    body { background:#111; color:#eee; font-family:system-ui,Arial,sans-serif; }
    .wrap { max-width: 1100px; margin: 24px auto; }
    .card { background:#1b1b1b; border-radius:14px; padding:16px; box-shadow:0 4px 18px rgba(0,0,0,.35); }
    img { width:100%; height:auto; border-radius:10px; }
    .meta { font-size: 14px; opacity:.8; margin-top: 8px; }
    code { background:#222; padding:2px 6px; border-radius:6px; }
    .pill { display:inline-block; padding:4px 10px; margin-right:8px; border-radius:999px; background:#2a2a2a; }
  </style>
</head>
<body>
  <div class="wrap">
    <h2>Hand Controller Stream (Edge Impulse FOMO)</h2>
    <div class="card">
      <img src="/video_feed" alt="live stream"/>
      <div class="meta">
        <span class="pill">Model: {{ model }}</span>
        <span class="pill">Input: {{ inp }}</span>
        <span class="pill">JPEG: {{ jpegq }}</span>
      </div>
      <p>Open this page from your Linux host: <code>http://KIT_IP:{{ port }}/</code></p>
      <p>Raw MJPEG endpoint: <a href="/video_feed">/video_feed</a></p>
    </div>
  </div>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(INDEX_HTML,
        model=os.path.basename(ARGS.model),
        inp=ARGS.input or "auto (USB cam)",
        jpegq=ARGS.jpeg_quality,
        port=ARGS.port
    )

@app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            if _latest_frame is None:
                time.sleep(0.01)
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + _latest_frame + b"\r\n")
            time.sleep(0.001)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/healthz")
def healthz():
    return "ok", 200

# -------------------- Main --------------------
if __name__ == "__main__":
    # Start background processing thread
    import threading
    t = threading.Thread(target=processing_worker, daemon=True)
    t.start()

    print(f"* Flask running on http://0.0.0.0:{ARGS.port} (open from your host: http://KIT_IP:{ARGS.port})")
    app.run(host=ARGS.host, port=ARGS.port, threaded=True, debug=False)
