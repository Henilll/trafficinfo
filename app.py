# # app.py
# import streamlit as st
# import cv2
# import numpy as np
# from ultralytics import YOLO
# from sort import Sort
# import threading
# import time

# st.set_page_config(page_title="Vehicle Counter Dashboard", layout="wide")

# # --------------------
# # SETTINGS
# # --------------------
# STREAM_URL = "http://109.206.96.58:8080/cam_1.cgi"
# YOLO_MODEL = "yolov8n.pt"
# FRAME_WIDTH = 640
# FRAME_HEIGHT = 360
# VEHICLE_CLASSES = [2, 3, 5, 7]

# # --------------------
# # GLOBAL VARIABLES
# # --------------------
# tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
# total_count = 0
# in_count = 0
# out_count = 0
# track_memory = set()
# frame_lock = threading.Lock()
# display_frame = None

# # --------------------
# # HELPER FUNCTIONS
# # --------------------
# def get_stream(url):
#     cap = cv2.VideoCapture(url)
#     while not cap.isOpened():
#         print("❌ Cannot open IP Camera stream. Retrying in 2s...")
#         cap.release()
#         time.sleep(2)
#         cap = cv2.VideoCapture(url)
#     return cap

# def process_frame(frame):
#     global total_count, in_count, out_count, track_memory
#     frame_resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
#     results = model(frame_resized, conf=0.4)[0]

#     detections = []
#     for box in results.boxes.data:
#         x1, y1, x2, y2, score, cls = box.tolist()
#         scale_x = frame.shape[1] / FRAME_WIDTH
#         scale_y = frame.shape[0] / FRAME_HEIGHT
#         x1 *= scale_x
#         x2 *= scale_x
#         y1 *= scale_y
#         y2 *= scale_y

#         if int(cls) in VEHICLE_CLASSES:
#             detections.append([x1, y1, x2, y2, score])

#     detections = np.array(detections)
#     tracked_objects = tracker.update(detections)

#     line_y = frame.shape[0] // 2
#     cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 2)

#     for x1, y1, x2, y2, track_id in tracked_objects:
#         x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
#         centroid_y = (y1 + y2) // 2

#         if int(track_id) not in track_memory:
#             track_memory.add(int(track_id))
#             total_count += 1
#             if centroid_y < line_y:
#                 in_count += 1
#             else:
#                 out_count += 1

#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(frame, f"ID {int(track_id)}", (x1, y1 - 5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#     cv2.putText(frame, f"Total: {total_count}", (20, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
#     cv2.putText(frame, f"IN: {in_count}", (20, 80),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
#     cv2.putText(frame, f"OUT: {out_count}", (20, 120),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

#     return frame

# # --------------------
# # VIDEO THREAD
# # --------------------
# def video_loop():
#     global display_frame
#     cap = get_stream(STREAM_URL)
#     while True:
#         ret, frame = cap.read()
#         if not ret or frame is None:
#             cap.release()
#             cap = get_stream(STREAM_URL)
#             continue
#         frame = process_frame(frame)
#         with frame_lock:
#             display_frame = frame.copy()

# # --------------------
# # LOAD YOLO
# # --------------------
# model = YOLO(YOLO_MODEL)

# # Start video thread
# thread = threading.Thread(target=video_loop, daemon=True)
# thread.start()

# # --------------------
# # STREAMLIT DASHBOARD
# # --------------------
# st.title("🚗 Vehicle Counter Dashboard")
# st.markdown("Live IN/OUT counting with YOLOv8 + SORT tracking")

# col1, col2 = st.columns([3,1])
# frame_placeholder = col1.empty()

# # Use empty placeholders for metrics
# total_placeholder = col2.empty()
# in_placeholder = col2.empty()
# out_placeholder = col2.empty()

# # Streamlit refresh loop
# refresh_rate = 100  # milliseconds

# while True:
#     with frame_lock:
#         if display_frame is not None:
#             frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
#             frame_placeholder.image(frame_rgb, channels="RGB")

#     total_placeholder.metric("Total Vehicles", total_count)
#     in_placeholder.metric("Vehicles IN", in_count)
#     out_placeholder.metric("Vehicles OUT", out_count)

#     time.sleep(refresh_rate / 1000)  # convert ms to seconds
# app.py
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort
import threading
import time
import pandas as pd

st.set_page_config(page_title="TraffiSight Dashboard", layout="wide")

# --------------------
# SETTINGS
# --------------------
STREAM_URL = "http://109.206.96.58:8080/cam_1.cgi"
YOLO_MODEL = "yolov8n.pt"
FRAME_WIDTH = 640
FRAME_HEIGHT = 360
VEHICLE_CLASSES = {2:"Car", 3:"Motorbike", 5:"Bus", 7:"Truck"}

# --------------------
# GLOBAL VARIABLES
# --------------------
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
total_count = 0
in_count = 0
out_count = 0
class_counts = {name:0 for name in VEHICLE_CLASSES.values()}
track_memory = set()
frame_lock = threading.Lock()
display_frame = None

# Store historical data for trend charts
history_len = 50
total_history = []
in_history = []
out_history = []

# --------------------
# HELPER FUNCTIONS
# --------------------
def get_stream(url):
    cap = cv2.VideoCapture(url)
    while not cap.isOpened():
        print("❌ Cannot open IP Camera stream. Retrying in 2s...")
        cap.release()
        time.sleep(2)
        cap = cv2.VideoCapture(url)
    return cap

def process_frame(frame):
    global total_count, in_count, out_count, track_memory, class_counts

    frame_resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    results = model(frame_resized, conf=0.4)[0]

    detections = []
    for box in results.boxes.data:
        x1, y1, x2, y2, score, cls = box.tolist()
        scale_x = frame.shape[1] / FRAME_WIDTH
        scale_y = frame.shape[0] / FRAME_HEIGHT
        x1 *= scale_x; x2 *= scale_x; y1 *= scale_y; y2 *= scale_y

        if int(cls) in VEHICLE_CLASSES:
            detections.append([x1, y1, x2, y2, score])

    detections = np.array(detections)
    tracked_objects = tracker.update(detections)

    # Virtual IN/OUT line
    line_y = frame.shape[0] // 2
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 2)

    # Reset per-frame class counts
    frame_class_counts = {name:0 for name in VEHICLE_CLASSES.values()}

    for x1, y1, x2, y2, track_id in tracked_objects:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        centroid_y = (y1 + y2) // 2

        # Find class of this box
        cls_name = None
        for box in results.boxes.data:
            _, by1, _, by2, _, cls = box.tolist()
            if abs(centroid_y - ((by1+by2)/2)) < 20:
                cls_name = VEHICLE_CLASSES[int(cls)]
                break

        if int(track_id) not in track_memory:
            track_memory.add(int(track_id))
            total_count += 1
            if centroid_y < line_y:
                in_count += 1
            else:
                out_count += 1
            if cls_name:
                class_counts[cls_name] += 1

        if cls_name:
            frame_class_counts[cls_name] += 1

        # Draw box + ID + class
        color = (0,255,0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{cls_name} ID {int(track_id)}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Stats overlay
    cv2.putText(frame, f"Total: {total_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    cv2.putText(frame, f"IN: {in_count}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(frame, f"OUT: {out_count}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    return frame

# --------------------
# VIDEO THREAD
# --------------------
def video_loop():
    global display_frame
    cap = get_stream(STREAM_URL)
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            cap = get_stream(STREAM_URL)
            continue
        frame = process_frame(frame)
        with frame_lock:
            display_frame = frame.copy()

# --------------------
# LOAD YOLO
# --------------------
model = YOLO(YOLO_MODEL)

# Start video thread
thread = threading.Thread(target=video_loop, daemon=True)
thread.start()

# --------------------
# STREAMLIT DASHBOARD
# --------------------
st.title("🚦 TraffiSight Dashboard")
st.markdown("Live IN/OUT counting with YOLOv8 + SORT tracking")

col1, col2 = st.columns([3,1])
frame_placeholder = col1.empty()

# Metrics placeholders
total_placeholder = col2.empty()
in_placeholder = col2.empty()
out_placeholder = col2.empty()

# Vehicle type counts
type_placeholders = {name: col2.empty() for name in VEHICLE_CLASSES.values()}

# Trend chart placeholder
chart_placeholder = st.empty()

# Refresh loop
refresh_rate = 0.1  # seconds

while True:
    with frame_lock:
        if display_frame is not None:
            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")

    # Update metrics
    total_placeholder.metric("Total Vehicles", total_count)
    in_placeholder.metric("Vehicles IN", in_count)
    out_placeholder.metric("Vehicles OUT", out_count)
    for name in VEHICLE_CLASSES.values():
        type_placeholders[name].metric(name, class_counts[name])

    # Update historical data
    total_history.append(total_count)
    in_history.append(in_count)
    out_history.append(out_count)
    total_history = total_history[-history_len:]
    in_history = in_history[-history_len:]
    out_history = out_history[-history_len:]

    df = pd.DataFrame({
        "Total": total_history,
        "IN": in_history,
        "OUT": out_history
    })

    chart_placeholder.line_chart(df)

    time.sleep(refresh_rate)
