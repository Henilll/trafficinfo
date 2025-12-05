# import cv2
# import streamlit as st
# from ultralytics import YOLO
# import numpy as np

# # -----------------------------
# # YOLO model
# # -----------------------------
# model = YOLO("yolov8n.pt")

# # Camera URL
# url = "http://109.206.96.58:8080/cam_1.cgi"

# # -----------------------------
# # Streamlit Dashboard Layout
# # -----------------------------
# st.set_page_config(layout="wide")
# st.title("🚗 Real-Time Vehicle Counting Dashboard")

# col1, col2 = st.columns([3, 1])

# # Counters
# IN_count = 0
# OUT_count = 0
# vehicle_total = 0

# vehicle_stats = {"car": 0, "bike": 0, "bus": 0, "truck": 0}
# class_map = {2: "car", 3: "bike", 5: "bus", 7: "truck"}

# LINE_Y = 300
# offset = 20
# previous_centers = {}

# # Dashboard values
# in_box = col2.metric("Vehicles IN", IN_count)
# out_box = col2.metric("Vehicles OUT", OUT_count)
# total_box = col2.metric("Total Vehicles", vehicle_total)

# car_box = col2.metric("Cars", 0)
# bike_box = col2.metric("Bikes", 0)
# bus_box = col2.metric("Buses", 0)
# truck_box = col2.metric("Trucks", 0)

# frame_container = col1.empty()

# # -----------------------------
# # Video Stream Loop
# # -----------------------------
# cap = cv2.VideoCapture(url)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         st.warning("Camera disconnected!")
#         break

#     results = model.predict(frame, conf=0.5)


#     if results[0].boxes.id is not None:
#         ids = results[0].boxes.id.cpu().numpy()
#         boxes = results[0].boxes.xyxy.cpu().numpy()
#         cls = results[0].boxes.cls.cpu().numpy()

#         for i, box in enumerate(boxes):
#             x1, y1, x2, y2 = box.astype(int)
#             center_y = (y1 + y2) // 2
#             obj_id = int(ids[i])
#             c = int(cls[i])

#             if c not in class_map:
#                 continue

#             vehicle_type = class_map[c]

#             if obj_id in previous_centers:
#                 prev_y = previous_centers[obj_id]

#                 # DOWN (IN)
#                 if prev_y < LINE_Y - offset and center_y > LINE_Y + offset:
#                     IN_count += 1
#                     vehicle_stats[vehicle_type] += 1

#                 # UP (OUT)
#                 elif prev_y > LINE_Y + offset and center_y < LINE_Y - offset:
#                     OUT_count += 1
#                     vehicle_stats[vehicle_type] += 1

#             previous_centers[obj_id] = center_y

#             # Draw box
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
#             cv2.putText(frame, vehicle_type, (x1, y1-10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

#     # Counting line
#     cv2.line(frame, (0, LINE_Y), (frame.shape[1], LINE_Y), (0,0,255), 2)

#     # Update dashboard
#     vehicle_total = IN_count + OUT_count

#     in_box.metric("Vehicles IN", IN_count)
#     out_box.metric("Vehicles OUT", OUT_count)
#     total_box.metric("Total Vehicles", vehicle_total)

#     car_box.metric("Cars", vehicle_stats['car'])
#     bike_box.metric("Bikes", vehicle_stats['bike'])
#     bus_box.metric("Buses", vehicle_stats['bus'])
#     truck_box.metric("Trucks", vehicle_stats['truck'])

#     # Stream image
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     frame_container.image(frame, use_column_width=True)
from ultralytics import YOLO
import cv2
import numpy as np
from sort import Sort
import time

# --------------------
#  SORT Tracker Init
# --------------------
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# --------------------
#  Load YOLO model
# --------------------
model = YOLO("yolov8n.pt")  # fast + lightweight

# --------------------
#  LIVE CAMERA STREAM
# --------------------
stream_url = "http://109.206.96.58:8080/cam_1.cgi"

def get_stream(url):
    cap = cv2.VideoCapture(url)
    while not cap.isOpened():
        print("❌ Cannot open IP Camera stream. Retrying in 2s...")
        cap.release()
        time.sleep(2)
        cap = cv2.VideoCapture(url)
    return cap

cap = get_stream(stream_url)
print("✅ LIVE Vehicle Detection Started... Press Q to quit")

vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
total_count = 0
track_memory = set()

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("⚠️ Failed to read frame. Reconnecting...")
        cap.release()
        cap = get_stream(stream_url)
        continue

    # Resize frame for faster YOLO inference
    frame_resized = cv2.resize(frame, (640, 360))
    
    # YOLO inference
    results = model(frame_resized, conf=0.4)[0]

    detections = []
    for box in results.boxes.data:
        x1, y1, x2, y2, score, cls = box.tolist()
        # Scale back to original frame size
        scale_x = frame.shape[1] / 640
        scale_y = frame.shape[0] / 360
        x1 *= scale_x
        x2 *= scale_x
        y1 *= scale_y
        y2 *= scale_y

        if int(cls) in vehicle_classes:
            detections.append([x1, y1, x2, y2, score])

    detections = np.array(detections)
    
    # SORT tracking
    tracked_objects = tracker.update(detections)

    for x1, y1, x2, y2, track_id in tracked_objects:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # Count new vehicles
        if int(track_id) not in track_memory:
            track_memory.add(int(track_id))
            total_count += 1

        # Draw box + ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {int(track_id)}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display total count
    cv2.putText(frame, f"Total Vehicles: {total_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    cv2.imshow("LIVE Vehicle Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

