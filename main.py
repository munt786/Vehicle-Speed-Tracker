import os
import sys

# Ensure OpenCV native DLL search paths are configured for Windows / MSYS2 environments
cv2_dir = os.path.join(sys.prefix, "lib", "python3.12", "site-packages", "cv2")
if os.path.exists(cv2_dir) and hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(cv2_dir)
        os.environ["PATH"] = cv2_dir + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

import base64
import time
import math
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO

app = FastAPI(title="Vehicle Speed Tracking System")

# Mount static files and setup Jinja2 templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize YOLOv8 Nano model for fast real-time object tracking
# COCO Vehicle classes: 2 (car), 3 (motorcycle), 5 (bus), 7 (truck)
VEHICLE_CLASSES = [2, 3, 5, 7]
print("Loading YOLOv8 model...")
model = YOLO("yolov8n.pt")
print("YOLOv8 model loaded successfully.")

# Target 2D Bird's-Eye View (BEV) dimensions
BEV_WIDTH = 500
BEV_HEIGHT = 1000
REAL_WORLD_DISTANCE_METERS = 10.0  # 10 meters scale for ROI height


class SpeedTracker:
    def __init__(self, max_history_sec: float = 2.0, min_time_diff: float = 0.04):
        """
        Tracks vehicle positions on a 2D Perspective transformed plane and calculates real-time speed.
        """
        self.max_history_sec = max_history_sec
        self.min_time_diff = min_time_diff
        # Stores track_id -> list of tuples: [(timestamp, bev_x, bev_y), ...]
        self.history: Dict[int, List[Tuple[float, float, float]]] = {}
        # Stores track_id -> smoothed speed (km/h)
        self.smoothed_speeds: Dict[int, float] = {}
        # Stores track_id -> vehicle class name
        self.classes: Dict[int, str] = {}
        # Stores track_id -> motion trail image points [(cx, cy), ...]
        self.img_trails: Dict[int, List[Tuple[int, int]]] = {}
        # Stores track_id -> record dict: {id, class, max_speed, last_speed, last_seen, is_speeding}
        self.vehicle_records: Dict[int, Dict] = {}
        self.last_positions: Dict[int, Tuple[float, float, float]] = {}  # track_id -> (cx, cy, timestamp)
        self.next_synthetic_id = 1000
        # Matrix and dst points
        self.M: Optional[np.ndarray] = None
        self.src_pts: Optional[np.ndarray] = None
        self.dst_pts = np.float32([
            [0, 0],
            [BEV_WIDTH, 0],
            [BEV_WIDTH, BEV_HEIGHT],
            [0, BEV_HEIGHT]
        ])

    def set_calibration(self, points: List[List[float]]) -> bool:
        """
        Sets source calibration points and computes homography matrix M.
        Sorts points in order: Top-Left, Top-Right, Bottom-Right, Bottom-Left.
        """
        if len(points) != 4:
            return False

        pts = np.array(points, dtype=np.float32)
        sums = pts.sum(axis=1)
        diffs = np.diff(pts, axis=1).flatten()

        top_left = pts[np.argmin(sums)]
        bottom_right = pts[np.argmax(sums)]
        top_right = pts[np.argmin(diffs)]
        bottom_left = pts[np.argmax(diffs)]

        ordered_src = np.float32([top_left, top_right, bottom_right, bottom_left])
        self.src_pts = ordered_src
        self.M = cv2.getPerspectiveTransform(ordered_src, self.dst_pts)
        return True

    def transform_point(self, x: float, y: float) -> Optional[Tuple[float, float]]:
        """
        Maps a 2D image coordinate (x,y) to Bird's-Eye View coordinate.
        """
        if self.M is None:
            return None

        pts = np.array([[[x, y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pts, self.M)
        bx, by = transformed[0][0]
        return float(bx), float(by)

    def update_vehicle(self, track_id: int, bbox: Tuple[float, float, float, float], class_name: str, current_time: float, roi_distance: float = 10.0, speed_limit: float = 50.0) -> Tuple[Optional[float], Optional[Tuple[float, float]]]:
        """
        Updates position history for a vehicle and calculates current speed in km/h instantly.
        """
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2.0)
        cy = int(y2)

        # Store motion trail points for video overlay
        if track_id not in self.img_trails:
            self.img_trails[track_id] = []
        self.img_trails[track_id].append((cx, cy))
        if len(self.img_trails[track_id]) > 18:
            self.img_trails[track_id].pop(0)

        # Store last known position for sticky re-association
        self.last_positions[track_id] = (float(cx), float(cy), current_time)

        bev_coord = self.transform_point(cx, cy)
        if bev_coord is None:
            return None, None

        bx, by = bev_coord
        self.classes[track_id] = class_name

        if track_id not in self.history:
            self.history[track_id] = []

        # Append new observation
        self.history[track_id].append((current_time, bx, by))

        # Prune old records older than max_history_sec
        self.history[track_id] = [
            record for record in self.history[track_id]
            if current_time - record[0] <= self.max_history_sec
        ]

        hist = self.history[track_id]
        if len(hist) < 2:
            smoothed_kmh = 0.0
        else:
            # Immediate responsive calculation between latest 2 frames
            prev_record = hist[-2]
            prev_time, px, py = prev_record
            dt = current_time - prev_time

            if dt <= 0.005:
                smoothed_kmh = self.smoothed_speeds.get(track_id, 0.0)
            else:
                # Distance on 2D BEV grid (in pixels)
                pixel_dist = math.sqrt((bx - px) ** 2 + (by - py) ** 2)

                # Dynamic scale factor: BEV_HEIGHT pixels = roi_distance meters
                meters_per_pixel = float(roi_distance) / BEV_HEIGHT
                dist_meters = pixel_dist * meters_per_pixel

                # Speed in meters per second -> km/h
                speed_mps = dist_meters / dt
                raw_speed_kmh = speed_mps * 3.6

                # Apply Exponential Moving Average (EMA) smoothing for stability
                prev_smoothed = self.smoothed_speeds.get(track_id, raw_speed_kmh)
                alpha = 0.65  # Smoothing factor
                smoothed_kmh = alpha * raw_speed_kmh + (1 - alpha) * prev_smoothed
        
        self.smoothed_speeds[track_id] = smoothed_kmh

        # Update vehicle persistent log record with dynamic speed limit
        current_record = self.vehicle_records.get(track_id, {
            "id": track_id,
            "class": class_name,
            "max_speed": 0.0,
            "last_speed": 0.0,
            "first_seen": current_time,
            "last_seen": current_time,
            "is_speeding": False
        })

        current_record["last_speed"] = round(smoothed_kmh, 1)
        current_record["last_seen"] = current_time
        if smoothed_kmh > current_record["max_speed"]:
            current_record["max_speed"] = round(smoothed_kmh, 1)
        
        # Dynamic speeding evaluation
        current_record["is_speeding"] = current_record["max_speed"] > speed_limit

        self.vehicle_records[track_id] = current_record

        return smoothed_kmh, bev_coord

    def cleanup_old_tracks(self, active_ids: List[int]):
        """Clean up active frame state while retaining vehicle history records."""
        for tid in list(self.history.keys()):
            if tid not in active_ids:
                del self.history[tid]
                if tid in self.smoothed_speeds:
                    del self.smoothed_speeds[tid]
                if tid in self.classes:
                    del self.classes[tid]
                if tid in self.img_trails:
                    del self.img_trails[tid]

    def get_top_recent_records(self, limit: int = 10) -> List[Dict]:
        """Returns the top recent / highest speed vehicle records for the UI sidebar log."""
        records = list(self.vehicle_records.values())
        # Sort primarily by last_seen (most recent) and secondarily by max_speed
        records.sort(key=lambda r: (r["last_seen"], r["max_speed"]), reverse=True)
        return records[:limit]


tracker = SpeedTracker()
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=25, detectShadows=False)


def merge_overlapping_boxes(boxes: List[List[int]], overlap_thresh: float = 0.3) -> List[List[int]]:
    """Merges overlapping bounding boxes into clean unified boxes."""
    if not boxes:
        return []
    
    # Sort boxes by area descending
    boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    merged = []
    
    for box in boxes:
        x1, y1, x2, y2 = box
        matched = False
        for m in merged:
            mx1, my1, mx2, my2 = m
            ix1, iy1 = max(x1, mx1), max(y1, my1)
            ix2, iy2 = min(x2, mx2), min(y2, my2)
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            intersection = iw * ih
            min_area = min((x2 - x1) * (y2 - y1), (mx2 - mx1) * (my2 - my1))
            
            if min_area > 0 and (intersection / min_area) > overlap_thresh:
                m[0] = min(mx1, x1)
                m[1] = min(my1, y1)
                m[2] = max(mx2, x2)
                m[3] = max(my2, y2)
                matched = True
                break
        if not matched:
            merged.append([x1, y1, x2, y2])
            
    return merged


@app.get("/")
async def get_index(request: Request):
    """Renders the main speed tracking web interface."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint handling client frames, calibration coordinates, YOLO tracking,
    perspective transformation, speed calculation, and returning annotated frames.
    """
    await websocket.accept()
    print("Client connected via WebSocket.")

    try:
        while True:
            # Receive data payload from frontend
            payload = await websocket.receive_json()

            image_data = payload.get("image")
            calibration_pts = payload.get("points", [])
            speed_limit = float(payload.get("speed_limit", 50.0))
            roi_distance = float(payload.get("roi_distance", 10.0))
            engine_mode = payload.get("engine", "motion")  # "motion" or "yolo"

            if not image_data:
                await websocket.send_json({"error": "No image data provided"})
                continue

            try:
                if "," in image_data:
                    base64_str = image_data.split(",")[1]
                else:
                    base64_str = image_data

                img_bytes = base64.b64decode(base64_str)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            except Exception as e:
                await websocket.send_json({"error": f"Failed to decode image: {str(e)}"})
                continue

            if frame is None:
                await websocket.send_json({"error": "Frame decoding produced None"})
                continue

            h, w, _ = frame.shape
            current_time = time.time()
            active_vehicles = []
            bev_overlay_points = []

            is_calibrated = False
            if len(calibration_pts) == 4:
                is_calibrated = tracker.set_calibration(calibration_pts)

            if is_calibrated and tracker.src_pts is not None:
                pts_int = tracker.src_pts.astype(np.int32).reshape((-1, 1, 2))
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts_int], (0, 255, 128), lineType=cv2.LINE_AA)
                cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
                cv2.polylines(frame, [pts_int], isClosed=True, color=(0, 255, 128), thickness=2, lineType=cv2.LINE_AA)

                # Label dynamic ROI distance marker
                cv2.putText(frame, f"ROI Calibrated ({roi_distance:.1f} Meters)", 
                            (int(pts_int[0][0][0]), max(25, int(pts_int[0][0][1]) - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 1, cv2.LINE_AA)

                xyxy_coords = []
                cls_ids = []
                confs = []
                raw_ids = []

                if engine_mode == "motion":
                    # ⚡ High-Speed Motion Subtraction Engine (0 CPU Lag, 60+ FPS on Render CPU)
                    fg_mask = bg_subtractor.apply(frame)
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
                    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
                    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel, iterations=3)

                    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    valid_cnts = [cnt for cnt in contours if cv2.contourArea(cnt) > 2000]
                    valid_cnts = sorted(valid_cnts, key=cv2.contourArea, reverse=True)[:3]

                    raw_boxes = []
                    for cnt in valid_cnts:
                        bx, by, bw, bh = cv2.boundingRect(cnt)
                        raw_boxes.append([bx, by, bx + bw, by + bh])

                    merged_boxes = merge_overlapping_boxes(raw_boxes, overlap_thresh=0.2)
                    for mb in merged_boxes:
                        xyxy_coords.append(mb)
                        cls_ids.append(-1)
                        confs.append(0.95)

                    xyxy_coords = np.array(xyxy_coords) if len(xyxy_coords) > 0 else np.empty((0, 4))
                else:
                    # 🤖 AI Neural Engine (YOLOv8 Vehicles)
                    results = model.track(frame, persist=True, verbose=False, classes=VEHICLE_CLASSES, conf=0.15, iou=0.5)

                    if results and len(results) > 0 and results[0].boxes is not None:
                        boxes = results[0].boxes
                        xyxy_coords = boxes.xyxy.cpu().numpy()
                        cls_ids = boxes.cls.int().cpu().tolist()
                        confs = boxes.conf.cpu().numpy()
                        if boxes.id is not None:
                            raw_ids = boxes.id.int().cpu().tolist()

                track_ids = []
                for idx, box in enumerate(xyxy_coords):
                    if idx < len(raw_ids):
                        track_ids.append(raw_ids[idx])
                    else:
                        bcx = (box[0] + box[2]) / 2.0
                        bcy = box[3]
                        best_id = None
                        best_dist = 120.0
                        for tid, (lcx, lcy, ltime) in tracker.last_positions.items():
                            if current_time - ltime < 1.0:
                                dist = math.sqrt((bcx - lcx)**2 + (bcy - lcy)**2)
                                if dist < best_dist:
                                    best_dist = dist
                                    best_id = tid
                        if best_id is None:
                            best_id = tracker.next_synthetic_id
                            tracker.next_synthetic_id += 1
                        track_ids.append(best_id)

                current_active_ids = []
                for box, track_id, cls_id, conf in zip(xyxy_coords, track_ids, cls_ids, confs):
                    current_active_ids.append(track_id)
                    x1, y1, x2, y2 = box
                    if cls_id == -1:
                        class_name = "Motion Target"
                    else:
                        class_name = model.names.get(cls_id, "vehicle").capitalize()

                    speed_kmh, bev_pt = tracker.update_vehicle(
                        track_id, (x1, y1, x2, y2), class_name, current_time, 
                        roi_distance=roi_distance, speed_limit=speed_limit
                    )

                    if bev_pt:
                        bev_overlay_points.append((track_id, bev_pt[0], bev_pt[1], speed_kmh or 0.0))

                    speed_display = f"{speed_kmh:.1f} km/h" if speed_kmh is not None else "Calculating..."

                    active_vehicles.append({
                        "id": track_id,
                        "class": class_name,
                        "speed": round(speed_kmh, 1) if speed_kmh else 0.0,
                        "confidence": round(float(conf), 2)
                    })

                    if speed_kmh is None or speed_kmh < (speed_limit * 0.8):
                        box_color = (0, 255, 0)
                    elif speed_kmh <= speed_limit:
                        box_color = (0, 215, 255)
                    else:
                        box_color = (0, 0, 255)

                    if track_id in tracker.img_trails:
                        trail = tracker.img_trails[track_id]
                        for i in range(1, len(trail)):
                            cv2.line(frame, trail[i - 1], trail[i], box_color, 2, cv2.LINE_AA)

                    ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), box_color, 2, lineType=cv2.LINE_AA)

                    cx, cy = int((ix1 + ix2) / 2), iy2
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 255), -1, lineType=cv2.LINE_AA)

                    label = f"#{track_id} {class_name} | {speed_display}"
                    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

                    badge_y1 = max(0, iy1 - text_h - 10)
                    badge_y2 = iy1
                    cv2.rectangle(frame, (ix1, badge_y1), (ix1 + text_w + 10, badge_y2), box_color, -1)
                    cv2.putText(frame, label, (ix1 + 5, max(14, iy1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

                tracker.cleanup_old_tracks(current_active_ids)

            else:
                num_pts = len(calibration_pts)
                if num_pts > 0:
                    for idx, pt in enumerate(calibration_pts):
                        px, py = int(pt[0]), int(pt[1])
                        cv2.circle(frame, (px, py), 6, (0, 255, 255), -1, lineType=cv2.LINE_AA)
                        cv2.putText(frame, str(idx + 1), (px + 8, py - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

                    if num_pts > 1:
                        pts_arr = np.array(calibration_pts, np.int32).reshape((-1, 1, 2))
                        cv2.polylines(frame, [pts_arr], isClosed=False, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)

                msg = f"CALIBRATION NEEDED: Click 4 points on road ({num_pts}/4 points set)"
                cv2.rectangle(frame, (10, 10), (min(w - 10, 620), 45), (10, 13, 20), -1)
                cv2.putText(frame, msg, (20, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

            # Generate Bird's-Eye View (BEV) Mini Map Overlay
            if is_calibrated and tracker.M is not None:
                bev_w, bev_h = 110, 180
                bev_img = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
                bev_img[:] = (15, 20, 30)
                cv2.rectangle(bev_img, (0, 0), (bev_w - 1, bev_h - 1), (0, 242, 254), 1)
                cv2.line(bev_img, (0, int(bev_h / 2)), (bev_w, int(bev_h / 2)), (50, 60, 80), 1)

                for tid, bx, by, spd in bev_overlay_points:
                    mx = int((bx / BEV_WIDTH) * bev_w)
                    my = int((by / BEV_HEIGHT) * bev_h)
                    mx = max(3, min(bev_w - 4, mx))
                    my = max(3, min(bev_h - 4, my))

                    dot_color = (0, 255, 0) if spd < speed_limit else (0, 0, 255)
                    cv2.circle(bev_img, (mx, my), 4, dot_color, -1, lineType=cv2.LINE_AA)
                    cv2.putText(bev_img, f"#{tid}", (mx + 5, my + 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

                margin = 15
                top_y = margin
                left_x = w - bev_w - margin

                if left_x > 0 and top_y + bev_h < h:
                    cv2.rectangle(frame, (left_x - 2, top_y - 20), (left_x + bev_w + 2, top_y + bev_h + 2), (10, 13, 20), -1)
                    cv2.putText(frame, f"BEV {roi_distance:.0f}m Grid", (left_x + 4, top_y - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 242, 254), 1, cv2.LINE_AA)
                    frame[top_y:top_y + bev_h, left_x:left_x + bev_w] = bev_img

            # Encode annotated frame to JPEG base64 (HD quality 80)
            _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            encoded_frame = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

            # Get Top 10 Recent / Speed Log
            top_records = tracker.get_top_recent_records(limit=10)
            total_detected = len(tracker.vehicle_records)
            speeding_violations = sum(1 for r in tracker.vehicle_records.values() if r["is_speeding"])

            # Send response back to WebSocket client
            await websocket.send_json({
                "status": "ok",
                "frame": encoded_frame,
                "calibrated": is_calibrated,
                "active_vehicles": active_vehicles,
                "top_records": top_records,
                "total_count": total_detected,
                "speeding_count": speeding_violations
            })

    except WebSocketDisconnect:
        print("WebSocket client disconnected.")
    except Exception as e:
        print(f"Error in WebSocket loop: {e}")
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
