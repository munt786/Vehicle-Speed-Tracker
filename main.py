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

import asyncio
import base64
import time
import math
from typing import List, Dict, Tuple, Optional, Any
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
REAL_WORLD_DISTANCE_METERS = 10.0  # Default 10 meters scale for ROI height


class SpeedTracker:
    def __init__(self, max_history_sec: float = 2.5, min_time_diff: float = 0.04):
        """
        Tracks vehicle positions on a 2D Perspective transformed plane and calculates real-time speed.
        Coordinates are normalized (0.0 to 1.0) to ensure complete device & resolution independence.
        """
        self.max_history_sec = max_history_sec
        self.min_time_diff = min_time_diff
        # Stores track_id -> list of tuples: [(timestamp, bev_x, bev_y), ...]
        self.history: Dict[int, List[Tuple[float, float, float]]] = {}
        # Stores track_id -> smoothed speed (km/h)
        self.smoothed_speeds: Dict[int, float] = {}
        # Stores track_id -> smoothed normalized image position (nx, ny)
        self.smoothed_pos: Dict[int, Tuple[float, float]] = {}
        # Stores track_id -> smoothed BEV position (bx, by)
        self.smoothed_bev: Dict[int, Tuple[float, float]] = {}
        # Stores track_id -> vehicle class name
        self.classes: Dict[int, str] = {}
        # Stores track_id -> motion trail normalized points [(nx, ny), ...]
        self.img_trails: Dict[int, List[Tuple[float, float]]] = {}
        # Stores track_id -> record dict: {id, class, max_speed, last_speed, last_seen, is_speeding}
        self.vehicle_records: Dict[int, Dict] = {}
        self.last_positions: Dict[int, Tuple[float, float, float]] = {}  # track_id -> (nx, ny, timestamp)
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
        Sets source calibration points (normalized 0.0 to 1.0)
        and computes perspective homography matrix M.
        Sorts points in order: Top-Left, Top-Right, Bottom-Right, Bottom-Left.
        """
        if len(points) != 4:
            return False

        try:
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
        except Exception as e:
            print(f"Calibration matrix calculation error: {e}")
            return False

    def transform_point(self, nx: float, ny: float) -> Optional[Tuple[float, float]]:
        """
        Maps a normalized 2D coordinate (nx, ny in range 0..1) to Bird's-Eye View coordinate safely.
        """
        if self.M is None:
            return None

        try:
            pts = np.array([[[nx, ny]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(pts, self.M)
            bx, by = transformed[0][0]
            if math.isnan(bx) or math.isnan(by) or math.isinf(bx) or math.isinf(by):
                return None
            return float(bx), float(by)
        except Exception:
            return None

    def update_vehicle(
        self,
        track_id: int,
        bbox_norm: Tuple[float, float, float, float],
        class_name: str,
        current_time: float,
        roi_distance: float = 10.0,
        speed_limit: float = 50.0
    ) -> Tuple[Optional[float], Optional[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        Updates vehicle position history using smoothed trajectory filtering and calculates
        windowed real-time speed in km/h without zigzag noise.
        """
        nx1, ny1, nx2, ny2 = bbox_norm
        raw_ncx = float((nx1 + nx2) / 2.0)
        raw_ncy = float(ny2)

        # 1. Centroid Trajectory Smoothing: Eliminates bounding-box jitter / shadow oscillations
        if track_id in self.smoothed_pos:
            prev_sx, prev_sy = self.smoothed_pos[track_id]
            ncx = 0.70 * raw_ncx + 0.30 * prev_sx
            ncy = 0.70 * raw_ncy + 0.30 * prev_sy
        else:
            ncx, ncy = raw_ncx, raw_ncy
        self.smoothed_pos[track_id] = (ncx, ncy)

        # Store smooth, straight motion trail
        if track_id not in self.img_trails:
            self.img_trails[track_id] = []
        self.img_trails[track_id].append((round(ncx, 4), round(ncy, 4)))
        if len(self.img_trails[track_id]) > 20:
            self.img_trails[track_id].pop(0)

        # Store last known position for tracking association
        self.last_positions[track_id] = (ncx, ncy, current_time)

        bev_coord = self.transform_point(ncx, ncy)
        if bev_coord is None:
            return None, None, self.img_trails[track_id]

        raw_bx, raw_by = bev_coord
        if track_id in self.smoothed_bev:
            prev_bx, prev_by = self.smoothed_bev[track_id]
            bx = 0.70 * raw_bx + 0.30 * prev_bx
            by = 0.70 * raw_by + 0.30 * prev_by
        else:
            bx, by = raw_bx, raw_by
        self.smoothed_bev[track_id] = (bx, by)

        self.classes[track_id] = class_name

        if track_id not in self.history:
            self.history[track_id] = []

        # Append new observation
        self.history[track_id].append((current_time, bx, by))

        # Prune records older than max_history_sec
        self.history[track_id] = [
            record for record in self.history[track_id]
            if current_time - record[0] <= self.max_history_sec
        ]

        hist = self.history[track_id]
        if len(hist) < 2:
            smoothed_kmh = 0.0
        else:
            # 2. Multi-Frame Windowed Speed Calculation (~0.18s to 0.25s temporal window)
            # Avoids instantaneous single-frame micro-jitter and calculates true linear velocity
            target_window_sec = 0.18
            reference_record = None
            for rec in reversed(hist[:-1]):
                rec_time, rx, ry = rec
                if (current_time - rec_time) >= target_window_sec:
                    reference_record = rec
                    break
            if reference_record is None:
                reference_record = hist[0]

            ref_time, rx, ry = reference_record
            dt = current_time - ref_time

            if dt < 0.03:
                smoothed_kmh = self.smoothed_speeds.get(track_id, 0.0)
            else:
                pixel_dist = math.sqrt((bx - rx) ** 2 + (by - ry) ** 2)
                meters_per_pixel = float(roi_distance) / BEV_HEIGHT
                dist_meters = pixel_dist * meters_per_pixel

                # Deadband Filter: Ignore micro-movements < 15cm to keep stationary cars at 0.0 km/h
                if dist_meters < 0.15:
                    raw_speed_kmh = 0.0
                else:
                    speed_mps = dist_meters / dt
                    raw_speed_kmh = speed_mps * 3.6

                # Noise spike clamp (ignore anomalies above 220 km/h)
                if raw_speed_kmh > 220.0:
                    raw_speed_kmh = self.smoothed_speeds.get(track_id, 0.0)

                prev_smoothed = self.smoothed_speeds.get(track_id, raw_speed_kmh)
                alpha = 0.50
                smoothed_kmh = alpha * raw_speed_kmh + (1.0 - alpha) * prev_smoothed

        self.smoothed_speeds[track_id] = smoothed_kmh

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

        current_record["is_speeding"] = current_record["max_speed"] > speed_limit
        self.vehicle_records[track_id] = current_record

        return smoothed_kmh, (bx, by), self.img_trails[track_id]

    def cleanup_old_tracks(self, active_ids: List[int]):
        """Clean up active frame state while retaining historical telemetry records."""
        for tid in list(self.history.keys()):
            if tid not in active_ids:
                del self.history[tid]
                if tid in self.smoothed_speeds:
                    del self.smoothed_speeds[tid]
                if tid in self.smoothed_pos:
                    del self.smoothed_pos[tid]
                if tid in self.smoothed_bev:
                    del self.smoothed_bev[tid]
                if tid in self.classes:
                    del self.classes[tid]
                if tid in self.img_trails:
                    del self.img_trails[tid]

    def get_top_recent_records(self, limit: int = 10) -> List[Dict]:
        """Returns the top recent / highest speed vehicle records for the sidebar log."""
        records = list(self.vehicle_records.values())
        records.sort(key=lambda r: (r["last_seen"], r["max_speed"]), reverse=True)
        return records[:limit]


tracker = SpeedTracker()
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=40, varThreshold=25, detectShadows=False)


def merge_overlapping_boxes(boxes: List[List[float]], overlap_thresh: float = 0.3) -> List[List[float]]:
    """Merges overlapping bounding boxes into clean unified boxes."""
    if not boxes:
        return []

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


def process_frame_sync(
    img_bytes: bytes,
    calibration_pts: List[List[float]],
    speed_limit: float,
    roi_distance: float,
    engine_mode: str
) -> Dict[str, Any]:
    """
    Synchronous worker thread function:
    Decodes frame, runs YOLO / ByteTrack or motion subtractor, calculates perspective speeds,
    and returns lightweight JSON telemetry (<500 bytes) with normalized coordinates.
    """
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Frame decoding produced None"}

    h, w, _ = frame.shape
    current_time = time.time()

    # Normalize calibration points if sent in pixel space
    norm_calibration_pts = []
    for pt in calibration_pts:
        px, py = pt[0], pt[1]
        if px > 1.05 or py > 1.05:
            norm_calibration_pts.append([px / w, py / h])
        else:
            norm_calibration_pts.append([px, py])

    is_calibrated = False
    if len(norm_calibration_pts) == 4:
        is_calibrated = tracker.set_calibration(norm_calibration_pts)

    active_vehicles = []
    bev_overlay_points = []
    current_active_ids = []

    if is_calibrated and tracker.src_pts is not None:
        xyxy_norm = []
        cls_ids = []
        confs = []
        raw_ids = []

        if engine_mode == "motion":
            # ⚡ Ultra-fast background motion subtraction
            fg_mask = bg_subtractor.apply(frame)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel, iterations=2)

            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_cnts = [cnt for cnt in contours if cv2.contourArea(cnt) > (w * h * 0.005)]
            valid_cnts = sorted(valid_cnts, key=cv2.contourArea, reverse=True)[:5]

            raw_boxes = []
            for cnt in valid_cnts:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                raw_boxes.append([bx / w, by / h, (bx + bw) / w, (by + bh) / h])

            merged_boxes = merge_overlapping_boxes(raw_boxes, overlap_thresh=0.25)
            for mb in merged_boxes:
                xyxy_norm.append(mb)
                cls_ids.append(-1)
                confs.append(0.95)

        else:
            # 🤖 Accelerated YOLOv8 + ByteTrack (imgsz=384 for 3x faster CPU execution)
            results = model.track(
                frame,
                persist=True,
                verbose=False,
                classes=VEHICLE_CLASSES,
                conf=0.20,
                iou=0.5,
                imgsz=384,
                tracker="bytetrack.yaml"
            )

            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                boxes_xyxy = boxes.xyxy.cpu().numpy()
                cls_list = boxes.cls.int().cpu().tolist()
                conf_list = boxes.conf.cpu().numpy().tolist()
                id_list = boxes.id.int().cpu().tolist() if boxes.id is not None else []

                for idx, b in enumerate(boxes_xyxy):
                    xyxy_norm.append([
                        float(b[0] / w),
                        float(b[1] / h),
                        float(b[2] / w),
                        float(b[3] / h)
                    ])
                    cls_ids.append(cls_list[idx])
                    confs.append(conf_list[idx])
                    if idx < len(id_list):
                        raw_ids.append(id_list[idx])

        # Track ID assignment and re-identification
        track_ids = []
        for idx, box in enumerate(xyxy_norm):
            if idx < len(raw_ids):
                track_ids.append(raw_ids[idx])
            else:
                bncx = (box[0] + box[2]) / 2.0
                bncy = box[3]
                best_id = None
                best_dist = 0.20  # Normalized distance threshold (20% of frame)
                for tid, (lcx, lcy, ltime) in tracker.last_positions.items():
                    if current_time - ltime < 1.0:
                        dist = math.sqrt((bncx - lcx) ** 2 + (bncy - lcy) ** 2)
                        if dist < best_dist:
                            best_dist = dist
                            best_id = tid
                if best_id is None:
                    best_id = tracker.next_synthetic_id
                    tracker.next_synthetic_id += 1
                track_ids.append(best_id)

        for box, track_id, cls_id, conf in zip(xyxy_norm, track_ids, cls_ids, confs):
            current_active_ids.append(track_id)
            if cls_id == -1:
                class_name = "Motion Target"
            else:
                class_name = model.names.get(cls_id, "vehicle").capitalize()

            speed_kmh, bev_pt, trail = tracker.update_vehicle(
                track_id, (box[0], box[1], box[2], box[3]), class_name, current_time,
                roi_distance=roi_distance, speed_limit=speed_limit
            )

            if bev_pt:
                bev_overlay_points.append({
                    "id": track_id,
                    "bx": round(float(bev_pt[0]), 1),
                    "by": round(float(bev_pt[1]), 1),
                    "speed": round(speed_kmh or 0.0, 1)
                })

            active_vehicles.append({
                "id": track_id,
                "class": class_name,
                "speed": round(speed_kmh, 1) if speed_kmh else 0.0,
                "confidence": round(float(conf), 2),
                "bbox": [round(float(c), 4) for c in box],
                "trail": trail,
                "is_speeding": bool(speed_kmh and speed_kmh > speed_limit)
            })

        tracker.cleanup_old_tracks(current_active_ids)

    top_records = tracker.get_top_recent_records(limit=10)
    total_detected = len(tracker.vehicle_records)
    speeding_violations = sum(1 for r in tracker.vehicle_records.values() if r["is_speeding"])

    # Returns pure lightweight JSON telemetry (~300 bytes) - NO Base64 video images!
    return {
        "status": "ok",
        "calibrated": is_calibrated,
        "active_vehicles": active_vehicles,
        "bev_points": bev_overlay_points,
        "top_records": top_records,
        "total_count": total_detected,
        "speeding_count": speeding_violations
    }


@app.get("/")
async def get_index(request: Request):
    """Renders the main speed tracking web interface."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    High-Performance WebSocket endpoint.
    Offloads frame processing & YOLO to worker threads and streams lightweight JSON telemetry.
    """
    await websocket.accept()
    print("Client connected via WebSocket.")

    try:
        while True:
            payload = await websocket.receive_json()

            image_data = payload.get("image")
            calibration_pts = payload.get("points", [])
            speed_limit = float(payload.get("speed_limit", 50.0))
            roi_distance = float(payload.get("roi_distance", 10.0))
            engine_mode = payload.get("engine", "motion")

            if not image_data:
                await websocket.send_json({"error": "No image data provided"})
                continue

            try:
                if "," in image_data:
                    base64_str = image_data.split(",")[1]
                else:
                    base64_str = image_data

                img_bytes = base64.b64decode(base64_str)
            except Exception as e:
                await websocket.send_json({"error": f"Failed to decode image: {str(e)}"})
                continue

            # Run detection and tracking in asynchronous worker thread
            result = await asyncio.to_thread(
                process_frame_sync,
                img_bytes,
                calibration_pts,
                speed_limit,
                roi_distance,
                engine_mode
            )

            # Send back lightweight JSON telemetry
            await websocket.send_json(result)

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
