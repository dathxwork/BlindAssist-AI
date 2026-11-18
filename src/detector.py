"""
Detector module for object detection, distance estimation, and spatial positioning.
Supports both Desktop OpenCV client (ObjectDetector) and FastAPI Web App (ObstacleDetector).
"""

import os
import sys
import threading
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import cv2
import torch
from ultralytics import YOLO

# Ensure root config and src modules are importable regardless of working directory
_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

import config
from src.visualizer import draw_corner_box, render_unicode_batch


# ==============================================================================
# 1. BẢNG PHÂN LOẠI VẬT THỂ & CẤU HÌNH KHOẢNG CÁCH DÀNH CHO WEB VÀ DESKTOP
# ==============================================================================

CUSTOM_CLASS_INFO = {
    "Bang hieu": {"vn_name": "Bảng hiệu", "real_height": 1.20, "category": "obstacle", "color": (246, 92, 139)},
    "Cay coi": {"vn_name": "Cây cối", "real_height": 2.50, "category": "obstacle", "color": (166, 184, 20)},
    "Cot": {"vn_name": "Cột điện", "real_height": 2.50, "category": "obstacle", "color": (212, 182, 6)},
    "Den giao thong": {"vn_name": "Đèn giao thông", "real_height": 2.80, "category": "obstacle", "color": (11, 158, 245)},
    "Nguoi": {"vn_name": "Người", "real_height": 1.65, "category": "person", "color": (129, 185, 16)},
    "O ga": {"vn_name": "Ổ gà", "real_height": None, "category": "obstacle", "color": (68, 68, 239), "is_ground": True},
    "Thung rac": {"vn_name": "Thùng rác", "real_height": 0.90, "category": "obstacle", "color": (184, 163, 148)},
    "Xe dap": {"vn_name": "Xe đạp", "real_height": 1.10, "category": "vehicle", "color": (246, 130, 59)},
    "Xe hoi": {"vn_name": "Ô tô", "real_height": 1.55, "category": "vehicle", "color": (8, 179, 234)},
    "Xe may": {"vn_name": "Xe máy", "real_height": 1.15, "category": "vehicle", "color": (153, 72, 236)},
}

COCO_CLASS_INFO = {
    "person": {"vn_name": "Người", "real_height": 1.65, "category": "person", "color": (129, 185, 16)},
    "car": {"vn_name": "Ô tô", "real_height": 1.55, "category": "vehicle", "color": (8, 179, 234)},
    "bicycle": {"vn_name": "Xe đạp", "real_height": 1.10, "category": "vehicle", "color": (246, 130, 59)},
    "motorcycle": {"vn_name": "Xe máy", "real_height": 1.15, "category": "vehicle", "color": (153, 72, 236)},
    "bus": {"vn_name": "Xe buýt", "real_height": 3.00, "category": "vehicle", "color": (11, 158, 245)},
    "truck": {"vn_name": "Xe tải", "real_height": 2.80, "category": "vehicle", "color": (0, 140, 255)},
    "traffic light": {"vn_name": "Đèn giao thông", "real_height": 2.80, "category": "obstacle", "color": (11, 158, 245)},
    "stop sign": {"vn_name": "Biển báo dừng", "real_height": 1.20, "category": "obstacle", "color": (68, 68, 239)},
    "fire hydrant": {"vn_name": "Trụ cứu hỏa", "real_height": 0.75, "category": "obstacle", "color": (68, 68, 239)},
    "bench": {"vn_name": "Ghế dài", "real_height": 0.85, "category": "obstacle", "color": (212, 182, 6)},
    "chair": {"vn_name": "Ghế", "real_height": 0.85, "category": "obstacle", "color": (212, 182, 6)},
    "potted plant": {"vn_name": "Chậu cây", "real_height": 0.80, "category": "obstacle", "color": (166, 184, 20)},
    "dining table": {"vn_name": "Bàn", "real_height": 0.75, "category": "obstacle", "color": (212, 182, 6)},
    "cat": {"vn_name": "Con mèo", "real_height": 0.35, "category": "obstacle", "color": (200, 120, 255)},
    "dog": {"vn_name": "Con chó", "real_height": 0.55, "category": "obstacle", "color": (200, 120, 255)},
    "backpack": {"vn_name": "Ba lô", "real_height": 0.50, "category": "obstacle", "color": (246, 130, 59)},
    "umbrella": {"vn_name": "Chiếc ô", "real_height": 0.90, "category": "obstacle", "color": (246, 92, 139)},
    "handbag": {"vn_name": "Túi xách", "real_height": 0.40, "category": "obstacle", "color": (246, 92, 139)},
    "suitcase": {"vn_name": "Va li", "real_height": 0.65, "category": "obstacle", "color": (246, 130, 59)},
    "bottle": {"vn_name": "Chai nước", "real_height": 0.25, "category": "obstacle", "color": (212, 182, 6)},
    "tv": {"vn_name": "Màn hình TV", "real_height": 0.65, "category": "obstacle", "color": (246, 130, 59)},
    "laptop": {"vn_name": "Máy tính xách tay", "real_height": 0.28, "category": "obstacle", "color": (246, 130, 59)},
    "cell phone": {"vn_name": "Điện thoại", "real_height": 0.16, "category": "obstacle", "color": (246, 130, 59)},
}

DEFAULT_HEIGHT = 1.00

DISTANCE_CONFIG = {
    "very_close": config.DISTANCE_TIERS["VERY_CLOSE"],
    "close": config.DISTANCE_TIERS["CLOSE"],
    "medium": config.DISTANCE_TIERS["MEDIUM"],
}


def get_distance_level(dist_m: Optional[float], is_reliable: bool = True) -> str:
    return config.get_distance_level(dist_m if dist_m is not None else -1.0, is_reliable=is_reliable)


PRECOMPUTED_LOOKUP: Dict[str, Dict[str, Any]] = {}
for k, v in {**COCO_CLASS_INFO, **CUSTOM_CLASS_INFO}.items():
    PRECOMPUTED_LOOKUP[k] = v
    PRECOMPUTED_LOOKUP[k.lower()] = v
    PRECOMPUTED_LOOKUP[k.replace(" ", "_")] = v
    PRECOMPUTED_LOOKUP[k.lower().replace(" ", "_")] = v
    PRECOMPUTED_LOOKUP[v["vn_name"]] = v
    PRECOMPUTED_LOOKUP[v["vn_name"].lower()] = v


class DangerousObject(dict):
    def __iter__(self):
        yield self["label"]
        yield self["direction"]
        yield self["distance"]

    def __getitem__(self, item):
        if isinstance(item, int):
            vals = [self["label"], self["direction"], self["distance"], self["box"]]
            return vals[item]
        return super().__getitem__(item)


def estimate_optical_distance(
    cords_or_h: Any,
    raw_label: str,
    focal_length: float = config.DEFAULT_FOCAL_LENGTH,
    frame_height: int = 720,
) -> Tuple[float, str, bool]:
    """
    Shared Pinhole Camera + Ground-Plane distance estimator used by both
    Desktop App (ObjectDetector) and Web App (ObstacleDetector).
    """
    fh = frame_height if frame_height and frame_height > 0 else 720
    fl_norm = float(focal_length) * (fh / 720.0)

    if isinstance(cords_or_h, (list, tuple)) and len(cords_or_h) >= 4:
        x1, y1, x2, y2 = [int(v) for v in cords_or_h[:4]]
        box_h = max(y2 - y1, 1)
    else:
        box_h = max(int(cords_or_h), 1)
        x1, y1, x2, y2 = 0, 0, 100, box_h

    info = PRECOMPUTED_LOOKUP.get(raw_label, {})
    is_pothole = (
        str(raw_label).lower() in ("o ga", "ổ gà", "o_ga", "pothole")
        or bool(info.get("is_ground", False))
    )

    if is_pothole:
        horizon_y = fh * 0.40
        if y2 <= horizon_y:
            return 6.0, "Xa", False

        ground_span = max(fh - horizon_y, 1.0)
        ground_ratio = (y2 - horizon_y) / ground_span
        est_dist = round(float(1.25 / max(ground_ratio, 0.18)), 1)
        level = get_distance_level(est_dist, is_reliable=True)
        return est_dist, level, True

    real_h = info.get("real_height") if info else DEFAULT_HEIGHT
    if real_h is None or real_h <= 0:
        return 3.0, "Chưa xác định", False

    dist = round(max(0.3, float((real_h * fl_norm) / box_h)), 1)
    level = get_distance_level(dist, is_reliable=True)
    return dist, level, True


# ==============================================================================
# 2. OBJECT DETECTOR (DÀNH CHO DESKTOP APP MAIN.PY)
# ==============================================================================

class DetectionResult:
    """Represents a single detected object with geometric & spatial attributes."""

    def __init__(self, label: str, confidence: float, bbox: List[int], distance: float, position: str):
        self.label = label
        self.confidence = confidence
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.distance = distance
        self.position = position

    @property
    def center_x(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2.0

    @property
    def center_y(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2.0

    @property
    def width(self) -> int:
        return max(1, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> int:
        return max(1, self.bbox[3] - self.bbox[1])


def resolve_accelerated_weights(model_path: str, device: str) -> str:
    """
    On CPU, automatically prefers a pre-exported dynamic ONNX Runtime model (<stem>.onnx)
    if available alongside <stem>.pt, cutting CPU inference latency by ~2x while keeping
    identical class metadata and accuracy.
    """
    if device == "cpu" and model_path.lower().endswith(".pt"):
        onnx_candidate = os.path.splitext(model_path)[0] + ".onnx"
        if os.path.exists(onnx_candidate):
            try:
                import onnxruntime  # noqa: F401
                return onnx_candidate
            except ImportError:
                pass
    return model_path


class ObjectDetector:
    """Wraps YOLO model inference and calculates spatial metrics for Desktop OpenCV App."""

    def __init__(
        self,
        model_path: str = "models/best.pt",
        fov_deg: float = config.CAMERA_FOV_DEGREES,
        focal_length: float = config.DEFAULT_FOCAL_LENGTH,
    ):
        if not os.path.exists(model_path):
            candidates = [
                os.path.join(_ROOT_DIR, "models", os.path.basename(model_path)),
                os.path.join(_ROOT_DIR, "models", "best.pt"),
                os.path.join(_ROOT_DIR, "models", "gpModel.pt"),
                os.path.join(_ROOT_DIR, "models", "yolo26n.pt"),
                os.path.join("models", model_path),
                "best.pt",
                "gpModel.pt",
            ]
            for c in candidates:
                if os.path.exists(c):
                    model_path = c
                    break

        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        runtime_path = resolve_accelerated_weights(model_path, self.device)
        try:
            self.model = YOLO(runtime_path, task="detect")
        except Exception:
            self.model = YOLO(model_path)
        self.fov_deg = fov_deg
        self.focal_length = focal_length

    def calculate_distance(self, bbox: List[int], frame_height: int, label: str) -> float:
        dist, _, _ = estimate_optical_distance(
            bbox,
            label,
            focal_length=self.focal_length,
            frame_height=frame_height,
        )
        return dist

    def calculate_position(self, bbox: List[int], frame_width: int) -> str:
        center_x = (bbox[0] + bbox[2]) / 2.0
        norm_x = center_x / float(max(1, frame_width))
        if norm_x < config.ZONE_LEFT_LIMIT:
            return "LEFT"
        elif norm_x > config.ZONE_RIGHT_LIMIT:
            return "RIGHT"
        return "FORWARD"

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.30,
    ) -> Tuple[List[DetectionResult], Optional[DetectionResult]]:
        frame_h, frame_w = frame.shape[:2]
        with torch.inference_mode():
            results = self.model.predict(
                frame,
                conf=conf_threshold,
                iou=0.45,
                device=self.device,
                verbose=False,
            )
            result = results[0]

        detections: List[DetectionResult] = []
        nearest_obj: Optional[DetectionResult] = None
        min_dist = float("inf")

        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            label = result.names.get(cls_id, f"obj_{cls_id}")
            confidence = float(box.conf[0].item())
            xyxy = [int(x) for x in box.xyxy[0].tolist()]

            dist = self.calculate_distance(xyxy, frame_h, label)
            pos = self.calculate_position(xyxy, frame_w)

            det = DetectionResult(
                label=label,
                confidence=confidence,
                bbox=xyxy,
                distance=dist,
                position=pos,
            )
            detections.append(det)

            if dist < min_dist:
                min_dist = dist
                nearest_obj = det

        return detections, nearest_obj


# ==============================================================================
# 3. OBSTACLE DETECTOR (DÀNH CHO FASTAPI WEB APP WEB_APP.PY)
# ==============================================================================

class ObstacleDetector:
    def __init__(self, model_path="models/best.pt", conf_threshold=0.25, focal_length=650):
        if not os.path.exists(model_path):
            candidates = [
                os.path.join(_ROOT_DIR, "models", os.path.basename(model_path)),
                os.path.join(_ROOT_DIR, "models", "best.pt"),
                os.path.join(_ROOT_DIR, "models", "gpModel.pt"),
                os.path.join(_ROOT_DIR, "models", "yolo26n.pt"),
                os.path.join("models", os.path.basename(model_path)),
                os.path.basename(model_path),
                "best.pt",
                "gpModel.pt",
                "yolo26n.pt",
            ]
            for c in candidates:
                if os.path.exists(c):
                    model_path = os.path.abspath(c)
                    break

        self.model_path = os.path.abspath(model_path) if os.path.exists(model_path) else model_path
        self.conf_threshold = conf_threshold
        self.focal_length = focal_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._lock = threading.Lock()
        self._model = None
        # Heuristic before lazy load, immediately refined once loaded
        self.is_custom_model = "best" in os.path.basename(self.model_path).lower()

    @property
    def model(self):
        if self._model is None:
            runtime_path = resolve_accelerated_weights(self.model_path, self.device)
            try:
                self._model = YOLO(runtime_path, task="detect")
            except Exception:
                self._model = YOLO(self.model_path)
            self.is_custom_model = self._detect_model_type(self._model)
        return self._model

    @staticmethod
    def _detect_model_type(yolo_model) -> bool:
        names = list(yolo_model.names.values()) if hasattr(yolo_model, "names") else []
        return any(name in CUSTOM_CLASS_INFO for name in names)

    def warmup(self):
        """Pre-warms inference kernels at fixed 320px streaming shape so there is zero cold-start or shape re-allocation."""
        try:
            dummy = np.zeros((192, 320, 3), dtype=np.uint8)
            with self._lock:
                with torch.inference_mode():
                    self.model.predict(
                        dummy,
                        conf=0.5,
                        imgsz=320,
                        device=self.device,
                        verbose=False,
                    )
        except Exception as e:
            print(f"[Detector] Warmup note: {e}")

    def ensure_loaded(self):
        with self._lock:
            _ = self.model
        return self.is_custom_model

    def load_model(self, model_path: str):
        if not os.path.exists(model_path):
            candidates = [
                os.path.join(_ROOT_DIR, "models", os.path.basename(model_path)),
                os.path.join("models", os.path.basename(model_path)),
                os.path.join(_ROOT_DIR, os.path.basename(model_path)),
                model_path,
            ]
            for c in candidates:
                if os.path.exists(c):
                    model_path = os.path.abspath(c)
                    break

        with self._lock:
            self.model_path = model_path
            runtime_path = resolve_accelerated_weights(model_path, self.device)
            try:
                self._model = YOLO(runtime_path, task="detect")
            except Exception:
                self._model = YOLO(model_path)
            self.is_custom_model = self._detect_model_type(self._model)
        self.warmup()

    def calculate_distance(self, cords_or_h, raw_label, focal_length=None, frame_height=None):
        fl = focal_length if focal_length is not None else self.focal_length
        fh = frame_height if frame_height and frame_height > 0 else 720
        return estimate_optical_distance(cords_or_h, raw_label, focal_length=fl, frame_height=fh)

    @staticmethod
    def get_position(frame_width: int, cords: List[int]) -> str:
        x1, _, x2, _ = cords
        x_center = (x1 + x2) * 0.5
        w = max(1.0, float(frame_width))
        ratio = x_center / w
        if ratio < config.ZONE_LEFT_LIMIT:
            return "bên trái"
        elif ratio > config.ZONE_RIGHT_LIMIT:
            return "bên phải"
        return "chính giữa"

    @staticmethod
    def format_distance_speech(distance):
        if distance is None:
            return ""
        d = round(float(distance), 1)
        if d < 1.0:
            return "cách chưa tới 1 mét" if d >= 0.7 else f"cách {d:.1f} mét".replace(".", " phẩy ")
        elif abs(d - 1.5) < 0.05:
            return "cách 1 mét rưỡi"
        elif abs(d - 2.5) < 0.05:
            return "cách 2 mét rưỡi"
        elif abs(d - 3.5) < 0.05:
            return "cách 3 mét rưỡi"
        elif d.is_integer() or abs(d - round(d)) < 0.15:
            return f"cách {int(round(d))} mét"
        else:
            return f"cách {d:.1f} mét".replace(".", " phẩy ")

    @staticmethod
    def generate_alert_message(label, direction, distance_level, distance=None):
        dir_speech = "phía trước" if direction == "chính giữa" else direction
        lbl = str(label).strip()
        lbl_lower = lbl.lower()
        dist_speech = ObstacleDetector.format_distance_speech(distance)

        if "ổ gà" in lbl_lower or "o ga" in lbl_lower:
            if dist_speech:
                return f"Cảnh báo, phát hiện ổ gà {dir_speech}, {dist_speech}."
            return f"Cảnh báo, phát hiện ổ gà {dir_speech}."

        if distance_level == "Rất gần" or (distance is not None and distance <= 1.5):
            if dist_speech:
                return f"Cảnh báo nguy hiểm, có {lbl_lower} ở {dir_speech}, {dist_speech}."
            return f"Cảnh báo nguy hiểm, có {lbl_lower} rất gần ở {dir_speech}."

        if lbl_lower in ["ô tô", "xe ô tô", "xe tải", "xe buýt"]:
            if dist_speech:
                return f"Có {lbl_lower} ở {dir_speech}, {dist_speech}, hãy chú ý."
            return f"Có {lbl_lower} ở {dir_speech}, hãy chú ý."

        if "người" in lbl_lower:
            if dist_speech:
                return f"Phát hiện người ở {dir_speech}, {dist_speech}."
            return f"Phát hiện người ở {dir_speech}."

        if dist_speech:
            return f"Có {lbl_lower} ở {dir_speech}, {dist_speech}."
        return f"Có {lbl_lower} ở {dir_speech}."

    @staticmethod
    def blur_person(image, cords):
        x1, y1, x2, y2 = cords
        h = y2 - y1
        blur_h = max(int(0.22 * h), 1)
        y1_clip = max(0, y1)
        y2_blur = min(image.shape[0], y1 + blur_h)
        x1_clip = max(0, x1)
        x2_clip = min(image.shape[1], x2)

        top_region = image[y1_clip:y2_blur, x1_clip:x2_clip]
        if top_region.size > 0 and top_region.shape[0] > 1 and top_region.shape[1] > 1:
            image[y1_clip:y2_blur, x1_clip:x2_clip] = cv2.GaussianBlur(top_region, (25, 25), 0)
        return image

    def process_frame(
        self,
        frame,
        danger_distance=3.0,
        enable_blur=True,
        conf_threshold=None,
        annotate=True,
        imgsz=None,
    ):
        if frame is None:
            return None, [], None

        h, w = frame.shape[:2]
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        target_imgsz = imgsz if imgsz is not None else 640

        with self._lock:
            with torch.inference_mode():
                results = self.model.predict(
                    frame,
                    conf=conf,
                    iou=0.45,
                    imgsz=target_imgsz,
                    device=self.device,
                    verbose=False,
                )
                result = results[0]
                names = dict(result.names) if hasattr(result, "names") else dict(self.model.names)
                boxes = result.boxes
                xyxy_arr = []
                cls_arr = []
                conf_arr = []
                if len(boxes) > 0:
                    xyxy_arr = boxes.xyxy.cpu().numpy().astype(int)
                    cls_arr = boxes.cls.cpu().numpy().astype(int)
                    conf_arr = boxes.conf.cpu().numpy().astype(float)
                del results, result, boxes

        detections = []
        badges = []

        if annotate:
            # Draw subtle spatial 3-zone lines on annotated static images
            left_x = int(w * config.ZONE_LEFT_LIMIT)
            right_x = int(w * config.ZONE_RIGHT_LIMIT)
            zone_overlay = frame.copy()
            cv2.line(zone_overlay, (left_x, 0), (left_x, h), (180, 160, 120), 1, cv2.LINE_AA)
            cv2.line(zone_overlay, (right_x, 0), (right_x, h), (180, 160, 120), 1, cv2.LINE_AA)
            cv2.addWeighted(zone_overlay, 0.35, frame, 0.65, 0, frame)

        if len(xyxy_arr) > 0:
            for i in range(len(xyxy_arr)):
                x1, y1, x2, y2 = [int(v) for v in xyxy_arr[i]]
                cords = [x1, y1, x2, y2]
                raw_label = str(names.get(int(cls_arr[i]), f"class_{cls_arr[i]}"))
                conf_val = round(float(conf_arr[i]), 2)

                info = PRECOMPUTED_LOOKUP.get(
                    raw_label,
                    {
                        "vn_name": raw_label,
                        "real_height": DEFAULT_HEIGHT,
                        "category": "obstacle",
                        "color": (246, 130, 59),
                    },
                )

                vn_name = str(info["vn_name"])
                category = str(info["category"])
                color = info.get("color", (246, 130, 59))

                distance, distance_level, is_reliable = self.calculate_distance(
                    cords, raw_label, frame_height=h
                )
                direction = str(self.get_position(w, cords))
                is_danger = bool(
                    distance <= danger_distance
                    or (raw_label in ["O ga", "Ổ gà"] and distance <= 4.0)
                )

                alert_msg = self.generate_alert_message(
                    vn_name, direction, distance_level, distance
                )

                if category == "person" and enable_blur and annotate:
                    frame = self.blur_person(frame, cords)

                if annotate:
                    box_color = (68, 68, 239) if is_danger else color
                    draw_corner_box(frame, x1, y1, x2, y2, box_color, is_danger=is_danger)
                    badges.append({
                        "text": f"{vn_name} • {distance:.1f}m",
                        "x": x1,
                        "y": y1,
                        "color_bgr": box_color,
                        "is_danger": is_danger,
                    })

                detection_item = {
                    "label": vn_name,
                    "raw_label": raw_label,
                    "category": category,
                    "confidence": conf_val,
                    "distance": distance,
                    "distance_level": distance_level,
                    "distance_type": "ước lượng",
                    "is_reliable": is_reliable,
                    "is_ground_object": bool(info.get("is_ground", False)),
                    "direction": direction,
                    "alert_message": alert_msg,
                    "center_x_ratio": round(float(((x1 + x2) * 0.5) / max(1, w)), 2),
                    "box": cords,
                    "is_dangerous": is_danger,
                    "Vật thể": vn_name,
                    "Hướng": direction,
                    "Khoảng cách (m)": distance,
                    "Mức cự ly": distance_level,
                    "Mức độ": "Nguy hiểm" if is_danger else "An toàn",
                }
                detections.append(detection_item)

        most_dangerous = None
        dangerous_items = [d for d in detections if d["is_dangerous"]]
        if dangerous_items:
            dangerous_items.sort(key=lambda d: d["distance"])
            most_dangerous = dangerous_items[0]

        if annotate:
            banner_str = None
            if most_dangerous:
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (68, 68, 239), 5)
                dir_txt = "PHÍA TRƯỚC" if most_dangerous["direction"] == "chính giữa" else most_dangerous["direction"].upper()
                banner_str = (
                    f"CẢNH BÁO NGUY HIỂM: {most_dangerous['label'].upper()} Ở {dir_txt} "
                    f"({most_dangerous['distance']}m - {most_dangerous['distance_level']})"
                )
            render_unicode_batch(
                frame,
                badges=badges,
                danger_banner=banner_str,
                zone_labels=("TRÁI", "PHÍA TRƯỚC", "PHẢI"),
            )

        return frame, detections, most_dangerous
