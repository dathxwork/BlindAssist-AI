import sys
import os

# Tối ưu luồng CPU (chừa 1 nhân vật lý cho trình duyệt giải mã Video 60 FPS mượt mà, chống tranh chấp luồng)
os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("MKL_NUM_THREADS", "3")

import io
import json
import time
import base64
import hashlib
import asyncio
import gc
import traceback
from typing import Optional, List, Dict, Any
from PIL import Image
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from pydantic import BaseModel
from gtts import gTTS

# Thư mục gốc & đường dẫn mô-đun
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.detector import ObstacleDetector, CUSTOM_CLASS_INFO, COCO_CLASS_INFO

# Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title="BlindAssist AI - Hệ thống Hỗ trợ Người Khiếm thị",
    description="Ứng dụng Web AI phát hiện vật cản, ước lượng khoảng cách quang học và cảnh báo đa phương thức tiếng Việt",
    version="3.0.0",
)

# Thêm CORS Middleware để cho phép truy cập từ thiết bị di động cùng mạng LAN
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo đường dẫn các thư mục
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
AUDIO_CACHE_DIR = os.path.join(BASE_DIR, "audio_cache")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

for _d in (TEMPLATES_DIR, STATIC_DIR, RESULTS_DIR, AUDIO_CACHE_DIR, MODELS_DIR, DATA_DIR):
    os.makedirs(_d, exist_ok=True)

# Gắn thư mục static, results và audio_cache
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")
app.mount("/audio_cache", StaticFiles(directory=AUDIO_CACHE_DIR), name="audio_cache")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

MODEL_DESCRIPTIONS = {
    "best.pt": "best.pt — 10 Lớp Vật Cản Đường Phố VN (Chuyên dụng)",
    "gpModel.pt": "gpModel.pt — YOLOv8n 80 Lớp Quốc Tế (COCO)",
    "yolo26n.pt": "yolo26n.pt — YOLO26n 80 Lớp Thế Hệ Mới (COCO)",
    "yolo11n.pt": "yolo11n.pt — YOLO11n 80 Lớp Nhẹ & Nhanh (COCO)",
    "yolov8n.pt": "yolov8n.pt — YOLOv8n 80 Lớp Tiêu Chuẩn (COCO)",
}


def discover_available_models() -> List[Dict[str, Any]]:
    """Tự động quét toàn bộ các trọng số mô hình (.pt) khả dụng trong models/ và thư mục gốc."""
    found: Dict[str, str] = {}
    for folder in (MODELS_DIR, BASE_DIR):
        if os.path.exists(folder):
            for fname in sorted(os.listdir(folder)):
                if fname.endswith(".pt") and fname not in found:
                    fpath = os.path.join(folder, fname)
                    if os.path.isfile(fpath):
                        found[fname] = fpath

    # Ưu tiên sắp xếp best.pt lên đầu
    priority_order = ["best.pt", "gpModel.pt", "yolo26n.pt", "yolo11n.pt", "yolov8n.pt"]
    ordered_names = sorted(
        found.keys(),
        key=lambda x: (priority_order.index(x) if x in priority_order else 99, x.lower()),
    )

    models_list = []
    for name in ordered_names:
        path = found[name]
        try:
            size_mb = round(os.path.getsize(path) / (1024 * 1024), 1)
        except OSError:
            size_mb = 0.0
        desc = MODEL_DESCRIPTIONS.get(name, f"{name} ({size_mb} MB)")
        models_list.append({
            "name": name,
            "label": f"{desc} [{size_mb} MB]" if size_mb > 0 else desc,
            "size_mb": size_mb,
            "is_custom": "best" in name.lower(),
        })
    return models_list


def parse_bool_param(val: Optional[str], default: bool) -> bool:
    if val is None:
        return default
    v = str(val).strip().lower()
    if v in ("1", "true", "on", "yes"):
        return True
    if v in ("0", "false", "off", "no"):
        return False
    return default


def build_class_details(is_custom: bool) -> List[Dict[str, Any]]:
    src_map = CUSTOM_CLASS_INFO if is_custom else COCO_CLASS_INFO
    details = []
    for i, (k, v) in enumerate(src_map.items()):
        details.append({
            "id": i,
            "raw_name": k,
            "display_name": v["vn_name"],
            "real_height": v.get("real_height"),
            "is_ground": bool(v.get("is_ground", False)),
            "method": "Hình học mặt đường (Ground-plane)" if v.get("is_ground") else "Quang học lỗ kim (Focal Length)",
        })
    return details


# Khởi tạo Detector (Singleton)
DEFAULT_MODEL = (
    os.path.join(MODELS_DIR, "best.pt")
    if os.path.exists(os.path.join(MODELS_DIR, "best.pt"))
    else "best.pt"
)
print(f"[*] Khởi động Detector với mô hình: {DEFAULT_MODEL}")
detector = ObstacleDetector(model_path=DEFAULT_MODEL, conf_threshold=0.25, focal_length=650)


@app.on_event("startup")
async def startup_warmup():
    """Làm nóng trước mô hình AI trên luồng nền để khung hình đầu tiên đạt tốc độ tối đa ngay lập tức."""
    asyncio.create_task(asyncio.to_thread(detector.warmup))

# Cấu hình hệ thống mặc định
system_settings = {
    "danger_distance": 3.0,       # Khoảng cách cảnh báo nguy hiểm (m)
    "conf_threshold": 0.25,       # Ngưỡng tin cậy nhận diện
    "enable_blur": True,          # Làm mờ khuôn mặt người đi bộ
    "focal_length": 650,          # Tiêu cự camera giả định (px)
    "speech_cooldown": 2.8,       # Thời gian giãn cách cảnh báo giọng nói (s)
    "speech_rate": 1.0,           # Tốc độ đọc giọng nói (0.5x - 2.0x)
    "enable_voice": True,         # Bật/tắt giọng nói TTS
    "enable_sonar": True,         # Bật/tắt âm thanh radar tiệm cận
    "show_zones": True,           # Hiển thị vạch chia 3 vùng không gian (Trái/Giữa/Phải)
    "voice_engine": "auto",       # "auto" | "google" | "browser"
    "distance_levels": {
        "very_close": 1.5,
        "close": 3.0,
        "medium": 5.0,
    },
}


class FrameRequest(BaseModel):
    frame_data: str
    danger_distance: Optional[float] = None
    conf_threshold: Optional[float] = None
    enable_blur: Optional[bool] = None
    return_image: bool = False
    imgsz: Optional[int] = None


class VideoSummaryRequest(BaseModel):
    video_name: str
    total_frames: int
    duration_seconds: float
    dangerous_count: int
    most_dangerous_label: Optional[str] = None
    most_dangerous_direction: Optional[str] = None
    most_dangerous_distance: Optional[float] = None
    snapshot_data: Optional[str] = None


# --- CÁC ROUTE GIAO DIỆN VÀ API ---

@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    """Trang chủ ứng dụng Web AI BlindAssist"""
    models_meta = discover_available_models()
    active_model_name = os.path.basename(detector.model_path)
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "active_model": detector.model_path,
            "active_model_name": active_model_name,
            "available_models": models_meta,
            "is_custom_model": detector.is_custom_model,
            "class_details": build_class_details(detector.is_custom_model),
            "settings": system_settings,
        },
    )


@app.get("/api/status")
async def get_status():
    """Lấy trạng thái hệ thống, danh sách mô hình và lớp đối tượng hiện hành"""
    models_meta = discover_available_models()
    samples = [
        f
        for f in ["test_video.mp4", "test_video1.mp4", "test_video2.mp4", "test_image.jpg"]
        if os.path.exists(os.path.join(DATA_DIR, f)) or os.path.exists(os.path.join(BASE_DIR, f))
    ]
    is_custom = detector.is_custom_model
    return {
        "status": "online",
        "device": detector.device,
        "active_model": detector.model_path,
        "active_model_name": os.path.basename(detector.model_path),
        "is_custom_model": is_custom,
        "available_models": [m["name"] for m in models_meta],
        "models_metadata": models_meta,
        "sample_files": samples,
        "classes": list(CUSTOM_CLASS_INFO.keys()) if is_custom else list(COCO_CLASS_INFO.keys()),
        "class_details": build_class_details(is_custom),
        "settings": system_settings,
    }


@app.post("/api/update_settings")
async def update_settings(
    danger_distance: Optional[float] = Form(None),
    conf_threshold: Optional[float] = Form(None),
    enable_blur: Optional[str] = Form(None),
    enable_sonar: Optional[str] = Form(None),
    show_zones: Optional[str] = Form(None),
    focal_length: Optional[int] = Form(None),
    speech_rate: Optional[float] = Form(None),
    speech_cooldown: Optional[float] = Form(None),
    voice_engine: Optional[str] = Form(None),
    model_name: Optional[str] = Form(None),
    form_submitted: Optional[str] = Form(None),
):
    """Cập nhật toàn diện các tham số thuật toán và mô hình AI"""
    global system_settings, detector

    if danger_distance is not None:
        system_settings["danger_distance"] = max(0.5, min(15.0, float(danger_distance)))
    if conf_threshold is not None:
        val_conf = max(0.10, min(0.95, float(conf_threshold)))
        system_settings["conf_threshold"] = val_conf
        detector.conf_threshold = val_conf
    if focal_length is not None:
        val_fl = max(300, min(1500, int(focal_length)))
        system_settings["focal_length"] = val_fl
        detector.focal_length = val_fl
    if speech_rate is not None:
        system_settings["speech_rate"] = max(0.5, min(2.0, float(speech_rate)))
    if speech_cooldown is not None:
        system_settings["speech_cooldown"] = max(1.0, min(10.0, float(speech_cooldown)))
    if voice_engine is not None and voice_engine in ("auto", "google", "browser"):
        system_settings["voice_engine"] = voice_engine

    # Xử lý chính xác checkbox boolean (kể cả khi trình duyệt bỏ trống checkbox tắt)
    if enable_blur is not None:
        system_settings["enable_blur"] = parse_bool_param(enable_blur, system_settings["enable_blur"])
    elif form_submitted == "1":
        system_settings["enable_blur"] = False

    if enable_sonar is not None:
        system_settings["enable_sonar"] = parse_bool_param(enable_sonar, system_settings["enable_sonar"])
    elif form_submitted == "1":
        system_settings["enable_sonar"] = False

    if show_zones is not None:
        system_settings["show_zones"] = parse_bool_param(show_zones, system_settings["show_zones"])
    elif form_submitted == "1":
        system_settings["show_zones"] = False

    if model_name:
        safe_name = os.path.basename(model_name)
        cand = (
            os.path.join(MODELS_DIR, safe_name)
            if os.path.exists(os.path.join(MODELS_DIR, safe_name))
            else os.path.join(BASE_DIR, safe_name)
        )
        if os.path.exists(cand) and os.path.abspath(cand) != os.path.abspath(detector.model_path):
            await asyncio.to_thread(detector.load_model, cand)

    return {
        "status": "success",
        "settings": system_settings,
        "active_model": detector.model_path,
        "active_model_name": os.path.basename(detector.model_path),
        "is_custom_model": detector.is_custom_model,
        "class_details": build_class_details(detector.is_custom_model),
    }


@app.post("/api/detect_frame")
async def detect_frame(req: FrameRequest):
    """
    Xử lý 1 khung hình từ webcam hoặc video trình duyệt (JSON Base64).
    Chạy suy luận qua asyncio.to_thread để không chặn Event Loop.
    """
    start_time = time.time()
    try:
        frame_data = req.frame_data
        if "," in frame_data:
            frame_data = frame_data.split(",", 1)[1]
        img_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return JSONResponse({"error": "Không thể giải mã hình ảnh"}, status_code=400)

        dd = req.danger_distance if req.danger_distance is not None else system_settings["danger_distance"]
        ct = req.conf_threshold if req.conf_threshold is not None else system_settings["conf_threshold"]
        eb = req.enable_blur if req.enable_blur is not None else system_settings["enable_blur"]

        processed_frame, detections, most_dangerous = await asyncio.to_thread(
            detector.process_frame,
            frame,
            danger_distance=dd,
            enable_blur=eb,
            conf_threshold=ct,
            annotate=req.return_image,
            imgsz=req.imgsz,
        )

        proc_time = round((time.time() - start_time) * 1000, 1)
        fps = round(1000.0 / max(proc_time, 1.0), 1)

        result_payload = {
            "success": True,
            "latency_ms": proc_time,
            "fps": fps,
            "detections_count": len(detections),
            "detections": detections,
            "most_dangerous": most_dangerous,
            "frame_width": frame.shape[1],
            "frame_height": frame.shape[0],
        }

        if req.return_image and processed_frame is not None:
            _, buffer = cv2.imencode(".jpg", processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            result_payload["processed_image"] = (
                "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")
            )

        return JSONResponse(result_payload)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


def load_and_resize_image(source, max_dim=960):
    """
    Tải và chuẩn hóa kích thước ảnh an toàn tuyệt đối (tối đa 960px cho ảnh sắc nét).
    Sử dụng Pillow draft mode cho JPEG giúp tiết kiệm RAM tối đa.
    """
    try:
        fp = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else str(source)
        with Image.open(fp) as img:
            if hasattr(img, "draft") and getattr(img, "format", "") in ("JPEG", "JPG", "MPO"):
                try:
                    img.draft("RGB", (max_dim, max_dim))
                except Exception:
                    pass
            if img.mode != "RGB":
                img = img.convert("RGB")
            w, h = img.size
            if max(w, h) > max_dim:
                scale = float(max_dim) / float(max(w, h))
                new_w, new_h = max(int(w * scale), 1), max(int(h * scale), 1)
                img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
            rgb_arr = np.array(img)
            return cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
    except Exception:
        if isinstance(source, (bytes, bytearray)):
            nparr = np.frombuffer(source, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            frame = cv2.imread(str(source))
        if frame is not None and max(frame.shape[:2]) > max_dim:
            h_o, w_o = frame.shape[:2]
            sc = float(max_dim) / float(max(h_o, w_o))
            frame = cv2.resize(frame, (int(w_o * sc), int(h_o * sc)), interpolation=cv2.INTER_AREA)
        return frame


@app.post("/api/detect_image")
async def detect_image(
    file: Optional[UploadFile] = File(None),
    sample_name: Optional[str] = Form(None),
    danger_distance: Optional[float] = Form(None),
    conf_threshold: Optional[float] = Form(None),
    enable_blur: Optional[str] = Form(None),
):
    """
    Xử lý ảnh tĩnh (tải lên hoặc chọn ảnh mẫu).
    Trả về ảnh đã vẽ khung bao Unicode Tiếng Việt sắc nét, danh sách vật thể và thông tin chi tiết.
    """
    start_time = time.time()
    try:
        frame = None
        if file is not None:
            contents = await file.read()
            frame = load_and_resize_image(contents, max_dim=960)
        elif sample_name:
            safe_sample = os.path.basename(sample_name)
            file_path = os.path.join(DATA_DIR, safe_sample)
            if not os.path.exists(file_path):
                file_path = os.path.join(BASE_DIR, safe_sample)
            if os.path.exists(file_path):
                frame = load_and_resize_image(file_path, max_dim=960)

        if frame is None:
            return JSONResponse({"error": "Không tìm thấy dữ liệu hình ảnh"}, status_code=400)

        gc.collect()

        dd = danger_distance if danger_distance is not None else system_settings["danger_distance"]
        ct = conf_threshold if conf_threshold is not None else system_settings["conf_threshold"]
        eb = parse_bool_param(enable_blur, system_settings["enable_blur"])

        processed_frame, detections, most_dangerous = await asyncio.to_thread(
            detector.process_frame,
            frame,
            danger_distance=dd,
            enable_blur=eb,
            conf_threshold=ct,
            annotate=True,
            imgsz=640,
        )

        gc.collect()

        _, buffer = cv2.imencode(".jpg", processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        img_base64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

        proc_time = round((time.time() - start_time) * 1000, 1)

        if most_dangerous:
            summary_text = (
                most_dangerous.get("alert_message")
                or f"Cảnh báo nguy hiểm: Có {most_dangerous['label']} ở {most_dangerous['direction']} "
                f"(cách {most_dangerous['distance']}m - {most_dangerous.get('distance_level', '')})."
            )
        elif len(detections) > 0:
            nearest = min(detections, key=lambda x: x["distance"])
            summary_text = f"Phát hiện {len(detections)} vật thể. {nearest.get('alert_message', '')}"
        else:
            summary_text = "Không phát hiện vật cản trong tầm quan sát. Đường đi an toàn."

        saved_filename = "latest_result.jpg"
        save_path = os.path.join(RESULTS_DIR, saved_filename)
        cv2.imwrite(save_path, processed_frame)

        nearest_dist = min((d["distance"] for d in detections), default=None)

        return {
            "success": True,
            "latency_ms": proc_time,
            "processed_image": img_base64,
            "saved_result": {
                "filename": saved_filename,
                "url": f"/results/{saved_filename}?t={int(time.time())}",
                "path": save_path,
            },
            "detections": detections,
            "most_dangerous": most_dangerous,
            "nearest_distance": nearest_dist,
            "summary_text": summary_text,
            "objects_count": len(detections),
            "image_size": f"{frame.shape[1]}×{frame.shape[0]}",
        }

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/save_video_summary")
async def save_video_summary(req: VideoSummaryRequest):
    """Lưu kết quả phân tích tổng kết khi chạy xong video vào results/"""
    try:
        summary_path = os.path.join(RESULTS_DIR, "latest_video_summary.json")
        data = {
            "video_name": req.video_name,
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_frames_analyzed": req.total_frames,
            "duration_seconds": req.duration_seconds,
            "dangerous_events_count": req.dangerous_count,
            "most_dangerous_obstacle": (
                {
                    "label": req.most_dangerous_label,
                    "direction": req.most_dangerous_direction,
                    "distance": req.most_dangerous_distance,
                }
                if req.most_dangerous_label
                else None
            ),
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        snapshot_url = None
        if req.snapshot_data and "," in req.snapshot_data:
            snap_b64 = req.snapshot_data.split(",", 1)[1]
            snap_bytes = base64.b64decode(snap_b64)
            snap_path = os.path.join(RESULTS_DIR, "latest_video_snapshot.jpg")
            with open(snap_path, "wb") as f:
                f.write(snap_bytes)
            snapshot_url = f"/results/latest_video_snapshot.jpg?t={int(time.time())}"

        return {
            "status": "success",
            "message": "Đã lưu báo cáo tổng kết phiên phân tích video",
            "summary_file": "results/latest_video_summary.json",
            "snapshot_url": snapshot_url,
            "data": data,
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/results")
async def list_saved_results():
    """Lấy danh sách các kết quả ảnh/video và báo cáo đã lưu trong thư mục results/"""
    files = []
    video_summary = None
    if os.path.exists(RESULTS_DIR):
        summary_path = os.path.join(RESULTS_DIR, "latest_video_summary.json")
        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    video_summary = json.load(f)
            except Exception:
                video_summary = None

        for f in sorted(os.listdir(RESULTS_DIR), reverse=True):
            if f.endswith((".jpg", ".png", ".mp4")):
                fpath = os.path.join(RESULTS_DIR, f)
                mtime = int(os.path.getmtime(fpath))
                files.append({
                    "filename": f,
                    "url": f"/results/{f}?t={mtime}",
                    "size_kb": round(os.path.getsize(fpath) / 1024, 1),
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime)),
                })
    return {"results_dir": RESULTS_DIR, "files": files, "video_summary": video_summary}


@app.delete("/api/results")
async def clear_saved_results():
    """Xóa các tệp kết quả tạm trong thư mục results/"""
    removed = 0
    if os.path.exists(RESULTS_DIR):
        for f in os.listdir(RESULTS_DIR):
            if f.endswith((".jpg", ".png", ".mp4", ".json")):
                try:
                    os.remove(os.path.join(RESULTS_DIR, f))
                    removed += 1
                except OSError:
                    pass
    return {"status": "success", "removed": removed}


@app.get("/api/sample_media")
async def get_sample_media():
    """Lấy danh sách các media mẫu có sẵn để kiểm thử nhanh"""
    samples = []
    definitions = [
        {"name": "test_image.jpg", "type": "image", "label": "Ảnh mẫu 1 - Ngã tư giao thông đô thị"},
        {"name": "test_video.mp4", "type": "video", "label": "Video mẫu 1 - Đường phố nhiều phương tiện"},
        {"name": "test_video1.mp4", "type": "video", "label": "Video mẫu 2 - Xe máy và người đi bộ"},
        {"name": "test_video2.mp4", "type": "video", "label": "Video mẫu 3 - Vật cản đa dạng lề đường"},
    ]
    for d in definitions:
        p = os.path.join(DATA_DIR, d["name"])
        if not os.path.exists(p):
            p = os.path.join(BASE_DIR, d["name"])
        if os.path.exists(p):
            size_mb = round(os.path.getsize(p) / (1024 * 1024), 2)
            samples.append({**d, "size_mb": size_mb})

    return {"samples": samples}


@app.get("/api/stream_sample/{filename}")
async def stream_sample_media(filename: str, request: Request):
    """Phát luồng video mẫu an toàn hỗ trợ HTTP 206 Partial Content (tua video mượt)"""
    safe_name = os.path.basename(filename)
    file_path = os.path.join(DATA_DIR, safe_name)
    if not os.path.exists(file_path):
        file_path = os.path.join(BASE_DIR, safe_name)
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        return Response("File not found", status_code=404)

    file_size = os.path.getsize(file_path)
    range_header = request.headers.get("range")
    content_type = "video/mp4" if safe_name.lower().endswith(".mp4") else "image/jpeg"

    cors_headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
        "Access-Control-Allow-Headers": "Range, Content-Type",
        "Accept-Ranges": "bytes",
    }

    if range_header:
        byte1, byte2 = 0, None
        match = range_header.replace("bytes=", "").split("-")
        if match[0]:
            byte1 = int(match[0])
        if len(match) > 1 and match[1]:
            byte2 = int(match[1])

        byte1 = max(0, min(byte1, file_size - 1))
        end_byte = min(byte2, file_size - 1) if byte2 is not None else file_size - 1
        length = max(0, end_byte - byte1 + 1)

        def iterfile():
            with open(file_path, "rb") as f:
                f.seek(byte1)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(64 * 1024, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        headers = {
            **cors_headers,
            "Content-Range": f"bytes {byte1}-{end_byte}/{file_size}",
            "Content-Length": str(length),
            "Content-Type": content_type,
        }
        return StreamingResponse(iterfile(), status_code=206, headers=headers)

    return FileResponse(file_path, media_type=content_type, headers=cors_headers)


@app.get("/api/tts")
async def text_to_speech(text: str = Query(..., description="Nội dung cần chuyển thành giọng nói")):
    """
    Text-To-Speech tiếng Việt tự nhiên qua gTTS có lưu đệm đĩa (0ms từ lần phát thứ 2),
    trả về trực tiếp bộ đệm byte đầy đủ để tránh hoàn toàn lỗi HTTP 416 Range Not Satisfiable.
    """
    clean_text = text.strip()
    if not clean_text:
        return JSONResponse({"error": "Empty text"}, status_code=400)

    cache_key = hashlib.md5(clean_text.lower().encode("utf-8")).hexdigest()
    mp3_cache_path = os.path.join(AUDIO_CACHE_DIR, f"{cache_key}.mp3")
    wav_cache_path = os.path.join(AUDIO_CACHE_DIR, f"{cache_key}.wav")

    if os.path.exists(mp3_cache_path):
        if os.path.getsize(mp3_cache_path) > 128:
            with open(mp3_cache_path, "rb") as f:
                audio_bytes = f.read()
            return Response(
                content=audio_bytes,
                media_type="audio/mpeg",
                headers={
                    "Cache-Control": "public, max-age=604800, immutable",
                    "Accept-Ranges": "none",
                    "X-TTS-Source": "BlindAssist-Cache-MP3",
                },
            )
        else:
            try:
                os.remove(mp3_cache_path)
            except OSError:
                pass

    try:
        def generate_gtts():
            tmp_path = mp3_cache_path + ".tmp"
            tts = gTTS(text=clean_text, lang="vi", slow=False)
            tts.save(tmp_path)
            if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 128:
                os.replace(tmp_path, mp3_cache_path)

        await asyncio.to_thread(generate_gtts)
        if os.path.exists(mp3_cache_path) and os.path.getsize(mp3_cache_path) > 128:
            with open(mp3_cache_path, "rb") as f:
                audio_bytes = f.read()
            return Response(
                content=audio_bytes,
                media_type="audio/mpeg",
                headers={
                    "Cache-Control": "public, max-age=604800, immutable",
                    "Accept-Ranges": "none",
                    "X-TTS-Source": "BlindAssist-Google-TTS-Vi",
                },
            )
    except Exception as gtts_err:
        try:
            import pyttsx3

            def generate_offline_wav():
                eng = pyttsx3.init()
                eng.setProperty("rate", 185)
                eng.save_to_file(clean_text, wav_cache_path)
                eng.runAndWait()

            if not os.path.exists(wav_cache_path) or os.path.getsize(wav_cache_path) <= 128:
                await asyncio.to_thread(generate_offline_wav)
            if os.path.exists(wav_cache_path) and os.path.getsize(wav_cache_path) > 128:
                with open(wav_cache_path, "rb") as f:
                    wav_bytes = f.read()
                return Response(
                    content=wav_bytes,
                    media_type="audio/wav",
                    headers={"Accept-Ranges": "none", "X-TTS-Source": "BlindAssist-Offline-SAPI5"},
                )
        except Exception:
            pass
        return JSONResponse({"error": str(gtts_err), "offline": True}, status_code=503)

    return JSONResponse({"error": "TTS generation empty"}, status_code=503)


# Bộ nhớ đệm Keyframe Video trong RAM giúp phát lại & tua lại tức thì 0ms
VIDEO_KEYFRAME_CACHE: Dict[tuple, Dict[str, Any]] = {}


@app.websocket("/ws/video_analyze")
async def websocket_video_analyze(
    websocket: WebSocket,
    filename: str = Query("test_video.mp4"),
):
    """
    Luồng giải mã và suy luận Video trực tiếp từ máy chủ bằng OpenCV C++ tuần tự:
    - Giải mã H.264 trực tiếp trên máy chủ (bỏ qua hoàn toàn bước nén/giải nén Base64 JPEG từ trình duyệt).
    - Đẩy trước (Pre-stream) các Keyframe cách nhau 0.20s về trình duyệt trước thời điểm phát thực tế,
      giúp trình duyệt nội suy 60 FPS với độ trễ hiển thị bằng 0.0ms tuyệt đối.
    """
    await websocket.accept()
    safe_name = os.path.basename(filename)
    file_path = os.path.join(DATA_DIR, safe_name)
    if not os.path.exists(file_path):
        file_path = os.path.join(BASE_DIR, safe_name)
    if not os.path.exists(file_path):
        await websocket.send_json({"type": "error", "error": f"Không tìm thấy video: {safe_name}"})
        await websocket.close()
        return

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        await websocket.send_json({"type": "error", "error": "Không thể mở luồng video"})
        await websocket.close()
        return

    try:
        fps_src = cap.get(cv2.CAP_PROP_FPS)
        if not fps_src or fps_src <= 0 or fps_src != fps_src:
            fps_src = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = (total_frames / fps_src) if total_frames > 0 else 60.0
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        cap_w = 640
        cap_h = max(1, int(round(cap_w * (vid_h / max(1, vid_w)))))

        step_frames = max(1, int(round(fps_src * 0.20)))  # 1 keyframe mỗi 0.20s (5 Hz)

        await websocket.send_json({
            "type": "meta",
            "filename": safe_name,
            "duration": round(duration, 2),
            "fps_src": round(fps_src, 2),
            "capW": cap_w,
            "capH": cap_h,
        })

        client_time = 0.0
        client_playing = False
        cur_frame_idx = 0
        need_seek = False
        seek_target_sec = 0.0

        def read_frame_at_step(current_idx: int, target_idx: int):
            if target_idx < current_idx or (target_idx - current_idx) > int(fps_src * 2.0):
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
                ok, frm = cap.read()
                return ok, frm, target_idx + 1
            while current_idx < target_idx:
                if not cap.grab():
                    return False, None, current_idx
                current_idx += 1
            ok, frm = cap.retrieve()
            if not ok:
                ok, frm = cap.read()
            else:
                current_idx += 1
            return ok, frm, current_idx

        target_idx = 0
        decoder_pos = 0

        while True:
            # Đọc toàn bộ thông điệp điều khiển mới nhất từ trình duyệt (không chặn)
            while True:
                try:
                    msg = await asyncio.wait_for(websocket.receive_json(), timeout=0.001)
                    action = msg.get("action")
                    if action == "seek":
                        seek_target_sec = max(0.0, float(msg.get("time", 0.0)))
                        client_time = seek_target_sec
                        need_seek = True
                    elif action == "pos":
                        client_time = max(0.0, float(msg.get("time", client_time)))
                        client_playing = bool(msg.get("playing", client_playing))
                except asyncio.TimeoutError:
                    break

            if need_seek:
                need_seek = False
                aligned_step = int(seek_target_sec * fps_src) // step_frames
                target_idx = max(0, aligned_step * step_frames)

            target_sec = target_idx / fps_src

            # Nếu đã kết thúc video hoặc đã đệm trước đủ xa (6.0s khi đang phát, 3.6s khi đang dừng), tạm nghỉ nhẹ
            max_lead = 6.0 if client_playing else 3.6
            if (total_frames > 0 and target_idx >= total_frames) or (target_sec > client_time + max_lead):
                await asyncio.sleep(0.06)
                continue

            # Nếu người dùng phát/tua vượt qua vị trí đang quét, nhảy ngay tới sát vị trí hiện tại để không bao giờ trễ
            if client_playing and target_sec < client_time - 0.25:
                aligned_step = int(client_time * fps_src) // step_frames
                target_idx = max(0, aligned_step * step_frames)
                target_sec = target_idx / fps_src

            dd = system_settings["danger_distance"]
            ct = system_settings["conf_threshold"]
            model_key = os.path.basename(detector.model_path)
            cache_key = (safe_name, model_key, round(ct, 2), round(dd, 1), target_idx)

            if cache_key in VIDEO_KEYFRAME_CACHE:
                cached_kf = VIDEO_KEYFRAME_CACHE[cache_key]
                await websocket.send_json(cached_kf)
                target_idx += step_frames
                await asyncio.sleep(0.002)
                continue

            t0 = time.perf_counter()
            ok, raw_frame, decoder_pos = await asyncio.to_thread(read_frame_at_step, decoder_pos, target_idx)
            if not ok or raw_frame is None:
                await asyncio.sleep(0.08)
                continue

            small_frame = cv2.resize(raw_frame, (cap_w, cap_h), interpolation=cv2.INTER_AREA)
            _, detections, most_dangerous = await asyncio.to_thread(
                detector.process_frame,
                small_frame,
                danger_distance=dd,
                enable_blur=False,
                conf_threshold=ct,
                annotate=False,
                imgsz=320,
            )
            infer_ms = round((time.perf_counter() - t0) * 1000, 1)

            kf_payload = {
                "type": "keyframe",
                "t": round(target_sec, 3),
                "fps": round(1000.0 / max(infer_ms, 1.0), 1),
                "latency_ms": infer_ms,
                "detections": detections,
                "most_dangerous": most_dangerous,
                "capW": cap_w,
                "capH": cap_h,
            }
            if len(VIDEO_KEYFRAME_CACHE) > 2500:
                VIDEO_KEYFRAME_CACHE.clear()
            VIDEO_KEYFRAME_CACHE[cache_key] = kf_payload

            await websocket.send_json(kf_payload)
            target_idx += step_frames

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS Video Analyze] Closed: {e}")
    finally:
        cap.release()


# --- WEBSOCKET CHO STREAMING THỜI GIAN THỰC SIÊU TỐC ---

@app.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    """
    Kênh WebSocket 2 chiều siêu tốc cho cả Webcam trực tiếp và Trình phân tích Video.
    Hỗ trợ cả chuỗi JSON có tham số tuỳ chỉnh lẫn chuỗi Base64 thuần túy.
    """
    await websocket.accept()
    try:
        while True:
            raw_msg = await websocket.receive_text()
            start_time = time.time()

            dd = system_settings["danger_distance"]
            ct = system_settings["conf_threshold"]
            eb = system_settings["enable_blur"]
            imgsz = None
            req_id = None
            frame_b64 = raw_msg

            if raw_msg.startswith("{"):
                try:
                    payload = json.loads(raw_msg)
                    frame_b64 = payload.get("frame_data", "")
                    if payload.get("danger_distance") is not None:
                        dd = float(payload["danger_distance"])
                    if payload.get("conf_threshold") is not None:
                        ct = float(payload["conf_threshold"])
                    if payload.get("enable_blur") is not None:
                        eb = bool(payload["enable_blur"])
                    if payload.get("imgsz") is not None:
                        imgsz = int(payload["imgsz"])
                    req_id = payload.get("req_id")
                except Exception:
                    await websocket.send_json({"success": False, "error": "JSON không hợp lệ"})
                    continue

            if "," in frame_b64:
                frame_b64 = frame_b64.split(",", 1)[1]

            try:
                img_bytes = base64.b64decode(frame_b64)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception:
                frame = None

            if frame is None:
                await websocket.send_json({"success": False, "error": "Khung hình không hợp lệ"})
                continue

            _, detections, most_dangerous = await asyncio.to_thread(
                detector.process_frame,
                frame,
                danger_distance=dd,
                enable_blur=eb,
                conf_threshold=ct,
                annotate=False,
                imgsz=imgsz,
            )

            latency = round((time.time() - start_time) * 1000, 1)

            await websocket.send_json({
                "success": True,
                "req_id": req_id,
                "latency_ms": latency,
                "fps": round(1000.0 / max(latency, 1.0), 1),
                "detections_count": len(detections),
                "detections": detections,
                "most_dangerous": most_dangerous,
                "frame_width": frame.shape[1],
                "frame_height": frame.shape[0],
                "width": frame.shape[1],
                "height": frame.shape[0],
            })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WebSocket] Stream closed: {e}")


if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 62)
    print("  HỆ THỐNG BLINDASSIST AI WEB SERVER v3.0")
    print("  Truy cập website tại: http://localhost:8000")
    print("=" * 62 + "\n")
    uvicorn.run("web_app:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
