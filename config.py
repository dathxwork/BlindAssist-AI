"""
Configuration module for BlindAssist (YOLO Object Detection & Audio Alert System).
Synchronized across Desktop OpenCV App (main.py) and FastAPI Web App (web_app.py).
"""

# Camera & Optical Parameters
CAMERA_FOV_DEGREES = 70.0   # Horizontal Field of View (degrees)
DEFAULT_FOCAL_LENGTH = 650  # Reference focal length in pixels at 720p height

# Spatial 3-Zone Thresholds (relative to frame width: [0.0 .. 1.0])
# Left zone:    0.00 to 0.35
# Forward zone: 0.35 to 0.65
# Right zone:   0.65 to 1.00
ZONE_LEFT_LIMIT = 0.35
ZONE_RIGHT_LIMIT = 0.65

# Alert & Distance Thresholds
DEFAULT_WARNING_DISTANCE = 5.0   # Default warning threshold for Desktop CLI (meters)
DEFAULT_DANGER_DISTANCE = 3.0    # Default critical danger threshold for Web UI (meters)
ALERT_COOLDOWN_SECONDS = 2.5     # Minimum time between consecutive voice alerts
SIGNIFICANT_DIST_CHANGE = 0.8    # Significant distance delta (m) to re-alert early

# 4 Distance Risk Assessment Tiers
DISTANCE_TIERS = {
    "VERY_CLOSE": 1.5,  # <= 1.5m: Rất gần (Nguy cơ va chạm cao, cảnh báo khẩn)
    "CLOSE": 3.0,       # 1.5m < d <= 3.0m: Gần (Cần chú ý giảm tốc)
    "MEDIUM": 5.0,      # 3.0m < d <= 5.0m: Trung bình (Cự ly an toàn đang tiến lại)
}


def get_distance_level(distance_m: float, is_reliable: bool = True) -> str:
    """Returns the Vietnamese 4-tier distance risk assessment level."""
    if not is_reliable or distance_m is None or distance_m <= 0:
        return "Chưa xác định"
    if distance_m <= DISTANCE_TIERS["VERY_CLOSE"]:
        return "Rất gần"
    elif distance_m <= DISTANCE_TIERS["CLOSE"]:
        return "Gần"
    elif distance_m <= DISTANCE_TIERS["MEDIUM"]:
        return "Trung bình"
    return "Xa"


# Unified Class Configuration (BGR colors for OpenCV)
# Supports both Custom Vietnamese 10-Class model (best.pt) and COCO 80-Class models (gpModel.pt, yolo26n.pt)
CLASS_CONFIG = {
    # --- Custom Vietnamese 10 Classes (best.pt) ---
    "Bang hieu": {
        "real_height": 1.20,
        "real_width": 0.80,
        "name_vi": "Bảng hiệu",
        "name_en": "Signboard",
        "category": "obstacle",
        "color": (246, 92, 139),
        "blur_privacy": False,
        "is_ground": False,
    },
    "Cay coi": {
        "real_height": 2.50,
        "real_width": 0.60,
        "name_vi": "Cây cối",
        "name_en": "Tree",
        "category": "obstacle",
        "color": (166, 184, 20),
        "blur_privacy": False,
        "is_ground": False,
    },
    "Cot": {
        "real_height": 2.50,
        "real_width": 0.30,
        "name_vi": "Cột điện",
        "name_en": "Pole",
        "category": "obstacle",
        "color": (212, 182, 6),
        "blur_privacy": False,
        "is_ground": False,
    },
    "Den giao thong": {
        "real_height": 2.80,
        "real_width": 0.40,
        "name_vi": "Đèn giao thông",
        "name_en": "Traffic light",
        "category": "obstacle",
        "color": (11, 158, 245),
        "blur_privacy": False,
        "is_ground": False,
    },
    "Nguoi": {
        "real_height": 1.65,
        "real_width": 0.50,
        "name_vi": "Người",
        "name_en": "Person",
        "category": "person",
        "color": (129, 185, 16),
        "blur_privacy": True,
        "is_ground": False,
    },
    "O ga": {
        "real_height": None,
        "real_width": 0.65,
        "name_vi": "Ổ gà",
        "name_en": "Pothole",
        "category": "obstacle",
        "color": (68, 68, 239),
        "blur_privacy": False,
        "is_ground": True,
    },
    "Thung rac": {
        "real_height": 0.90,
        "real_width": 0.50,
        "name_vi": "Thùng rác",
        "name_en": "Trash can",
        "category": "obstacle",
        "color": (184, 163, 148),
        "blur_privacy": False,
        "is_ground": False,
    },
    "Xe dap": {
        "real_height": 1.10,
        "real_width": 0.60,
        "name_vi": "Xe đạp",
        "name_en": "Bicycle",
        "category": "vehicle",
        "color": (246, 130, 59),
        "blur_privacy": False,
        "is_ground": False,
    },
    "Xe hoi": {
        "real_height": 1.55,
        "real_width": 1.80,
        "name_vi": "Xe ô tô",
        "name_en": "Car",
        "category": "vehicle",
        "color": (8, 179, 234),
        "blur_privacy": False,
        "is_ground": False,
    },
    "Xe may": {
        "real_height": 1.15,
        "real_width": 0.70,
        "name_vi": "Xe máy",
        "name_en": "Motorcycle",
        "category": "vehicle",
        "color": (153, 72, 236),
        "blur_privacy": False,
        "is_ground": False,
    },

    # --- COCO Standard Classes (gpModel.pt / yolo26n.pt) ---
    "person": {
        "real_height": 1.65,
        "real_width": 0.50,
        "name_vi": "Người",
        "name_en": "Person",
        "category": "person",
        "color": (129, 185, 16),
        "blur_privacy": True,
        "is_ground": False,
    },
    "car": {
        "real_height": 1.55,
        "real_width": 1.80,
        "name_vi": "Xe ô tô",
        "name_en": "Car",
        "category": "vehicle",
        "color": (8, 179, 234),
        "blur_privacy": False,
        "is_ground": False,
    },
    "motorcycle": {
        "real_height": 1.15,
        "real_width": 0.70,
        "name_vi": "Xe máy",
        "name_en": "Motorcycle",
        "category": "vehicle",
        "color": (153, 72, 236),
        "blur_privacy": False,
        "is_ground": False,
    },
    "bicycle": {
        "real_height": 1.10,
        "real_width": 0.60,
        "name_vi": "Xe đạp",
        "name_en": "Bicycle",
        "category": "vehicle",
        "color": (246, 130, 59),
        "blur_privacy": False,
        "is_ground": False,
    },
    "bus": {
        "real_height": 3.00,
        "real_width": 2.50,
        "name_vi": "Xe buýt",
        "name_en": "Bus",
        "category": "vehicle",
        "color": (11, 158, 245),
        "blur_privacy": False,
        "is_ground": False,
    },
    "truck": {
        "real_height": 2.80,
        "real_width": 2.40,
        "name_vi": "Xe tải",
        "name_en": "Truck",
        "category": "vehicle",
        "color": (0, 140, 255),
        "blur_privacy": False,
        "is_ground": False,
    },
    "traffic light": {
        "real_height": 2.80,
        "real_width": 0.40,
        "name_vi": "Đèn giao thông",
        "name_en": "Traffic light",
        "category": "obstacle",
        "color": (11, 158, 245),
        "blur_privacy": False,
        "is_ground": False,
    },
    "stop sign": {
        "real_height": 1.20,
        "real_width": 0.75,
        "name_vi": "Biển báo dừng",
        "name_en": "Stop sign",
        "category": "obstacle",
        "color": (68, 68, 239),
        "blur_privacy": False,
        "is_ground": False,
    },
    "fire hydrant": {
        "real_height": 0.75,
        "real_width": 0.35,
        "name_vi": "Trụ cứu hỏa",
        "name_en": "Fire hydrant",
        "category": "obstacle",
        "color": (68, 68, 239),
        "blur_privacy": False,
        "is_ground": False,
    },
    "bench": {
        "real_height": 0.85,
        "real_width": 1.40,
        "name_vi": "Ghế dài",
        "name_en": "Bench",
        "category": "obstacle",
        "color": (212, 182, 6),
        "blur_privacy": False,
        "is_ground": False,
    },
    "chair": {
        "real_height": 0.85,
        "real_width": 0.55,
        "name_vi": "Ghế",
        "name_en": "Chair",
        "category": "obstacle",
        "color": (212, 182, 6),
        "blur_privacy": False,
        "is_ground": False,
    },
    "potted plant": {
        "real_height": 0.80,
        "real_width": 0.45,
        "name_vi": "Chậu cây",
        "name_en": "Potted plant",
        "category": "obstacle",
        "color": (166, 184, 20),
        "blur_privacy": False,
        "is_ground": False,
    },
    "dog": {
        "real_height": 0.55,
        "real_width": 0.50,
        "name_vi": "Con chó",
        "name_en": "Dog",
        "category": "obstacle",
        "color": (200, 120, 255),
        "blur_privacy": False,
        "is_ground": False,
    },
    "cat": {
        "real_height": 0.35,
        "real_width": 0.35,
        "name_vi": "Con mèo",
        "name_en": "Cat",
        "category": "obstacle",
        "color": (255, 100, 220),
        "blur_privacy": False,
        "is_ground": False,
    },
    "backpack": {
        "real_height": 0.50,
        "real_width": 0.35,
        "name_vi": "Ba lô",
        "name_en": "Backpack",
        "category": "obstacle",
        "color": (246, 130, 59),
        "blur_privacy": False,
        "is_ground": False,
    },
    "suitcase": {
        "real_height": 0.65,
        "real_width": 0.45,
        "name_vi": "Va li",
        "name_en": "Suitcase",
        "category": "obstacle",
        "color": (246, 130, 59),
        "blur_privacy": False,
        "is_ground": False,
    },
    "umbrella": {
        "real_height": 0.90,
        "real_width": 1.00,
        "name_vi": "Chiếc ô",
        "name_en": "Umbrella",
        "category": "obstacle",
        "color": (246, 92, 139),
        "blur_privacy": False,
        "is_ground": False,
    },
    "dining table": {
        "real_height": 0.75,
        "real_width": 1.20,
        "name_vi": "Bàn",
        "name_en": "Table",
        "category": "obstacle",
        "color": (212, 182, 6),
        "blur_privacy": False,
        "is_ground": False,
    },
}

DEFAULT_OBJECT_COLOR = (246, 130, 59)
NEAREST_OBJECT_COLOR = (68, 68, 239)  # Red in BGR for closest / critical obstacle

# Direction translations
DIRECTIONS = {
    "LEFT": {
        "en": "LEFT",
        "vi": "BÊN TRÁI",
        "vi_lower": "bên trái",
    },
    "FORWARD": {
        "en": "AHEAD",
        "vi": "PHÍA TRƯỚC",
        "vi_lower": "chính giữa",
    },
    "RIGHT": {
        "en": "RIGHT",
        "vi": "BÊN PHẢI",
        "vi_lower": "bên phải",
    },
}
