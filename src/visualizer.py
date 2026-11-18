"""
Visualizer module for drawing modern HUD bounding boxes, privacy blur, zone lanes,
and UTF-8 Vietnamese Unicode labels (fixing OpenCV Hershey ASCII '???' limitation).
"""

from __future__ import annotations
import os
from functools import lru_cache
from typing import List, Optional, Tuple, Dict, Any, TYPE_CHECKING
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config

if TYPE_CHECKING:
    from src.detector import DetectionResult


@lru_cache(maxsize=16)
def get_unicode_font(size: int = 16, bold: bool = True) -> ImageFont.ImageFont:
    """Loads a cached system TrueType font that supports full Vietnamese Unicode."""
    win_fonts = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts")
    candidates = (
        [
            os.path.join(win_fonts, "segoeuib.ttf"),
            os.path.join(win_fonts, "arialbd.ttf"),
            os.path.join(win_fonts, "tahomabd.ttf"),
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        ]
        if bold
        else [
            os.path.join(win_fonts, "segoeui.ttf"),
            os.path.join(win_fonts, "arial.ttf"),
            os.path.join(win_fonts, "tahoma.ttf"),
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
        ]
    )
    for font_path in candidates:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def bgr_to_rgb(color_bgr: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Converts an OpenCV BGR tuple to PIL RGB tuple."""
    return (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))


def draw_corner_box(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color_bgr: Tuple[int, int, int],
    is_danger: bool = False,
) -> None:
    """Draws a modern AI bounding box with emphasized HUD corner brackets."""
    base_thick = 2 if is_danger else 1
    corner_thick = 4 if is_danger else 3
    cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, base_thick, cv2.LINE_AA)

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    clen = max(8, min(22, int(min(bw, bh) * 0.22)))

    # Top-left
    cv2.line(frame, (x1, y1), (x1 + clen, y1), color_bgr, corner_thick, cv2.LINE_AA)
    cv2.line(frame, (x1, y1), (x1, y1 + clen), color_bgr, corner_thick, cv2.LINE_AA)
    # Top-right
    cv2.line(frame, (x2, y1), (x2 - clen, y1), color_bgr, corner_thick, cv2.LINE_AA)
    cv2.line(frame, (x2, y1), (x2, y1 + clen), color_bgr, corner_thick, cv2.LINE_AA)
    # Bottom-left
    cv2.line(frame, (x1, y2), (x1 + clen, y2), color_bgr, corner_thick, cv2.LINE_AA)
    cv2.line(frame, (x1, y2), (x1, y2 - clen), color_bgr, corner_thick, cv2.LINE_AA)
    # Bottom-right
    cv2.line(frame, (x2, y2), (x2 - clen, y2), color_bgr, corner_thick, cv2.LINE_AA)
    cv2.line(frame, (x2, y2), (x2, y2 - clen), color_bgr, corner_thick, cv2.LINE_AA)


def render_unicode_batch(
    frame: np.ndarray,
    badges: Optional[List[Dict[str, Any]]] = None,
    danger_banner: Optional[str] = None,
    zone_labels: Optional[Tuple[str, str, str]] = None,
    hud_left: Optional[str] = None,
    hud_right: Optional[str] = None,
    center_banner: Optional[str] = None,
) -> np.ndarray:
    """
    Renders all UTF-8 Vietnamese labels, pill badges, HUD, and banners in a single
    PIL pass and writes back into `frame` in-place.
    """
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    scale = max(0.75, min(1.35, w / 960.0))
    font_badge = get_unicode_font(max(12, int(14 * scale)), bold=True)
    font_zone = get_unicode_font(max(11, int(12 * scale)), bold=True)
    font_hud = get_unicode_font(max(13, int(15 * scale)), bold=True)
    font_banner = get_unicode_font(max(14, int(16 * scale)), bold=True)
    font_pause = get_unicode_font(max(20, int(26 * scale)), bold=True)

    # 1. Zone labels along bottom
    if zone_labels:
        left_x = int(w * config.ZONE_LEFT_LIMIT)
        right_x = int(w * config.ZONE_RIGHT_LIMIT)
        centers = [left_x // 2, (left_x + right_x) // 2, (right_x + w) // 2]
        for cx, ztxt in zip(centers, zone_labels):
            bbox = draw.textbbox((0, 0), ztxt, font=font_zone)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            tx, ty = cx - tw // 2, h - th - 14
            draw.rounded_rectangle(
                [tx - 8, ty - 3, tx + tw + 8, ty + th + 5],
                radius=6,
                fill=(15, 23, 42, 165),
                outline=(148, 163, 184, 70),
                width=1,
            )
            draw.text((tx, ty), ztxt, font=font_zone, fill=(203, 213, 225, 225))

    # 2. Object detection pill badges
    if badges:
        for item in badges:
            txt = item["text"]
            x1, y1 = int(item["x"]), int(item["y"])
            rgb_col = bgr_to_rgb(item["color_bgr"])
            is_danger = bool(item.get("is_danger", False))

            tb = draw.textbbox((0, 0), txt, font=font_badge)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            pad_x, pad_y = 8, 4
            bx1 = max(2, min(w - tw - pad_x * 2 - 2, x1))
            by2 = y1 - 3 if y1 - th - pad_y * 2 - 4 >= 2 else min(h - 4, y1 + th + pad_y * 2 + 4)
            by1 = by2 - th - pad_y * 2

            bg_fill = (220, 38, 38, 235) if is_danger else (15, 23, 42, 220)
            border_col = (254, 202, 202, 255) if is_danger else (*rgb_col, 235)
            text_col = (255, 255, 255, 255) if is_danger else (*rgb_col, 255)
            # Ensure bright legible text on dark pill
            if not is_danger:
                lum = 0.299 * rgb_col[0] + 0.587 * rgb_col[1] + 0.114 * rgb_col[2]
                if lum < 120:
                    text_col = (248, 250, 252, 255)

            draw.rounded_rectangle(
                [bx1, by1, bx1 + tw + pad_x * 2, by2],
                radius=6,
                fill=bg_fill,
                outline=border_col,
                width=1,
            )
            draw.text((bx1 + pad_x, by1 + pad_y - tb[1]), txt, font=font_badge, fill=text_col)

    # 3. Top HUD bar (Desktop App)
    if hud_left or hud_right:
        hud_h = max(38, int(42 * scale))
        draw.rectangle([0, 0, w, hud_h], fill=(15, 23, 42, 205))
        draw.line([(0, hud_h), (w, hud_h)], fill=(59, 130, 246, 120), width=1)
        if hud_left:
            tb = draw.textbbox((0, 0), hud_left, font=font_hud)
            th = tb[3] - tb[1]
            draw.text((14, (hud_h - th) // 2 - tb[1]), hud_left, font=font_hud, fill=(241, 245, 249, 255))
        if hud_right:
            tb = draw.textbbox((0, 0), hud_right, font=font_hud)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            draw.rounded_rectangle(
                [w - tw - 26, 6, w - 10, hud_h - 6],
                radius=6,
                fill=(239, 68, 68, 220),
            )
            draw.text(
                (w - tw - 18, (hud_h - th) // 2 - tb[1]),
                hud_right,
                font=font_hud,
                fill=(255, 255, 255, 255),
            )

    # 4. Danger alert bottom banner
    if danger_banner:
        tb = draw.textbbox((0, 0), danger_banner, font=font_banner)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        pad_x, pad_y = 16, 8
        bx1 = max(8, (w - tw) // 2 - pad_x)
        bx2 = min(w - 8, (w + tw) // 2 + pad_x)
        by2 = h - 10
        by1 = by2 - th - pad_y * 2
        draw.rounded_rectangle(
            [bx1, by1, bx2, by2],
            radius=8,
            fill=(220, 38, 38, 235),
            outline=(254, 202, 202, 240),
            width=2,
        )
        draw.text(
            ((w - tw) // 2, by1 + pad_y - tb[1]),
            danger_banner,
            font=font_banner,
            fill=(255, 255, 255, 255),
        )

    # 5. Center status / Pause overlay
    if center_banner:
        tb = draw.textbbox((0, 0), center_banner, font=font_pause)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        cx1, cy1 = (w - tw) // 2 - 24, (h - th) // 2 - 14
        cx2, cy2 = (w + tw) // 2 + 24, (h + th) // 2 + 14
        draw.rounded_rectangle(
            [cx1, cy1, cx2, cy2],
            radius=12,
            fill=(15, 23, 42, 230),
            outline=(59, 130, 246, 220),
            width=2,
        )
        draw.text(
            ((w - tw) // 2, cy1 + 14 - tb[1]),
            center_banner,
            font=font_pause,
            fill=(248, 250, 252, 255),
        )

    combined = Image.alpha_composite(pil_img, overlay).convert("RGB")
    frame[:] = cv2.cvtColor(np.asarray(combined), cv2.COLOR_RGB2BGR)
    return frame


class Visualizer:
    """Handles rendering of overlays, privacy masking, and HUD for Desktop OpenCV App."""

    def __init__(self, language: str = "vi", enable_blur: bool = True):
        self.language = language.lower()
        self.enable_blur = enable_blur

    def blur_person_privacy(self, image: np.ndarray, bbox: List[int]) -> np.ndarray:
        """
        Blurs the head/face region of a detected person ('Nguoi' or 'person') for privacy protection.
        Safely clamps coordinates to frame boundaries.
        """
        if not self.enable_blur:
            return image

        img_h, img_w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        h = y2 - y1

        face_h = max(2, int(0.22 * h))
        clamped_x1 = max(0, min(img_w, x1))
        clamped_x2 = max(0, min(img_w, x2))
        clamped_y1 = max(0, min(img_h, y1))
        clamped_y2 = max(0, min(img_h, y1 + face_h))

        if clamped_x2 - clamped_x1 > 1 and clamped_y2 - clamped_y1 > 1:
            head_region = image[clamped_y1:clamped_y2, clamped_x1:clamped_x2]
            if head_region.size > 0:
                image[clamped_y1:clamped_y2, clamped_x1:clamped_x2] = cv2.GaussianBlur(
                    head_region, (25, 25), 0
                )

        return image

    def draw_zone_lines(self, frame: np.ndarray) -> np.ndarray:
        """Draws subtle lane boundary lines to visualize LEFT, FORWARD, and RIGHT zones."""
        h, w = frame.shape[:2]
        left_x = int(w * config.ZONE_LEFT_LIMIT)
        right_x = int(w * config.ZONE_RIGHT_LIMIT)

        overlay = frame.copy()
        cv2.line(overlay, (left_x, 0), (left_x, h), (180, 160, 120), 1, cv2.LINE_AA)
        cv2.line(overlay, (right_x, 0), (right_x, h), (180, 160, 120), 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        labels = (
            ("TRÁI", "PHÍA TRƯỚC", "PHẢI")
            if self.language == "vi"
            else ("LEFT", "AHEAD", "RIGHT")
        )
        render_unicode_batch(frame, zone_labels=labels)
        return frame

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[DetectionResult],
        nearest_obj: Optional[DetectionResult] = None,
    ) -> np.ndarray:
        """Renders corner bounding boxes, privacy blur, and Vietnamese Unicode badges."""
        # 1. Apply privacy blur on all persons ('Nguoi' in best.pt and 'person' in COCO)
        if self.enable_blur:
            for det in detections:
                class_cfg = config.CLASS_CONFIG.get(det.label, {})
                if det.label.lower() in ("person", "nguoi", "người") or class_cfg.get("blur_privacy", False):
                    self.blur_person_privacy(frame, det.bbox)

        # 2. Draw boxes & collect Unicode badges
        badges: List[Dict[str, Any]] = []
        for det in detections:
            is_nearest = nearest_obj is not None and det == nearest_obj
            is_danger = is_nearest and det.distance <= config.DEFAULT_DANGER_DISTANCE
            class_cfg = config.CLASS_CONFIG.get(det.label, {})

            color = config.NEAREST_OBJECT_COLOR if is_danger else class_cfg.get("color", config.DEFAULT_OBJECT_COLOR)
            x1, y1, x2, y2 = det.bbox
            draw_corner_box(frame, x1, y1, x2, y2, color, is_danger=is_danger)

            if self.language == "vi":
                display_label = class_cfg.get("name_vi", det.label)
                pos_text = config.DIRECTIONS.get(det.position, {}).get("vi", det.position)
            else:
                display_label = class_cfg.get("name_en", det.label)
                pos_text = config.DIRECTIONS.get(det.position, {}).get("en", det.position)

            caption = f"{display_label} • {det.distance:.1f}m ({pos_text})"
            badges.append({
                "text": caption,
                "x": x1,
                "y": y1,
                "color_bgr": color,
                "is_danger": is_danger,
            })

        if badges:
            render_unicode_batch(frame, badges=badges)

        return frame

    def draw_hud(
        self,
        frame: np.ndarray,
        fps: float,
        total_objects: int,
        nearest_obj: Optional[DetectionResult] = None,
        is_paused: bool = False,
        volume: float = 1.0,
    ) -> np.ndarray:
        """Renders modern top HUD bar with FPS, object count, volume, and nearest obstacle warning."""
        vol_str = f"Loa: {int(round(volume * 100))}%" if volume > 0.01 else "Loa: TẮT"
        if self.language == "vi":
            info_text = f"FPS: {fps:.1f}  |  Vật cản: {total_objects}  |  {vol_str}"
        else:
            vol_en = f"Vol: {int(round(volume * 100))}%" if volume > 0.01 else "Vol: MUTE"
            info_text = f"FPS: {fps:.1f}  |  Objects: {total_objects}  |  {vol_en}"

        warning_text = None
        if nearest_obj is not None:
            class_cfg = config.CLASS_CONFIG.get(nearest_obj.label, {})
            if self.language == "vi":
                name = class_cfg.get("name_vi", nearest_obj.label)
                pos = config.DIRECTIONS.get(nearest_obj.position, {}).get("vi", nearest_obj.position)
                warning_text = f"CẢNH BÁO: {name} • {nearest_obj.distance:.1f}m [{pos}]"
            else:
                name = class_cfg.get("name_en", nearest_obj.label)
                pos = config.DIRECTIONS.get(nearest_obj.position, {}).get("en", nearest_obj.position)
                warning_text = f"WARNING: {name} @ {nearest_obj.distance:.1f}m [{pos}]"

        pause_text = None
        if is_paused:
            pause_text = (
                "TẠM DỪNG (Nhấn 'P' hoặc Space để tiếp tục)"
                if self.language == "vi"
                else "PAUSED (Press 'P' or Space to Resume)"
            )

        render_unicode_batch(
            frame,
            hud_left=info_text,
            hud_right=warning_text,
            center_banner=pause_text,
        )
        return frame
