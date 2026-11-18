"""
BlindAssist: Real-Time YOLO Obstacle Detection & Audio Assistance System (Desktop CLI).
Main entry point for webcam/video processing, optical distance estimation, and voice alerts.
"""

import argparse
import os
import sys
import time
import cv2

import config
from src.detector import ObjectDetector
from src.voice_assistant import VoiceAssistant
from src.visualizer import Visualizer


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BlindAssist - YOLO Obstacle Detection & Spatial Audio Warning System"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="test_video.mp4",
        help="Input video source: webcam index ('0', '1') or path to video file (default: test_video.mp4)",
    )
    default_model = (
        "models/best.pt"
        if os.path.exists("models/best.pt")
        else ("best.pt" if os.path.exists("best.pt") else "models/gpModel.pt")
    )
    parser.add_argument(
        "--model",
        type=str,
        default=default_model,
        help=f"Path to YOLO weights file (default: {default_model})",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.30,
        help="YOLO detection confidence threshold (default: 0.30)",
    )
    parser.add_argument(
        "--lang",
        type=str,
        choices=["vi", "en"],
        default="vi",
        help="Audio alert language: 'vi' (Vietnamese) or 'en' (English) (default: vi)",
    )
    parser.add_argument(
        "--alert-dist",
        type=float,
        default=config.DEFAULT_WARNING_DISTANCE,
        help=f"Distance threshold in meters to trigger voice alerts (default: {config.DEFAULT_WARNING_DISTANCE})",
    )
    parser.add_argument(
        "--volume",
        type=float,
        default=1.0,
        help="Speaker audio volume from 0.0 to 1.0 (like laptop volume, default: 1.0)",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable text-to-speech audio feedback",
    )
    parser.add_argument(
        "--no-blur",
        action="store_true",
        help="Disable automatic privacy face blurring",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Optional path to save the processed output video (e.g., results/ket_qua.mp4)",
    )
    parser.add_argument(
        "--no-view",
        action="store_true",
        help="Do not display the visual window (useful for headless servers or automated batch runs)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional maximum number of frames to process (0 = unlimited, useful for testing)",
    )
    return parser.parse_args()


def open_video_source(source_str: str) -> cv2.VideoCapture:
    if source_str.isdigit():
        cap = cv2.VideoCapture(int(source_str))
    else:
        candidates = [source_str, os.path.join("data", source_str)]
        resolved = next((p for p in candidates if os.path.exists(p)), None)
        if not resolved:
            print(f"[Error] Source file not found: {source_str}")
            sys.exit(1)
        cap = cv2.VideoCapture(resolved)

    if not cap.isOpened():
        print(f"[Error] Could not open video source: {source_str}")
        sys.exit(1)

    return cap


def main():
    args = parse_arguments()

    print("==========================================================")
    print("  BlindAssist AI - Obstacle Detection & Spatial Audio     ")
    print("==========================================================")
    print(f"Source:         {args.source}")
    print(f"Model:          {args.model}")
    print(f"Language:       {'Tiếng Việt' if args.lang == 'vi' else 'English'}")
    print(f"Warning Limit:  {args.alert_dist}m")
    print(f"Audio Feedback: {'Disabled' if args.no_audio else 'Enabled'}")
    print(f"Privacy Blur:   {'Disabled' if args.no_blur else 'Enabled'}")
    print("Shortcuts:      [Space/P] Pause | [M] Mute | []/[] Vol | [S] Speed | [R] Repeat | [B] Blur | [Q/ESC] Quit")
    print("==========================================================")

    model_path = args.model
    if not os.path.exists(model_path):
        alt_path = os.path.join("models", os.path.basename(model_path))
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            print(f"[Error] Model weights file not found: {args.model}")
            sys.exit(1)

    detector = ObjectDetector(model_path=model_path)
    visualizer = Visualizer(language=args.lang, enable_blur=not args.no_blur)

    voice_assistant = None
    if not args.no_audio:
        voice_assistant = VoiceAssistant(language=args.lang, volume=args.volume)

    cap = open_video_source(args.source)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    if fps_in <= 0 or fps_in != fps_in:
        fps_in = 30.0

    video_writer = None
    if args.save:
        save_dir = os.path.dirname(args.save)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(args.save, fourcc, fps_in, (frame_w, frame_h))
        print(f"[Info] Saving output video to: {args.save}")

    is_paused = False
    prev_time = time.time()
    current_fps = 0.0
    last_raw_frame = None
    last_detections = []
    last_nearest = None
    processed_count = 0

    try:
        while cap.isOpened():
            if not is_paused:
                ret, frame = cap.read()
                if not ret:
                    print("[Info] End of video stream.")
                    break

                now = time.time()
                dt = now - prev_time
                prev_time = now
                if dt > 0:
                    current_fps = (
                        0.9 * current_fps + 0.1 * (1.0 / dt)
                        if current_fps > 0
                        else 1.0 / dt
                    )

                last_raw_frame = frame.copy()
                detections, nearest_obj = detector.detect(frame, conf_threshold=args.conf)
                last_detections = detections
                last_nearest = nearest_obj

                if nearest_obj and nearest_obj.distance <= args.alert_dist and voice_assistant:
                    voice_assistant.alert(
                        nearest_obj.label,
                        nearest_obj.distance,
                        nearest_obj.position,
                    )

                current_vol = voice_assistant.volume if voice_assistant else 0.0
                visualizer.draw_zone_lines(frame)
                visualizer.draw_detections(frame, detections, nearest_obj)
                visualizer.draw_hud(
                    frame,
                    current_fps,
                    len(detections),
                    nearest_obj,
                    is_paused=False,
                    volume=current_vol,
                )

                processed_count += 1
                if video_writer is not None:
                    video_writer.write(frame)

                if args.max_frames > 0 and processed_count >= args.max_frames:
                    print(f"[Info] Reached max-frames limit ({args.max_frames}).")
                    break
            else:
                if last_raw_frame is not None:
                    frame = last_raw_frame.copy()
                    current_vol = voice_assistant.volume if voice_assistant else 0.0
                    visualizer.draw_zone_lines(frame)
                    visualizer.draw_detections(frame, last_detections, last_nearest)
                    visualizer.draw_hud(
                        frame,
                        current_fps,
                        len(last_detections),
                        last_nearest,
                        is_paused=True,
                        volume=current_vol,
                    )
                time.sleep(0.03)

            if not args.no_view:
                cv2.imshow("BlindAssist AI - Visual & Spatial Audio Guidance", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in [ord("q"), ord("Q"), 27]:
                    print("[Info] User requested exit.")
                    break
                elif key in [ord("p"), ord("P"), ord(" ")]:
                    is_paused = not is_paused
                elif key in [ord("+"), ord("="), ord("]")]:
                    if voice_assistant:
                        voice_assistant.change_volume(0.1)
                        print(f"[Audio] Volume: {int(round(voice_assistant.volume * 100))}%")
                elif key in [ord("-"), ord("_"), ord("[")]:
                    if voice_assistant:
                        voice_assistant.change_volume(-0.1)
                        print(f"[Audio] Volume: {int(round(voice_assistant.volume * 100))}%")
                elif key in [ord("m"), ord("M")]:
                    if voice_assistant:
                        is_unmuted = voice_assistant.toggle_mute()
                        state_str = (
                            f"{int(round(voice_assistant.volume * 100))}%"
                            if is_unmuted
                            else "MUTE"
                        )
                        print(f"[Audio] Volume: {state_str}")
                elif key in [ord("r"), ord("R")]:
                    if voice_assistant:
                        voice_assistant.repeat_last()
                        print("[Audio] Repeating last alert.")
                elif key in [ord("s"), ord("S")]:
                    if voice_assistant:
                        new_rate = voice_assistant.cycle_speech_rate()
                        print(f"[Audio] Speech speed: {new_rate:.2f}x")
                elif key in [ord("b"), ord("B")]:
                    visualizer.enable_blur = not visualizer.enable_blur
                    print(f"[Privacy] Face blur: {'ON' if visualizer.enable_blur else 'OFF'}")

    except KeyboardInterrupt:
        print("\n[Info] Interrupted by user.")
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        if not args.no_view:
            cv2.destroyAllWindows()
        if voice_assistant is not None:
            voice_assistant.stop()
        print(f"[Info] BlindAssist shut down cleanly ({processed_count} frames processed).")


if __name__ == "__main__":
    main()
