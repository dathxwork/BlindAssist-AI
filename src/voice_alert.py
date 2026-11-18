"""
Flexible Voice Alert adapter built on top of the unified VoiceAssistant engine.
Supports dictionary payloads (from ObstacleDetector) as well as positional arguments.
"""

import time
from src.voice_assistant import VoiceAssistant as _BaseVoiceAssistant


class VoiceAssistant(_BaseVoiceAssistant):
    """
    Backward-compatible multi-threaded Vietnamese TTS helper supporting:
      - alert(most_dangerous_dict)
      - alert(label, direction, distance)
      - speak(text)
    """

    def __init__(self, cooldown: float = 2.5, speech_rate: float = 1.0, volume: float = 1.0):
        super().__init__(
            language="vi",
            volume=volume,
            rate_multiplier=speech_rate,
            enable_sonar=False,
        )
        self.cooldown = cooldown
        self.last_spoken = {}

    def alert(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], dict):
            item = args[0]
            label = str(item.get("label", item.get("Vật thể", "Vật cản")))
            direction = str(item.get("direction", item.get("Hướng", "phía trước")))
            distance = float(item.get("distance", item.get("Khoảng cách (m)", 0.0)))
            custom_msg = item.get("alert_message")
        elif len(args) >= 3:
            label = str(args[0])
            direction = str(args[1])
            distance = float(args[2])
            custom_msg = None
        elif "label" in kwargs:
            label = str(kwargs.get("label", "Vật cản"))
            direction = str(kwargs.get("direction", "phía trước"))
            distance = float(kwargs.get("distance", 0.0))
            custom_msg = kwargs.get("alert_message")
        else:
            return

        now = time.time()
        key = f"{label}_{direction}"
        if key in self.last_spoken and (now - self.last_spoken[key]) < self.cooldown:
            return

        self.last_spoken[key] = now
        if custom_msg:
            message = custom_msg
        else:
            rounded_dist = round(distance * 2) / 2
            dist_str = str(int(rounded_dist)) if rounded_dist.is_integer() else str(rounded_dist).replace(".", " phẩy ")
            dir_speech = "phía trước" if direction == "chính giữa" else direction
            message = f"Cảnh báo, có {label.lower()} ở {dir_speech}, cách {dist_str} mét."

        self.speak(message)
