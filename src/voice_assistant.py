"""
Voice Assistant module for multi-threaded Text-to-Speech (TTS) & Sonar proximity audio
feedback on Desktop (main.py), supporting both Vietnamese (gTTS cached + SAPI5 vi-VN)
and English (SAPI5), with laptop-style volume, mute, and speech rate controls.
"""

import hashlib
import os
import queue
import subprocess
import time
from threading import Thread, Event
from typing import Optional, Tuple

try:
    import winsound
except ImportError:
    winsound = None

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

try:
    from gtts import gTTS
except ImportError:
    gTTS = None

import config


class VoiceAssistant:
    """Manages non-blocking TTS & proximity beeps running on a background daemon thread."""

    SPEECH_RATES = [0.8, 1.0, 1.25, 1.5, 2.0]

    def __init__(
        self,
        language: str = "vi",
        speech_rate: int = 200,
        volume: float = 1.0,
        rate_multiplier: float = 1.0,
        enable_sonar: bool = True,
    ):
        self.language = language.lower()
        self.base_speech_rate = speech_rate
        self.rate_multiplier = max(0.5, min(2.0, float(rate_multiplier)))
        self.volume = max(0.0, min(1.0, float(volume)))
        self.prev_volume = self.volume if self.volume > 0.01 else 1.0
        self.enable_sonar = enable_sonar

        self._queue: queue.Queue = queue.Queue(maxsize=5)
        self._stop_event = Event()
        self._last_alert_time: float = 0.0
        self._last_alert_key: Optional[Tuple[str, str, float]] = None
        self._last_spoken_text: str = ""
        self._last_sonar_time: float = 0.0

        self._cache_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "audio_cache",
        )
        os.makedirs(self._cache_dir, exist_ok=True)

        self._thread = Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def set_volume(self, val: float):
        """Sets speaker volume from 0.0 to 1.0 like a laptop volume control."""
        self.volume = max(0.0, min(1.0, float(val)))

    def change_volume(self, delta: float):
        """Increases or decreases volume relatively (e.g. +0.1 or -0.1)."""
        self.set_volume(self.volume + delta)

    def toggle_mute(self) -> bool:
        """Toggles between mute (0.0) and previous volume level."""
        if self.volume > 0.01:
            self.prev_volume = self.volume
            self.volume = 0.0
            return False  # Muted
        self.volume = self.prev_volume if self.prev_volume > 0.05 else 1.0
        return True       # Unmuted

    def cycle_speech_rate(self) -> float:
        """Cycles speech rate multiplier (0.8x -> 1.0x -> 1.25x -> 1.5x -> 2.0x)."""
        nxt = 1.0
        for r in self.SPEECH_RATES:
            if r > self.rate_multiplier + 0.04:
                nxt = r
                break
        else:
            nxt = self.SPEECH_RATES[0]
        self.rate_multiplier = nxt
        return self.rate_multiplier

    def repeat_last(self):
        """Re-speaks the most recent alert message immediately."""
        if self._last_spoken_text and self.volume > 0.01:
            self.speak(self._last_spoken_text)

    def speak(self, text: str):
        """Pushes an arbitrary message to the speech queue."""
        if not text or self.volume <= 0.01:
            return
        self._last_spoken_text = text
        try:
            while not self._queue.empty():
                self._queue.get_nowait()
        except Exception:
            pass
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            pass

    def _speak_gtts_windows(self, message: str) -> bool:
        """Plays natural Vietnamese speech via cached gTTS + Windows Media Player COM."""
        if gTTS is None or os.name != "nt":
            return False

        key = hashlib.md5(message.strip().lower().encode("utf-8")).hexdigest()
        mp3_path = os.path.join(self._cache_dir, f"{key}.mp3")

        try:
            if not os.path.exists(mp3_path):
                tts = gTTS(text=message, lang="vi", slow=False)
                tts.save(mp3_path)

            vol_int = max(0, min(100, int(round(self.volume * 100))))
            if vol_int == 0:
                return True

            safe_path = mp3_path.replace("'", "''")
            ps_cmd = (
                f"$wmp = New-Object -ComObject WMPlayer.OCX; "
                f"$wmp.settings.volume = {vol_int}; "
                f"$wmp.settings.rate = {self.rate_multiplier:.2f}; "
                f"$wmp.URL = '{safe_path}'; "
                f"$wmp.controls.play(); "
                f"$c = 0; "
                f"while ($wmp.playState -ne 1 -and $c -lt 45) {{ Start-Sleep -Milliseconds 90; $c++ }}; "
                f"$wmp.close();"
            )
            creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            subprocess.run(
                ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", ps_cmd],
                capture_output=True,
                timeout=6,
                creationflags=creation_flags,
            )
            return True
        except Exception as e:
            print(f"[VoiceAssistant] gTTS fallback warning: {e}")
            return False

    def _worker_loop(self):
        """Background daemon worker initializing SAPI5 / gTTS and speaking queued items."""
        engine = None
        has_native_vn_voice = False

        if pyttsx3 is not None:
            try:
                engine = pyttsx3.init()
                voices = engine.getProperty("voices") or []
                for v in voices:
                    v_name = getattr(v, "name", "").lower()
                    v_langs = str(getattr(v, "languages", [])).lower()
                    if self.language == "vi":
                        # Strict check: NEVER match 'an' inside 'Susan', 'Brian', 'Alan'
                        if (
                            "vietnamese" in v_name
                            or "tiếng việt" in v_name
                            or "microsoft an" in v_name
                            or "vi-vn" in v_langs
                            or "vi_vn" in v_langs
                        ):
                            engine.setProperty("voice", v.id)
                            has_native_vn_voice = True
                            break
                    else:
                        if "english" in v_name or "en-us" in v_langs or "en_us" in v_langs or "zira" in v_name or "david" in v_name:
                            engine.setProperty("voice", v.id)
                            break
            except Exception as e:
                print(f"[VoiceAssistant] SAPI5 init note: {e}")
                engine = None

        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.2)
                if item is None:
                    break

                text = str(item).strip()
                if not text or self.volume <= 0.01:
                    continue

                spoken = False
                # For Vietnamese without native SAPI5 vi-VN voice, prefer natural gTTS + cache
                if self.language == "vi" and not has_native_vn_voice:
                    spoken = self._speak_gtts_windows(text)

                if not spoken and engine is not None and self.volume > 0.01:
                    try:
                        engine.setProperty("volume", self.volume)
                        engine.setProperty("rate", int(self.base_speech_rate * self.rate_multiplier))
                        engine.say(text)
                        engine.runAndWait()
                    except Exception as e:
                        print(f"[VoiceAssistant] TTS speech error: {e}")

                with self._queue.mutex:
                    self._queue.queue.clear()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[VoiceAssistant] Worker error: {e}")
                time.sleep(0.1)

    def maybe_sonar_beep(self, distance: float):
        """Emits a short proximity radar beep on Windows if obstacle is very close."""
        if not self.enable_sonar or self.volume <= 0.01 or winsound is None:
            return
        if distance > 2.5:
            return
        now = time.time()
        interval = 0.28 if distance <= 1.5 else 0.65
        freq = 880 if distance <= 1.5 else 660
        if now - self._last_sonar_time >= interval:
            self._last_sonar_time = now
            Thread(target=lambda: winsound.Beep(freq, 55), daemon=True).start()

    def alert(self, label: str, distance: float, position: str):
        """Queues a voice alert if cooldown & significance thresholds are met."""
        self.maybe_sonar_beep(distance)

        now = time.time()
        rounded_dist = round(distance * 2) / 2.0
        rounded_dist_str = (
            str(int(rounded_dist)) if rounded_dist.is_integer() else f"{rounded_dist:.1f}"
        )

        time_elapsed = now - self._last_alert_time
        current_key = (label, position, rounded_dist)

        if time_elapsed < config.ALERT_COOLDOWN_SECONDS:
            if self._last_alert_key is not None:
                last_label, last_pos, last_dist = self._last_alert_key
                if (
                    last_label == label
                    and last_pos == position
                    and abs(last_dist - rounded_dist) < config.SIGNIFICANT_DIST_CHANGE
                ):
                    return

        class_info = config.CLASS_CONFIG.get(label, {})
        if self.language == "vi":
            name_vi = class_info.get("name_vi", label)
            pos_vi = config.DIRECTIONS.get(position, {}).get("vi_lower", position)
            if pos_vi == "chính giữa":
                pos_vi = "phía trước"
            dist_vi = rounded_dist_str.replace(".", " phẩy ")
            prefix = "Cảnh báo nguy hiểm, có " if distance <= 1.5 else "Có "
            text = f"{prefix}{name_vi.lower()} ở {pos_vi}, cách {dist_vi} mét."
        else:
            name_en = class_info.get("name_en", label)
            pos_en = config.DIRECTIONS.get(position, {}).get("en", position)
            prefix = "Danger, " if distance <= 1.5 else ""
            text = f"{prefix}{name_en} {rounded_dist_str} meters {pos_en}."

        self._last_alert_time = now
        self._last_alert_key = current_key
        self.speak(text)

    def stop(self):
        """Signals worker thread to stop cleanly."""
        self._stop_event.set()
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        self._thread.join(timeout=1.0)
