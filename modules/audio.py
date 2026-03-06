"""Audio guidance module for Mech Training AI.

Provides text-to-speech announcements for detected parts and service steps.
Runs in a background daemon thread so it never blocks the video loop.
Uses a cooldown per class name to avoid repeating the same announcement.

The TTS engine (pyttsx3) must be created and used within the same thread
on Windows — this is a critical platform constraint.
"""

import threading
import queue
import time


class AudioGuidance:
    """Text-to-speech guidance running in a background thread."""

    def __init__(self, cooldown=5.0, rate=150):
        self.cooldown = cooldown
        self.rate = rate
        self._last_announced = {}  # {class_name: timestamp}
        self._queue = queue.Queue()
        self._running = True
        self._ready = False
        self._error_msg = None

        # Start TTS in its own thread (pyttsx3 must be created and used
        # within the same thread on Windows)
        self._thread = threading.Thread(target=self._audio_loop, daemon=True)
        self._thread.start()

        # Wait for engine to initialize (up to 3 seconds)
        for _ in range(30):
            if self._ready or self._error_msg:
                break
            time.sleep(0.1)

        if self._ready:
            print("  Audio engine : READY")
            # Play startup confirmation
            self._queue.put("Audio ready.")
        else:
            error = self._error_msg or "Unknown error"
            print(f"  Audio engine : FAILED ({error})")
            print("  The program will run without audio guidance.")
            print("  To fix: pip install pyttsx3 (in your venv)")

    def _audio_loop(self):
        """Background thread that processes speech requests."""
        try:
            import pyttsx3
        except ImportError:
            self._error_msg = "pyttsx3 not installed"
            return

        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)

            # Try to set a clear voice
            try:
                voices = engine.getProperty("voices")
                if voices and len(voices) > 0:
                    engine.setProperty("voice", voices[0].id)
            except Exception:
                pass  # Use default voice

            self._ready = True

            while self._running:
                try:
                    text = self._queue.get(timeout=0.5)
                    if text:
                        engine.say(text)
                        engine.runAndWait()
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"  Audio playback error: {e}")
                    continue

        except Exception as e:
            self._error_msg = str(e)
            self._ready = False

    def announce(self, class_name):
        """Queue an announcement if this class hasn't been announced recently."""
        if not self._ready:
            return

        now = time.time()

        # Check cooldown
        if class_name in self._last_announced:
            if now - self._last_announced[class_name] < self.cooldown:
                return

        self._last_announced[class_name] = now

        # Clean up class name for speech
        spoken_name = class_name.replace("_", " ").title()
        self._queue.put(spoken_name)

    def announce_detections(self, detections):
        """Announce the top detection (highest confidence only)."""
        if not detections:
            return
        best = max(detections, key=lambda d: d["confidence"])
        self.announce(best["class_name"])

    def announce_step(self, text):
        """Announce a service procedure step or part info (bypasses cooldown)."""
        if self._ready and text:
            # Clear any pending announcements to avoid queue backup
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._queue.put(text)

    def shutdown(self):
        """Stop the audio thread."""
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
