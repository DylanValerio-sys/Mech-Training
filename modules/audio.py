import pyttsx3
import threading
import queue
import time


class AudioGuidance:
    """Text-to-speech guidance that announces detected parts.

    Runs in a background daemon thread so it never blocks the video loop.
    Uses a cooldown per class name to avoid repeating the same announcement.
    """

    def __init__(self, cooldown=5.0, rate=160):
        self.cooldown = cooldown
        self.rate = rate
        self._last_announced = {}  # {class_name: timestamp}
        self._queue = queue.Queue()
        self._running = True

        # Start TTS in its own thread (pyttsx3 must be created and used
        # within the same thread on Windows)
        self._thread = threading.Thread(target=self._audio_loop, daemon=True)
        self._thread.start()

    def _audio_loop(self):
        """Background thread that processes speech requests."""
        engine = pyttsx3.init()
        engine.setProperty("rate", self.rate)

        while self._running:
            try:
                text = self._queue.get(timeout=0.5)
                engine.say(text)
                engine.runAndWait()
            except queue.Empty:
                continue
            except Exception:
                # If TTS fails, don't crash the thread
                continue

    def announce(self, class_name, extra_info=""):
        """Queue an announcement if this class hasn't been announced recently."""
        now = time.time()

        # Check cooldown
        if class_name in self._last_announced:
            if now - self._last_announced[class_name] < self.cooldown:
                return

        self._last_announced[class_name] = now

        text = f"Detected: {class_name}"
        if extra_info:
            text += f". {extra_info}"

        self._queue.put(text)

    def announce_detections(self, detections):
        """Process a list of detections and announce new ones."""
        for det in detections:
            self.announce(det["class_name"])

    def shutdown(self):
        """Stop the audio thread."""
        self._running = False
