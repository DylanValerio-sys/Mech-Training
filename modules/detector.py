import os
import time
from ultralytics import YOLO

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to fine-tuned engine parts model
CUSTOM_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "engine_parts_best.pt")

# Fallback: YOLO-World zero-shot classes
AUTO_PARTS = [
    "battery", "alternator", "radiator", "engine",
    "belt", "air filter", "oil filter", "exhaust",
    "turbo", "tire", "wheel", "headlight",
    "brake", "bumper", "hood", "wiper",
    "mirror", "hose", "fan", "fuse box",
    "spark plug", "muffler", "coolant tank",
]
AUTO_TOOLS = [
    "wrench", "screwdriver", "pliers", "hammer", "socket",
]

PART_CATEGORIES = {}
for p in AUTO_PARTS:
    PART_CATEGORIES[p] = "part"
for t in AUTO_TOOLS:
    PART_CATEGORIES[t] = "tool"


class Detector:
    """Wraps YOLO model with frame skipping and detection stabilisation.

    Stabilisation reduces flickering by:
    - Smoothing bounding box positions between frames
    - Keeping a detection "alive" for a brief holdover period
    - Requiring a detection to appear in multiple frames before showing
    """

    def __init__(self, model_path="yolov8n.pt", confidence=0.15, mode="standard",
                 infer_size=640, skip_frames=4):
        self.mode = mode
        self.infer_size = infer_size
        self.skip_frames = skip_frames
        self._frame_count = 0
        self._cached_detections = None
        self.using_custom = False

        # Stabilisation: track detections across frames
        self._tracks = {}       # {class_name: TrackedDetection}
        self._holdover = 0.4    # Keep detection visible for 0.4s after last seen
        self._smooth = 0.4      # Box smoothing factor (0=no smooth, 1=max smooth)

        if mode == "automotive":
            if os.path.exists(CUSTOM_MODEL_PATH):
                self.confidence = confidence if confidence < 0.5 else 0.25
                self.model = YOLO(CUSTOM_MODEL_PATH)
                self.using_custom = True
                class_names = list(self.model.names.values())
                print(f"  Fine-tuned engine model loaded")
                print(f"  {len(class_names)} part classes available")
                print(f"  Confidence: {self.confidence:.0%} | Size: {infer_size}px | Skip: {skip_frames}")
                for name in class_names:
                    if name.lower() not in PART_CATEGORIES:
                        PART_CATEGORIES[name.lower()] = "part"
                        PART_CATEGORIES[name] = "part"
            else:
                self.confidence = confidence if confidence < 0.5 else 0.15
                self.model = YOLO("yolov8s-worldv2.pt")
                classes = AUTO_PARTS + AUTO_TOOLS
                self.model.set_classes(classes)
                print(f"  YOLO-World loaded ({len(classes)} classes, zero-shot fallback)")
                print(f"  For better detection, train a custom model first")
        else:
            self.confidence = confidence
            self.model = YOLO(model_path)

    def detect(self, frame):
        """Run detection with frame skipping and stabilisation."""
        self._frame_count += 1

        # Only run model every N frames; reuse cached on others
        if self._cached_detections is not None and self._frame_count % self.skip_frames != 0:
            return self._cached_detections

        results = self.model(frame, conf=self.confidence, imgsz=self.infer_size,
                             verbose=False)

        # Parse raw detections
        raw = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                class_name = self.model.names[cls_id]
                raw.append({
                    "class_name": class_name,
                    "confidence": conf,
                    "box": (int(x1), int(y1), int(x2), int(y2)),
                    "category": PART_CATEGORIES.get(
                        class_name, PART_CATEGORIES.get(class_name.lower(), "object")),
                })

        # Stabilise detections
        detections = self._stabilise(raw)
        self._cached_detections = detections
        return detections

    def _stabilise(self, raw_detections):
        """Smooth detections across frames to reduce flickering.

        - Smooths bounding box positions using exponential moving average
        - Keeps detections visible briefly after they disappear (holdover)
        - Prevents rapid class switching
        """
        now = time.time()

        # Update tracks with current raw detections
        seen_classes = set()
        for det in raw_detections:
            name = det["class_name"]
            seen_classes.add(name)

            if name in self._tracks:
                # Update existing track — smooth the bounding box
                track = self._tracks[name]
                s = self._smooth
                old = track["box"]
                new = det["box"]
                track["box"] = (
                    int(old[0] * s + new[0] * (1 - s)),
                    int(old[1] * s + new[1] * (1 - s)),
                    int(old[2] * s + new[2] * (1 - s)),
                    int(old[3] * s + new[3] * (1 - s)),
                )
                track["confidence"] = det["confidence"]
                track["last_seen"] = now
                track["frames_seen"] += 1
            else:
                # New detection
                self._tracks[name] = {
                    "class_name": name,
                    "confidence": det["confidence"],
                    "box": det["box"],
                    "category": det["category"],
                    "last_seen": now,
                    "frames_seen": 1,
                }

        # Build output: include tracks that are recent enough
        output = []
        expired = []
        for name, track in self._tracks.items():
            age = now - track["last_seen"]
            if age > self._holdover:
                expired.append(name)
                continue

            # Only show detections seen in at least 2 frames (reduces one-off noise)
            if track["frames_seen"] >= 2 or name in seen_classes:
                output.append({
                    "class_name": track["class_name"],
                    "confidence": track["confidence"],
                    "box": track["box"],
                    "category": track["category"],
                })

        # Clean up expired tracks
        for name in expired:
            del self._tracks[name]

        return output
