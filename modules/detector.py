from ultralytics import YOLO

# Focused automotive parts list — shorter, simpler names for better zero-shot detection.
# YOLO-World works best with ~25 visually distinctive single-word or short class names.
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

# Category mapping for color-coding
PART_CATEGORIES = {}
for p in AUTO_PARTS:
    PART_CATEGORIES[p] = "part"
for t in AUTO_TOOLS:
    PART_CATEGORIES[t] = "tool"


class Detector:
    """Wraps YOLO model for object detection inference.

    Supports two modes:
    - 'standard': Pre-trained YOLOv8 with 80 COCO classes
    - 'automotive': YOLO-World open-vocabulary with automotive parts/tools

    Uses frame skipping to maintain smooth video on slow hardware:
    detection runs every N frames, cached results are reused in between.
    """

    def __init__(self, model_path="yolov8n.pt", confidence=0.15, mode="standard",
                 infer_size=640, skip_frames=4):
        self.mode = mode
        self.infer_size = infer_size
        self.skip_frames = skip_frames
        self._frame_count = 0
        self._cached_detections = None

        if mode == "automotive":
            # Use lower confidence for zero-shot detection (default 0.15)
            self.confidence = confidence if confidence < 0.5 else 0.15
            self.model = YOLO("yolov8s-worldv2.pt")
            classes = AUTO_PARTS + AUTO_TOOLS
            self.model.set_classes(classes)
            print(f"  YOLO-World loaded with {len(classes)} automotive classes")
            print(f"  Confidence: {self.confidence:.0%} | Size: {infer_size}px | Skip: {skip_frames}")
        else:
            self.confidence = confidence
            self.model = YOLO(model_path)

    def detect(self, frame):
        """Run detection with frame skipping for performance.

        Only runs the AI model every N frames. Returns cached results
        on skipped frames so the video stays smooth.

        Each detection is a dict with:
            - class_name: str
            - confidence: float (0-1)
            - box: tuple (x1, y1, x2, y2) in pixel coordinates
            - category: str ('part', 'tool', or 'object')
        """
        self._frame_count += 1

        # Only run inference every N frames; reuse cached results on others
        if self._cached_detections is not None and self._frame_count % self.skip_frames != 0:
            return self._cached_detections

        results = self.model(frame, conf=self.confidence, imgsz=self.infer_size,
                             verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                class_name = self.model.names[cls_id]

                detections.append({
                    "class_name": class_name,
                    "confidence": conf,
                    "box": (int(x1), int(y1), int(x2), int(y2)),
                    "category": PART_CATEGORIES.get(class_name, "object"),
                })

        self._cached_detections = detections
        return detections
