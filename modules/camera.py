import cv2


class Camera:
    """Wraps OpenCV VideoCapture for webcam, phone camera, or video file.

    Supports:
    - USB webcam: source=0 (or 1, 2 for other cameras)
    - Phone as webcam via DroidCam: source=2 (virtual webcam driver)
    - Video file: source="path/to/video.mp4"

    Auto-rotates portrait frames (height > width) to landscape for
    phone cameras like DroidCam held in portrait orientation.
    """

    def __init__(self, source=0, width=1280, height=720, auto_rotate=True):
        self.source = source
        self.auto_rotate = auto_rotate

        if isinstance(source, str) and (source.startswith("http") or source.endswith((".mp4", ".avi", ".mkv"))):
            # URL stream (phone cam) or video file
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                raise RuntimeError(
                    f"Could not connect to: {source}\n"
                    "For phone webcam, make sure:\n"
                    "  1. DroidCam/IP Webcam app is running on your phone\n"
                    "  2. Phone and PC are on the same Wi-Fi network\n"
                    "  3. The URL is correct (check the app for the IP address)"
                )
            source_type = "URL stream" if source.startswith("http") else "video file"
            print(f"Connected to {source_type}: {source}")
        else:
            # USB webcam with index
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                # Try alternative indices
                for idx in [0, 1, 2]:
                    if idx == source:
                        continue
                    self.cap = cv2.VideoCapture(idx)
                    if self.cap.isOpened():
                        print(f"Camera found at index {idx}")
                        break
                if not self.cap.isOpened():
                    raise RuntimeError(
                        "No camera found. Options:\n"
                        "  1. Connect a USB webcam\n"
                        "  2. Use your phone as a webcam with DroidCam:\n"
                        "     python main.py --auto --cam 2"
                    )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera opened: {actual_w}x{actual_h}")

        if auto_rotate and actual_h > actual_w:
            print("  Portrait orientation detected - auto-rotate ON")

    def get_frame(self):
        """Capture and return a single frame, or None if capture fails.

        Auto-rotates portrait frames (height > width) to landscape.
        """
        ret, frame = self.cap.read()
        if not ret:
            return None

        if self.auto_rotate and frame.shape[0] > frame.shape[1]:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        return frame

    def release(self):
        """Release the camera resource."""
        self.cap.release()
