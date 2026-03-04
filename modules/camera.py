import cv2


class Camera:
    """Wraps OpenCV VideoCapture for webcam, phone camera, or video file.

    Supports:
    - USB webcam: source=0 (or 1, 2 for other cameras)
    - Phone as webcam via URL: source="http://192.168.1.X:4747/video" (DroidCam)
                                source="http://192.168.1.X:8080/video" (IP Webcam)
    - Video file: source="path/to/video.mp4"
    """

    def __init__(self, source=0, width=1280, height=720):
        self.source = source

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
                        "  2. Use your phone as a webcam with --phone flag\n"
                        "     Install DroidCam on your phone, then run:\n"
                        "     python main.py --auto --phone http://YOUR_PHONE_IP:4747/video"
                    )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera opened: {actual_w}x{actual_h}")

    def get_frame(self):
        """Capture and return a single frame, or None if capture fails."""
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        """Release the camera resource."""
        self.cap.release()
