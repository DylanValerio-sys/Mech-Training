import cv2


class Camera:
    """Wraps OpenCV VideoCapture for webcam access."""

    def __init__(self, source=0, width=1280, height=720):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            # Try alternative indices if the default fails
            for idx in [0, 1, 2]:
                if idx == source:
                    continue
                self.cap = cv2.VideoCapture(idx)
                if self.cap.isOpened():
                    print(f"Camera found at index {idx}")
                    break
            if not self.cap.isOpened():
                raise RuntimeError(
                    "No camera found. Connect a webcam and try again."
                )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Read actual resolution (camera may not support requested size)
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
