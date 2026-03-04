import cv2
import time


class OverlayRenderer:
    """Draws detection overlays on video frames."""

    # Color palette by detection category (BGR format)
    CATEGORY_COLORS = {
        "part": (0, 100, 255),    # Orange - automotive parts
        "tool": (255, 180, 0),    # Light blue - tools
        "object": (0, 255, 0),    # Green - general objects
    }

    def __init__(self, show_fps=True, show_confidence=True):
        self.show_fps = show_fps
        self.show_confidence = show_confidence
        self.prev_time = time.time()
        self.fps = 0.0
        self.mode_label = ""

    def draw(self, frame, detections):
        """Draw bounding boxes, labels, and pinpoints on the frame.

        Args:
            frame: numpy array (BGR image from OpenCV)
            detections: list of dicts from Detector.detect()

        Returns:
            Annotated frame (same array, modified in-place).
        """
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            class_name = det["class_name"]
            conf = det["confidence"]
            category = det.get("category", "object")
            color = self.CATEGORY_COLORS.get(category, self.CATEGORY_COLORS["object"])

            # Bounding box with rounded-corner look (thicker lines)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Corner accents (short lines at each corner for a tech look)
            corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
            # Top-left
            cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 3)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 3)
            # Top-right
            cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, 3)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, 3)
            # Bottom-left
            cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, 3)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, 3)
            # Bottom-right
            cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 3)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, 3)

            # Center pinpoint marker (crosshair style)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), 4, color, -1)        # Center dot
            cv2.circle(frame, (cx, cy), 10, color, 2)         # Outer ring
            cv2.line(frame, (cx - 15, cy), (cx - 6, cy), color, 1)   # Left tick
            cv2.line(frame, (cx + 6, cy), (cx + 15, cy), color, 1)   # Right tick
            cv2.line(frame, (cx, cy - 15), (cx, cy - 6), color, 1)   # Top tick
            cv2.line(frame, (cx, cy + 6), (cx, cy + 15), color, 1)   # Bottom tick

            # Label text
            if self.show_confidence:
                label = f"{class_name} {conf:.0%}"
            else:
                label = class_name

            # Background rectangle for text readability
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                frame,
                (x1, y1 - text_h - baseline - 6),
                (x1 + text_w + 8, y1),
                color,
                -1,
            )
            cv2.putText(
                frame, label, (x1 + 4, y1 - baseline - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
            )

            # Category tag (small text below the label)
            if category != "object":
                tag = category.upper()
                (tw, th), _ = cv2.getTextSize(
                    tag, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1
                )
                cv2.rectangle(
                    frame, (x1, y2), (x1 + tw + 6, y2 + th + 6), color, -1
                )
                cv2.putText(
                    frame, tag, (x1 + 3, y2 + th + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1
                )

        # --- HUD elements ---
        h, w = frame.shape[:2]

        # FPS counter
        if self.show_fps:
            current_time = time.time()
            dt = current_time - self.prev_time
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
            self.prev_time = current_time

            fps_text = f"FPS: {self.fps:.1f}"
            cv2.putText(
                frame, fps_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
            )

        # Detection count
        count_text = f"Detections: {len(detections)}"
        cv2.putText(
            frame, count_text, (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
        )

        # Mode label (top-right)
        if self.mode_label:
            (mw, mh), _ = cv2.getTextSize(
                self.mode_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            cv2.putText(
                frame, self.mode_label, (w - mw - 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )

        # Legend (bottom-left)
        legend_y = h - 60
        for cat, col in self.CATEGORY_COLORS.items():
            cv2.circle(frame, (15, legend_y), 6, col, -1)
            cv2.putText(
                frame, cat.capitalize(), (28, legend_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1
            )
            legend_y += 20

        return frame
