import cv2
import time
import textwrap


class OverlayRenderer:
    """Draws detection overlays and service procedure HUD on video frames."""

    # Color palette by detection category (BGR format)
    CATEGORY_COLORS = {
        "part": (0, 100, 255),    # Orange - automotive parts
        "tool": (255, 180, 0),    # Light blue - tools
        "object": (0, 255, 0),    # Green - general objects
    }

    # Service mode colors (BGR)
    SERVICE_BG = (40, 40, 40)         # Dark panel background
    SERVICE_ACCENT = (0, 200, 255)    # Orange-yellow accent
    SERVICE_TEXT = (230, 230, 230)    # Light text
    SERVICE_DIM = (140, 140, 140)    # Dimmed text
    SERVICE_GREEN = (0, 200, 0)      # Completed/go
    SERVICE_TOOL_COLOR = (255, 180, 0)  # Tool highlight (light blue)
    SERVICE_TORQUE_COLOR = (0, 140, 255)  # Torque highlight (orange)

    def __init__(self, show_fps=True, show_confidence=True):
        self.show_fps = show_fps
        self.show_confidence = show_confidence
        self.prev_time = time.time()
        self.fps = 0.0
        self.mode_label = ""

    def draw(self, frame, detections):
        """Draw bounding boxes, labels, and pinpoints on the frame."""
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            class_name = det["class_name"]
            conf = det["confidence"]
            category = det.get("category", "object")
            color = self.CATEGORY_COLORS.get(category, self.CATEGORY_COLORS["object"])

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Corner accents
            corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
            cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 3)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 3)
            cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, 3)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, 3)
            cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, 3)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, 3)
            cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 3)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, 3)

            # Center pinpoint marker
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), 4, color, -1)
            cv2.circle(frame, (cx, cy), 10, color, 2)
            cv2.line(frame, (cx - 15, cy), (cx - 6, cy), color, 1)
            cv2.line(frame, (cx + 6, cy), (cx + 15, cy), color, 1)
            cv2.line(frame, (cx, cy - 15), (cx, cy - 6), color, 1)
            cv2.line(frame, (cx, cy + 6), (cx, cy + 15), color, 1)

            # Label text
            if self.show_confidence:
                label = f"{class_name} {conf:.0%}"
            else:
                label = class_name

            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                frame, (x1, y1 - text_h - baseline - 6),
                (x1 + text_w + 8, y1), color, -1,
            )
            cv2.putText(
                frame, label, (x1 + 4, y1 - baseline - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
            )

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
        self._draw_hud(frame, detections)
        return frame

    def draw_service(self, frame, procedure, detections=None):
        """Draw service procedure overlay on the frame.

        Shows the current step info panel, progress bar, and section
        indicator. Also draws any detections if present.

        Args:
            frame: BGR image from OpenCV
            procedure: ProcedureGuide instance
            detections: optional list of detection dicts
        """
        if detections:
            for det in detections:
                self._draw_detection_box(frame, det)

        h, w = frame.shape[:2]
        step = procedure.current_step

        # --- Top bar: progress ---
        self._draw_progress_bar(frame, w, procedure)

        # --- Bottom panel: current step info ---
        self._draw_step_panel(frame, w, h, step, procedure)

        # --- FPS counter (top-left, small) ---
        self._draw_fps(frame)

        return frame

    def _draw_detection_box(self, frame, det):
        """Draw a single detection bounding box (simplified for service mode)."""
        x1, y1, x2, y2 = det["box"]
        class_name = det["class_name"]
        conf = det["confidence"]
        category = det.get("category", "object")
        color = self.CATEGORY_COLORS.get(category, self.CATEGORY_COLORS["object"])

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Corner accents
        corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 3)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 3)
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, 3)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, 3)
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, 3)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, 3)
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 3)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, 3)

        label = f"{class_name} {conf:.0%}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            frame, (x1, y1 - text_h - 6), (x1 + text_w + 6, y1), color, -1
        )
        cv2.putText(
            frame, label, (x1 + 3, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
        )

    def _draw_progress_bar(self, frame, w, procedure):
        """Draw progress bar and section name at the top of the frame."""
        bar_h = 40
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), self.SERVICE_BG, -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        # Progress bar
        bar_x = 10
        bar_y = 28
        bar_w = w - 20
        bar_fill = int(bar_w * procedure.progress_fraction)

        # Background track
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 6),
                       (80, 80, 80), -1)
        # Filled portion
        if bar_fill > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_fill, bar_y + 6),
                           self.SERVICE_ACCENT, -1)

        # Section label and step counter (top of bar)
        step = procedure.current_step
        section_text = f"SEC {step['section_id']}: {step['section_name'].upper()}"
        cv2.putText(frame, section_text, (12, 20),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.SERVICE_ACCENT, 1)

        # Step counter (right side)
        counter = f"{procedure.current_index + 1}/{procedure.total_steps}"
        (cw, _), _ = cv2.getTextSize(counter, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(frame, counter, (w - cw - 12, 20),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.SERVICE_TEXT, 1)

    def _draw_step_panel(self, frame, w, h, step, procedure):
        """Draw the current step info panel at the bottom of the frame."""
        panel_h = 160
        panel_y = h - panel_h

        # Semi-transparent panel background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, panel_y), (w, h), self.SERVICE_BG, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Top border accent line
        cv2.line(frame, (0, panel_y), (w, panel_y), self.SERVICE_ACCENT, 2)

        # Check number badge
        check_text = f"#{step['check']}"
        badge_x = 15
        badge_y = panel_y + 12
        (bw, bh), _ = cv2.getTextSize(check_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (badge_x - 4, badge_y - 2),
                       (badge_x + bw + 8, badge_y + bh + 6), self.SERVICE_ACCENT, -1)
        cv2.putText(frame, check_text, (badge_x + 2, badge_y + bh + 1),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # Step title
        title_x = badge_x + bw + 20
        cv2.putText(frame, step["title"].upper(), (title_x, panel_y + 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.SERVICE_TEXT, 2)

        # Description (wrapped)
        desc_y = panel_y + 55
        wrapped = textwrap.wrap(step["description"], width=80)
        for line in wrapped[:2]:
            cv2.putText(frame, line, (15, desc_y),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.SERVICE_DIM, 1)
            desc_y += 18

        # Detail text (work outline, wrapped)
        detail = step.get("detail", "")
        if detail:
            detail_y = desc_y + 5
            detail_wrapped = textwrap.wrap(detail, width=90)
            for line in detail_wrapped[:2]:
                cv2.putText(frame, line, (15, detail_y),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.SERVICE_TEXT, 1)
                detail_y += 16

        # Tools (bottom-left)
        info_y = h - 20
        if step["tools"]:
            tools_text = f"TOOLS: {step['tools']}"
            cv2.putText(frame, tools_text, (15, info_y),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.SERVICE_TOOL_COLOR, 1)

        # Torque (bottom-right of panel, if applicable)
        if step["torque"]:
            torque_text = f"TORQUE: {step['torque']}"
            (tw, _), _ = cv2.getTextSize(torque_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.putText(frame, torque_text, (w - tw - 15, info_y),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.SERVICE_TORQUE_COLOR, 1)

        # Navigation hints (bottom-center)
        nav_text = "SPACE: Next | BACKSPACE: Prev | 1-7: Section | Q: Quit"
        (nw, _), _ = cv2.getTextSize(nav_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        nav_x = (w - nw) // 2
        cv2.putText(frame, nav_text, (nav_x, h - 5),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)

    def _draw_fps(self, frame):
        """Draw FPS counter (small, top-left)."""
        if self.show_fps:
            current_time = time.time()
            dt = current_time - self.prev_time
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
            self.prev_time = current_time

    def _draw_hud(self, frame, detections):
        """Draw the standard mode HUD elements."""
        h, w = frame.shape[:2]

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

        count_text = f"Detections: {len(detections)}"
        cv2.putText(
            frame, count_text, (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
        )

        if self.mode_label:
            (mw, mh), _ = cv2.getTextSize(
                self.mode_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            cv2.putText(
                frame, self.mode_label, (w - mw - 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )

        legend_y = h - 60
        for cat, col in self.CATEGORY_COLORS.items():
            cv2.circle(frame, (15, legend_y), 6, col, -1)
            cv2.putText(
                frame, cat.capitalize(), (28, legend_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1
            )
            legend_y += 20
