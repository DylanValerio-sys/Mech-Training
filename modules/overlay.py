"""Modern AR overlay renderer for Mech Training AI.

Clean, performant overlay designed for AR glasses:
- Lightweight detection borders with corner accents (no heavy glow)
- Muddy agent status badge (IDLE/LISTENING/THINKING/SPEAKING)
- Subtitle bar for voice agent responses
- HUD-style frame elements
- Smooth, fluid rendering optimized for moving camera
"""

import cv2
import numpy as np
import time
import math
import textwrap


class OverlayRenderer:
    """Draws modern AR overlay with clean highlights and Muddy agent UI."""

    # ── Color Palette (BGR) ─────────────────────────────────────────
    CYAN       = (220, 220, 0)      # Primary accent
    AMBER      = (0, 180, 255)      # Warning
    RED        = (60, 40, 255)       # Error / critical
    GREEN      = (100, 230, 0)      # OK / listening
    WHITE      = (240, 240, 240)    # Primary text
    LIGHT      = (200, 200, 200)    # Secondary text
    DIM        = (120, 120, 120)    # Tertiary text
    DARK_BG    = (25, 22, 18)       # Panel background
    NEAR_BLACK = (12, 10, 8)        # Deep background

    # State colors for Muddy badge
    STATE_COLORS = {
        "IDLE":      (120, 120, 120),   # Gray
        "LISTENING": (100, 230, 0),     # Green
        "THINKING":  (0, 180, 255),     # Amber
        "SPEAKING":  (220, 220, 0),     # Cyan
        "ERROR":     (60, 40, 255),     # Red
    }

    def __init__(self, show_fps=True, show_confidence=True, knowledge_db=None):
        self.show_fps = show_fps
        self.show_confidence = show_confidence
        self.knowledge_db = knowledge_db
        self.muddy = None  # Set by main.py after init

        # Timing
        self.prev_time = time.time()
        self.fps = 0.0
        self._start_time = time.time()

    # ================================================================
    #  PUBLIC API
    # ================================================================

    def draw(self, frame, detections):
        """Draw the complete AR overlay onto the frame."""
        h, w = frame.shape[:2]
        now = time.time()

        # HUD frame corners
        self._draw_hud_corners(frame, w, h)

        # Top status bar
        self._draw_top_bar(frame, w, h, len(detections) if detections else 0)

        # Detection overlays
        if detections:
            sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
            # Primary detection (brightest, with accents)
            self._draw_primary(frame, sorted_dets[0], w, h)
            # Secondary detections (subtle)
            for det in sorted_dets[1:]:
                self._draw_secondary(frame, det, w, h)
        else:
            self._draw_empty_state(frame, w, h)

        # Muddy agent UI
        if self.muddy:
            self._draw_muddy_badge(frame, w, h, now)
            self._draw_subtitles(frame, w, h)
            self._draw_controls_bar_muddy(frame, w, h)
        else:
            self._draw_controls_bar_basic(frame, w, h)

        self._update_fps()
        return frame

    # ---- Service mode (kept for compatibility) ----
    def draw_service(self, frame, procedure, detections=None):
        """Draw service procedure overlay on the frame."""
        h, w = frame.shape[:2]
        if detections:
            for det in detections:
                x1, y1, x2, y2 = det["box"]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.CYAN, 2)
                label = f"{det['class_name']} {det['confidence']:.0%}"
                self._put_text(frame, label, (x1 + 4, y1 - 8), 0.45, self.CYAN, 1)

        step = procedure.current_step
        self._draw_service_panel(frame, w, h, step, procedure)
        return frame

    # ================================================================
    #  DETECTION HIGHLIGHTS (lightweight, fluid)
    # ================================================================

    def _draw_primary(self, frame, det, w, h):
        """Draw the primary detection with clean border + corner accents + label."""
        x1, y1, x2, y2 = det["box"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            return

        # Get category color
        color = self.CYAN
        cat_label = ""
        if self.knowledge_db:
            cat_info = self.knowledge_db.get_category_info(det["class_name"])
            color = cat_info.get("color", self.CYAN)
            cat_label = cat_info.get("label", "")

        # Clean 2px border
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Corner accents (thicker, overlapping the rectangle)
        corner = max(12, min(28, (x2 - x1) // 5, (y2 - y1) // 5))
        ct = 3
        cv2.line(frame, (x1, y1), (x1 + corner, y1), color, ct)
        cv2.line(frame, (x1, y1), (x1, y1 + corner), color, ct)
        cv2.line(frame, (x2, y1), (x2 - corner, y1), color, ct)
        cv2.line(frame, (x2, y1), (x2, y1 + corner), color, ct)
        cv2.line(frame, (x1, y2), (x1 + corner, y2), color, ct)
        cv2.line(frame, (x1, y2), (x1, y2 - corner), color, ct)
        cv2.line(frame, (x2, y2), (x2 - corner, y2), color, ct)
        cv2.line(frame, (x2, y2), (x2, y2 - corner), color, ct)

        # Center crosshair
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.line(frame, (cx - 10, cy), (cx - 3, cy), color, 1)
        cv2.line(frame, (cx + 3, cy), (cx + 10, cy), color, 1)
        cv2.line(frame, (cx, cy - 10), (cx, cy - 3), color, 1)
        cv2.line(frame, (cx, cy + 3), (cx, cy + 10), color, 1)
        cv2.circle(frame, (cx, cy), 2, color, -1)

        # Floating label
        name = det["class_name"].replace("_", " ").upper()
        conf = det["confidence"]
        label = f"{name} {conf:.0%}"
        if cat_label:
            label = f"{cat_label} | {label}"

        # Position label above box (or below if no room)
        label_y = y1 - 10 if y1 > 30 else y2 + 18
        label_x = max(4, x1)

        self._put_text(frame, label, (label_x, label_y), 0.50, color, 1)

    def _draw_secondary(self, frame, det, w, h):
        """Draw secondary detections as subtle thin outlines with small labels."""
        x1, y1, x2, y2 = det["box"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.DIM, 1)

        name = det["class_name"].replace("_", " ")
        self._put_text(frame, name, (x1 + 2, y1 - 4), 0.32, self.DIM, 1)

    # ================================================================
    #  MUDDY AGENT UI
    # ================================================================

    def _draw_muddy_badge(self, frame, w, h, now):
        """Draw the Muddy status badge in the top-right area."""
        state = self.muddy.state
        color = self.STATE_COLORS.get(state, self.DIM)

        # Badge position (below top bar)
        badge_x = w - 180
        badge_y = 40

        # Badge background
        overlay = frame.copy()
        cv2.rectangle(overlay, (badge_x, badge_y), (w - 8, badge_y + 28),
                       self.DARK_BG, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Left accent
        cv2.line(frame, (badge_x, badge_y), (badge_x, badge_y + 28), color, 3)

        # Status dot (pulsing for LISTENING)
        dot_x = badge_x + 14
        dot_y = badge_y + 14
        dot_radius = 5

        if state == "LISTENING":
            # Pulsing green dot
            pulse = 0.5 + 0.5 * math.sin(now * 6)
            r = int(dot_radius + 2 * pulse)
            cv2.circle(frame, (dot_x, dot_y), r, color, -1)
        elif state == "THINKING":
            # Animated dots
            for i in range(3):
                offset = int(3 * math.sin(now * 4 + i * 1.5))
                cv2.circle(frame, (dot_x + i * 8, dot_y + offset), 3, color, -1)
        elif state == "SPEAKING":
            # Sound wave bars
            for i in range(4):
                bar_h = int(4 + 6 * abs(math.sin(now * 5 + i * 1.2)))
                bx = dot_x + i * 5
                cv2.line(frame, (bx, dot_y - bar_h), (bx, dot_y + bar_h), color, 2)
        else:
            # Static dot
            cv2.circle(frame, (dot_x, dot_y), dot_radius, color, -1)

        # State text
        state_label = f"MUDDY  {state}"
        if state == "ERROR" and self.muddy.error_msg:
            state_label = f"MUDDY  ERR"
        cv2.putText(frame, state_label, (dot_x + 14, badge_y + 19),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)

    def _draw_subtitles(self, frame, w, h):
        """Draw the question/answer subtitle bar at the bottom."""
        muddy = self.muddy
        if not muddy:
            return

        # Only show subtitles if we have content and it's recent (< 8 seconds)
        has_question = bool(muddy.subtitle_question)
        has_answer = bool(muddy.subtitle_answer)
        age = muddy.subtitle_age

        if not has_answer and not (muddy.state in ("LISTENING", "THINKING")):
            return
        if has_answer and age > 8 and muddy.state == "IDLE":
            return

        # Subtitle area dimensions
        sub_h = 70 if has_answer else 35
        sub_y = h - sub_h - 28  # Above controls bar

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, sub_y), (w - 8, sub_y + sub_h), self.DARK_BG, -1)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

        font = cv2.FONT_HERSHEY_SIMPLEX
        y = sub_y + 18

        # Show question
        if has_question:
            q_text = f"Q: {muddy.subtitle_question}"
            # Truncate if too long
            max_chars = w // 8
            if len(q_text) > max_chars:
                q_text = q_text[:max_chars - 3] + "..."
            cv2.putText(frame, q_text, (18, y), font, 0.38, self.DIM, 1, cv2.LINE_AA)
            y += 18

        # Show answer
        if has_answer:
            # Fade out effect (reduce alpha as age increases past 5s)
            a_color = self.WHITE
            if age > 5:
                fade = max(0.3, 1.0 - (age - 5) / 3.0)
                a_color = tuple(int(c * fade) for c in self.WHITE)

            # Word wrap the answer
            max_chars = w // 8
            for line in textwrap.wrap(muddy.subtitle_answer, width=max_chars)[:2]:
                cv2.putText(frame, line, (18, y), font, 0.42, a_color, 1, cv2.LINE_AA)
                y += 18

        elif muddy.state == "LISTENING":
            self._put_text(frame, "Listening...", (18, y), 0.45, self.GREEN, 1)
        elif muddy.state == "THINKING":
            self._put_text(frame, "Thinking...", (18, y), 0.45, self.AMBER, 1)

    # ================================================================
    #  HUD ELEMENTS
    # ================================================================

    def _draw_hud_corners(self, frame, w, h):
        """Draw L-shaped corner brackets for AR HUD feel."""
        corner = 30
        t = 2
        c = self.DIM
        cv2.line(frame, (4, 4), (4 + corner, 4), c, t)
        cv2.line(frame, (4, 4), (4, 4 + corner), c, t)
        cv2.line(frame, (w - 5, 4), (w - 5 - corner, 4), c, t)
        cv2.line(frame, (w - 5, 4), (w - 5, 4 + corner), c, t)
        cv2.line(frame, (4, h - 5), (4 + corner, h - 5), c, t)
        cv2.line(frame, (4, h - 5), (4, h - 5 - corner), c, t)
        cv2.line(frame, (w - 5, h - 5), (w - 5 - corner, h - 5), c, t)
        cv2.line(frame, (w - 5, h - 5), (w - 5, h - 5 - corner), c, t)

    def _draw_top_bar(self, frame, w, h, det_count):
        """Draw the top status bar."""
        bar_h = 32
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), self.NEAR_BLACK, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.line(frame, (0, bar_h), (w, bar_h), self.CYAN, 1)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Left: FPS
        if self.show_fps:
            cv2.putText(frame, f"FPS {self.fps:.0f}", (12, 22), font, 0.45, self.DIM, 1)

        # Center: App name
        title = "MECH TRAINING AI"
        (tw, _), _ = cv2.getTextSize(title, font, 0.45, 1)
        cv2.putText(frame, title, ((w - tw) // 2, 22), font, 0.45, self.CYAN, 1)

        # Right: Detection count
        right_text = f"DETECTED: {det_count}"
        (rw, _), _ = cv2.getTextSize(right_text, font, 0.38, 1)
        cv2.putText(frame, right_text, (w - rw - 12, 22), font, 0.38, self.LIGHT, 1)

    def _draw_controls_bar_muddy(self, frame, w, h):
        """Draw bottom controls bar when Muddy is active."""
        bar_h = 24
        bar_y = h - bar_h
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, bar_y), (w, h), self.NEAR_BLACK, -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

        controls = 'Say "Hey Muddy" to ask a question   |   [+/-] Confidence   [Q] Quit'
        font = cv2.FONT_HERSHEY_SIMPLEX
        (cw, _), _ = cv2.getTextSize(controls, font, 0.33, 1)
        cv2.putText(frame, controls, ((w - cw) // 2, h - 8), font, 0.33, self.DIM, 1)

    def _draw_controls_bar_basic(self, frame, w, h):
        """Draw bottom controls bar without Muddy."""
        bar_h = 24
        bar_y = h - bar_h
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, bar_y), (w, h), self.NEAR_BLACK, -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

        controls = "[+/-] Confidence   [Q] Quit"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (cw, _), _ = cv2.getTextSize(controls, font, 0.33, 1)
        cv2.putText(frame, controls, ((w - cw) // 2, h - 8), font, 0.33, self.DIM, 1)

    def _draw_empty_state(self, frame, w, h):
        """Draw 'no detection' scanning indicator."""
        elapsed = time.time() - self._start_time
        dots = "." * (int(elapsed * 2) % 4)
        text = f"SCANNING{dots}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, _), _ = cv2.getTextSize(text, font, 0.6, 1)
        self._put_text(frame, text, ((w - tw) // 2, h // 2), 0.6, self.DIM, 1)

        hint = "Point camera at engine components"
        (hw, _), _ = cv2.getTextSize(hint, font, 0.4, 1)
        self._put_text(frame, hint, ((w - hw) // 2, h // 2 + 28), 0.4, self.DIM, 1)

    # ================================================================
    #  SERVICE MODE (backward compat)
    # ================================================================

    def _draw_service_panel(self, frame, w, h, step, procedure):
        """Draw service step panel at bottom."""
        panel_h = 140
        panel_y = h - panel_h
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, panel_y), (w, h), self.DARK_BG, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.line(frame, (0, panel_y), (w, panel_y), self.CYAN, 2)

        # Progress bar
        bar_w = w - 20
        bar_fill = int(bar_w * procedure.progress_fraction)
        cv2.rectangle(frame, (10, panel_y + 6), (10 + bar_w, panel_y + 10), (60, 60, 60), -1)
        if bar_fill > 0:
            cv2.rectangle(frame, (10, panel_y + 6), (10 + bar_fill, panel_y + 10), self.CYAN, -1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        y = panel_y + 30
        check_text = f"#{step['check']}"
        cv2.putText(frame, check_text, (15, y), font, 0.7, self.CYAN, 2)
        cv2.putText(frame, step["title"].upper(), (65, y), font, 0.55, self.WHITE, 1)

        y += 22
        cv2.putText(frame, step["description"], (15, y), font, 0.4, self.LIGHT, 1)

        detail = step.get("detail", "")
        if detail:
            y += 18
            for line in textwrap.wrap(detail, width=90)[:2]:
                cv2.putText(frame, line, (15, y), font, 0.35, self.DIM, 1)
                y += 15

        info_y = h - 12
        if step["tools"]:
            cv2.putText(frame, f"TOOLS: {step['tools']}", (15, info_y), font, 0.38, self.AMBER, 1)
        if step["torque"]:
            torque_text = f"TORQUE: {step['torque']}"
            (tw, _), _ = cv2.getTextSize(torque_text, font, 0.38, 1)
            cv2.putText(frame, torque_text, (w - tw - 15, info_y), font, 0.38, self.RED, 1)

    # ================================================================
    #  HELPERS
    # ================================================================

    def _put_text(self, frame, text, pos, scale=0.5, color=None, thickness=1):
        """Draw text with dark outline for readability on any background."""
        if color is None:
            color = self.WHITE
        x, y = pos
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                     (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                     color, thickness, cv2.LINE_AA)

    def _update_fps(self):
        """Update FPS counter with exponential moving average."""
        now = time.time()
        dt = now - self.prev_time
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
        self.prev_time = now
