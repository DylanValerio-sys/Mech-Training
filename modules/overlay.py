"""Modern AR-style overlay renderer for Mech Training AI.

Renders detection overlays with:
- Neon glow borders fitted to detected component contours
- Floating label cards with category badges
- Expandable knowledge/fault info panel
- HUD-style screen frame elements
- Scan animation on new detections
- Single-part spotlight with cycling support
"""

import cv2
import numpy as np
import time
import textwrap


class OverlayRenderer:
    """Draws modern AR overlay with neon glow highlights and info panels."""

    # ── Color Palette (BGR) ─────────────────────────────────────────
    CYAN       = (220, 220, 0)      # Primary accent — bright cyan
    CYAN_GLOW  = (200, 180, 0)      # Dimmer glow layer
    AMBER      = (0, 180, 255)      # Warning accent
    RED        = (60, 40, 255)       # Error / critical
    GREEN      = (100, 230, 0)      # OK / success
    WHITE      = (240, 240, 240)    # Primary text
    LIGHT      = (200, 200, 200)    # Secondary text
    DIM        = (120, 120, 120)    # Tertiary text
    DARK_BG    = (25, 22, 18)       # Panel background
    NEAR_BLACK = (12, 10, 8)        # Deep background

    # ── Timing ──────────────────────────────────────────────────────
    SCAN_DURATION  = 0.8    # New detection scan animation (seconds)
    PULSE_PERIOD   = 2.0    # Glow pulse cycle (seconds)

    def __init__(self, show_fps=True, show_confidence=True, knowledge_db=None):
        self.show_fps = show_fps
        self.show_confidence = show_confidence
        self.knowledge_db = knowledge_db
        self.mode_label = ""

        # Timing
        self.prev_time = time.time()
        self.fps = 0.0
        self._start_time = time.time()

        # State
        self._last_primary = None
        self._scan_start = 0.0
        self._selected_idx = 0
        self._show_info = False
        self._contour_cache = None
        self._contour_class = None

    # ================================================================
    #  PUBLIC API
    # ================================================================

    def draw(self, frame, detections, show_info=False):
        """Draw the complete AR overlay onto the frame."""
        h, w = frame.shape[:2]
        self._show_info = show_info
        now = time.time()

        # Always draw HUD frame
        self._draw_hud_corners(frame, w, h)

        if not detections:
            self._draw_empty_state(frame, w, h)
            self._draw_top_bar(frame, w, h, 0)
            self._update_fps()
            return frame

        # Sort by confidence
        sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)

        # Clamp selection
        if self._selected_idx >= len(sorted_dets):
            self._selected_idx = 0

        primary = sorted_dets[self._selected_idx]
        others = [d for i, d in enumerate(sorted_dets) if i != self._selected_idx]

        # Check for new primary detection → trigger scan animation
        current_name = primary["class_name"]
        if current_name != self._last_primary:
            self._last_primary = current_name
            self._scan_start = now
            self._contour_cache = None
            self._contour_class = None

        scan_age = now - self._scan_start
        is_scanning = scan_age < self.SCAN_DURATION

        # --- Draw layers (back to front) ---

        # 1. Secondary detections (subtle dotted outlines)
        for det in others:
            self._draw_secondary(frame, det, w, h)

        # 2. Primary detection — neon glow highlight
        contour = self._get_contour(frame, primary)
        self._draw_primary_highlight(frame, primary, contour, w, h, is_scanning, scan_age, now)

        # 3. Floating label card
        self._draw_label_card(frame, primary, w, h)

        # 4. Knowledge / info panel (if toggled)
        if self._show_info and self.knowledge_db:
            self._draw_info_panel(frame, primary, w, h)

        # 5. Top HUD bar
        self._draw_top_bar(frame, w, h, len(detections))

        # 6. Bottom controls hint
        self._draw_controls_bar(frame, w, h)

        self._update_fps()
        return frame

    def cycle_selection(self, direction=1):
        """Cycle the selected detection (1=next, -1=prev)."""
        self._selected_idx += direction
        self._contour_cache = None
        self._contour_class = None

    def toggle_info(self):
        """Toggle the info panel visibility."""
        self._show_info = not self._show_info
        return self._show_info

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
        self._draw_service_progress(frame, w, procedure)
        self._draw_service_panel(frame, w, h, step, procedure)
        return frame

    # ================================================================
    #  CONTOUR EXTRACTION
    # ================================================================

    def _get_contour(self, frame, det):
        """Extract or return cached contour for the primary detection."""
        name = det["class_name"]
        if self._contour_cache is not None and self._contour_class == name:
            return self._contour_cache

        contour = self._extract_contour(frame, det["box"])
        self._contour_cache = contour
        self._contour_class = name
        return contour

    def _extract_contour(self, frame, box):
        """Use edge detection within the bounding box to find part contours."""
        x1, y1, x2, y2 = box
        h, w = frame.shape[:2]

        # Clamp and pad
        pad = 4
        x1c = max(0, x1 - pad)
        y1c = max(0, y1 - pad)
        x2c = min(w, x2 + pad)
        y2c = min(h, y2 + pad)

        if x2c <= x1c or y2c <= y1c:
            return None

        roi = frame[y1c:y2c, x1c:x2c]
        if roi.size == 0:
            return None

        # Edge detection
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)

        # Connect nearby edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Get the largest contour
        largest = max(contours, key=cv2.contourArea)

        # Must be at least 10% of the ROI area to be meaningful
        roi_area = (x2c - x1c) * (y2c - y1c)
        if cv2.contourArea(largest) < roi_area * 0.08:
            return None

        # Offset contour back to frame coordinates
        largest = largest + np.array([x1c, y1c])
        return largest

    # ================================================================
    #  PRIMARY DETECTION
    # ================================================================

    def _draw_primary_highlight(self, frame, det, contour, w, h, is_scanning, scan_age, now):
        """Draw neon glow highlight on the primary detected part."""
        x1, y1, x2, y2 = det["box"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        if x2 <= x1 or y2 <= y1:
            return

        # Get category color
        color = self.CYAN
        if self.knowledge_db:
            cat_info = self.knowledge_db.get_category_info(det["class_name"])
            color = cat_info.get("color", self.CYAN)

        # Pulse effect (subtle brightness oscillation)
        pulse = 0.85 + 0.15 * np.sin(now * 2 * np.pi / self.PULSE_PERIOD)

        # Scan animation: sweep line across detection box
        if is_scanning:
            scan_progress = scan_age / self.SCAN_DURATION
            scan_y = int(y1 + (y2 - y1) * scan_progress)
            # Bright scan line
            overlay = frame.copy()
            cv2.line(overlay, (x1, scan_y), (x2, scan_y), self.WHITE, 2)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # --- Neon glow effect ---
        if contour is not None and len(contour) > 4:
            # Draw contour with glow layers
            self._draw_neon_contour(frame, contour, color, pulse)
        else:
            # Fallback: glow rectangle
            self._draw_neon_rect(frame, (x1, y1), (x2, y2), color, pulse)

        # --- Subtle fill ---
        fill_alpha = 0.08 * pulse
        overlay = frame.copy()
        if contour is not None and len(contour) > 4:
            cv2.drawContours(overlay, [contour], -1, color, -1)
        else:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, fill_alpha, frame, 1.0 - fill_alpha, 0, frame)

        # --- Crosshair at center ---
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        ch_size = 12
        ch_gap = 4
        c_scaled = tuple(int(v * pulse) for v in color)
        cv2.line(frame, (cx - ch_size, cy), (cx - ch_gap, cy), c_scaled, 1)
        cv2.line(frame, (cx + ch_gap, cy), (cx + ch_size, cy), c_scaled, 1)
        cv2.line(frame, (cx, cy - ch_size), (cx, cy - ch_gap), c_scaled, 1)
        cv2.line(frame, (cx, cy + ch_gap), (cx, cy + ch_size), c_scaled, 1)
        cv2.circle(frame, (cx, cy), 2, c_scaled, -1)

    def _draw_neon_contour(self, frame, contour, color, pulse=1.0):
        """Draw a contour with neon glow effect (multiple layers)."""
        # Outer glow (thick, transparent)
        overlay = frame.copy()
        cv2.drawContours(overlay, [contour], -1, color, 8)
        cv2.addWeighted(overlay, 0.10 * pulse, frame, 1.0 - 0.10 * pulse, 0, frame)

        # Mid glow
        overlay2 = frame.copy()
        cv2.drawContours(overlay2, [contour], -1, color, 4)
        cv2.addWeighted(overlay2, 0.20 * pulse, frame, 1.0 - 0.20 * pulse, 0, frame)

        # Sharp inner contour
        cv2.drawContours(frame, [contour], -1, color, 2)

    def _draw_neon_rect(self, frame, pt1, pt2, color, pulse=1.0):
        """Draw a rectangle with neon glow effect."""
        x1, y1 = pt1
        x2, y2 = pt2

        # Outer glow
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), color, 6)
        cv2.addWeighted(overlay, 0.10 * pulse, frame, 1.0 - 0.10 * pulse, 0, frame)

        # Mid glow
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), color, 3)
        cv2.addWeighted(overlay2, 0.20 * pulse, frame, 1.0 - 0.20 * pulse, 0, frame)

        # Sharp border
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Corner accents
        corner = max(10, min(30, (x2 - x1) // 5, (y2 - y1) // 5))
        ct = 3
        cv2.line(frame, (x1, y1), (x1 + corner, y1), color, ct)
        cv2.line(frame, (x1, y1), (x1, y1 + corner), color, ct)
        cv2.line(frame, (x2, y1), (x2 - corner, y1), color, ct)
        cv2.line(frame, (x2, y1), (x2, y1 + corner), color, ct)
        cv2.line(frame, (x1, y2), (x1 + corner, y2), color, ct)
        cv2.line(frame, (x1, y2), (x1, y2 - corner), color, ct)
        cv2.line(frame, (x2, y2), (x2 - corner, y2), color, ct)
        cv2.line(frame, (x2, y2), (x2, y2 - corner), color, ct)

    # ================================================================
    #  SECONDARY DETECTIONS
    # ================================================================

    def _draw_secondary(self, frame, det, w, h):
        """Draw secondary detections as subtle thin outlines with tiny labels."""
        x1, y1, x2, y2 = det["box"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.DIM, 1)

        # Tiny label
        name = det["class_name"].replace("_", " ")
        self._put_text(frame, name, (x1 + 2, y1 - 4), 0.32, self.DIM, 1)

    # ================================================================
    #  FLOATING LABEL CARD
    # ================================================================

    def _draw_label_card(self, frame, det, w, h):
        """Draw a modern floating label card near the detection."""
        x1, y1, x2, y2 = det["box"]
        name = det["class_name"].replace("_", " ").upper()
        conf = det["confidence"]

        # Get category info
        cat_label = ""
        cat_color = self.CYAN
        if self.knowledge_db:
            cat_info = self.knowledge_db.get_category_info(det["class_name"])
            cat_label = cat_info.get("label", "")
            cat_color = cat_info.get("color", self.CYAN)

        # Card dimensions
        font = cv2.FONT_HERSHEY_SIMPLEX
        name_scale = 0.6
        name_thick = 2
        (name_w, name_h), _ = cv2.getTextSize(name, font, name_scale, name_thick)

        badge_text = f" {cat_label} " if cat_label else ""
        (badge_w, badge_h), _ = cv2.getTextSize(badge_text, font, 0.38, 1)

        conf_text = f"{conf:.0%}"
        (conf_w, conf_h), _ = cv2.getTextSize(conf_text, font, 0.42, 1)

        card_w = max(name_w + 24, badge_w + conf_w + 30)
        card_h = 52

        # Position: prefer above and to the right of the detection
        # Calculate ideal positions
        cx = (x1 + x2) // 2
        card_x = min(max(8, cx - card_w // 2), w - card_w - 8)

        if y1 > card_h + 20:
            card_y = y1 - card_h - 12
        else:
            card_y = y2 + 12

        # Connecting line from card to detection center
        line_start = (card_x + card_w // 2, card_y + card_h if card_y < y1 else card_y)
        line_end = (cx, y1 if card_y < y1 else y2)
        cv2.line(frame, line_start, line_end, cat_color, 1)

        # Draw card background
        overlay = frame.copy()
        cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h),
                       self.DARK_BG, -1)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

        # Card border (left accent)
        cv2.line(frame, (card_x, card_y), (card_x, card_y + card_h), cat_color, 3)

        # Part name
        text_y = card_y + 22
        cv2.putText(frame, name, (card_x + 10, text_y), font, name_scale, self.WHITE, name_thick)

        # Bottom row: category badge + confidence
        row_y = card_y + card_h - 10
        if badge_text:
            # Category badge (pill shape)
            bx = card_x + 10
            cv2.rectangle(frame, (bx, row_y - badge_h - 2), (bx + badge_w + 4, row_y + 3),
                           cat_color, -1)
            cv2.putText(frame, badge_text, (bx + 2, row_y), font, 0.38, (0, 0, 0), 1)

            # Confidence
            cx_text = bx + badge_w + 12
        else:
            cx_text = card_x + 10

        cv2.putText(frame, conf_text, (cx_text, row_y), font, 0.42, self.LIGHT, 1)

    # ================================================================
    #  INFO PANEL (Expandable)
    # ================================================================

    def _draw_info_panel(self, frame, det, w, h):
        """Draw the expanded knowledge panel at the bottom of the screen."""
        info = self.knowledge_db.lookup(det["class_name"])
        if not info:
            return

        panel_h = 140
        panel_y = h - panel_h - 30  # Above the controls bar

        # Panel background
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, panel_y), (w - 8, panel_y + panel_h), self.DARK_BG, -1)
        cv2.addWeighted(overlay, 0.90, frame, 0.10, 0, frame)

        # Top accent line
        cat_info = self.knowledge_db.get_category_info(det["class_name"])
        accent = cat_info.get("color", self.CYAN)
        cv2.line(frame, (8, panel_y), (w - 8, panel_y), accent, 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        y = panel_y + 20

        # Part name and description
        name = info.get("name", det["class_name"])
        desc = info.get("description", "")
        self._put_text(frame, name.upper(), (18, y), 0.55, accent, 2)
        y += 18

        # Description (wrapped)
        for line in textwrap.wrap(desc, width=80)[:2]:
            self._put_text(frame, line, (18, y), 0.38, self.LIGHT, 1)
            y += 16

        y += 4

        # Faults section
        faults = info.get("faults", [])
        if faults:
            self._put_text(frame, "COMMON FAULTS:", (18, y), 0.38, self.AMBER, 1)
            y += 15
            for fault in faults[:3]:
                severity = fault.get("severity", "medium")
                s_color = self.RED if severity == "high" else self.AMBER if severity == "medium" else self.GREEN
                marker = "!" if severity == "high" else "*"
                text = f"  {marker} {fault['name']}: {fault['visual']}"
                self._put_text(frame, text[:90], (18, y), 0.34, s_color, 1)
                y += 14

        # Tools and safety (right column)
        tools = info.get("tools", [])
        safety = info.get("safety", "")
        right_x = w // 2 + 20
        ry = panel_y + 20

        if tools:
            self._put_text(frame, "TOOLS:", (right_x, ry), 0.38, self.GREEN, 1)
            ry += 15
            tools_text = ", ".join(tools[:4])
            for line in textwrap.wrap(tools_text, width=40)[:2]:
                self._put_text(frame, line, (right_x, ry), 0.34, self.LIGHT, 1)
                ry += 14

        if safety:
            ry += 6
            self._put_text(frame, "SAFETY:", (right_x, ry), 0.38, self.RED, 1)
            ry += 15
            for line in textwrap.wrap(safety, width=40)[:2]:
                self._put_text(frame, line, (right_x, ry), 0.34, self.LIGHT, 1)
                ry += 14

        # Service checks reference
        checks = info.get("service_checks", [])
        if checks:
            check_text = "SERVICE CHECKS: " + ", ".join(f"#{c}" for c in checks)
            self._put_text(frame, check_text, (18, panel_y + panel_h - 8), 0.34, self.DIM, 1)

    # ================================================================
    #  HUD ELEMENTS
    # ================================================================

    def _draw_hud_corners(self, frame, w, h):
        """Draw L-shaped corner brackets for an AR HUD feel."""
        corner = 30
        t = 2
        c = self.DIM

        # Top-left
        cv2.line(frame, (4, 4), (4 + corner, 4), c, t)
        cv2.line(frame, (4, 4), (4, 4 + corner), c, t)
        # Top-right
        cv2.line(frame, (w - 5, 4), (w - 5 - corner, 4), c, t)
        cv2.line(frame, (w - 5, 4), (w - 5, 4 + corner), c, t)
        # Bottom-left
        cv2.line(frame, (4, h - 5), (4 + corner, h - 5), c, t)
        cv2.line(frame, (4, h - 5), (4, h - 5 - corner), c, t)
        # Bottom-right
        cv2.line(frame, (w - 5, h - 5), (w - 5 - corner, h - 5), c, t)
        cv2.line(frame, (w - 5, h - 5), (w - 5, h - 5 - corner), c, t)

    def _draw_top_bar(self, frame, w, h, det_count):
        """Draw the top status bar with FPS, mode, and detection count."""
        bar_h = 32
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), self.NEAR_BLACK, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.line(frame, (0, bar_h), (w, bar_h), self.CYAN, 1)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Left: FPS
        if self.show_fps:
            fps_text = f"FPS {self.fps:.0f}"
            cv2.putText(frame, fps_text, (12, 22), font, 0.45, self.DIM, 1)

        # Center: App name
        title = "MECH TRAINING AI"
        (tw, _), _ = cv2.getTextSize(title, font, 0.45, 1)
        cv2.putText(frame, title, ((w - tw) // 2, 22), font, 0.45, self.CYAN, 1)

        # Right: Detection count + mode
        right_text = f"DETECTED: {det_count}"
        if self.mode_label:
            right_text = f"{self.mode_label}  |  {right_text}"
        (rw, _), _ = cv2.getTextSize(right_text, font, 0.38, 1)
        cv2.putText(frame, right_text, (w - rw - 12, 22), font, 0.38, self.LIGHT, 1)

    def _draw_controls_bar(self, frame, w, h):
        """Draw bottom controls hint bar."""
        bar_h = 24
        bar_y = h - bar_h

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, bar_y), (w, h), self.NEAR_BLACK, -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

        controls = "[TAB] Cycle Parts   [I] Info Panel   [SPACE] Read Aloud   [+/-] Confidence   [Q] Quit"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (cw, _), _ = cv2.getTextSize(controls, font, 0.33, 1)
        cv2.putText(frame, controls, ((w - cw) // 2, h - 8), font, 0.33, self.DIM, 1)

    def _draw_empty_state(self, frame, w, h):
        """Draw 'no detection' state — scanning indicator."""
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Animated scanning dots
        elapsed = time.time() - self._start_time
        dots = "." * (int(elapsed * 2) % 4)
        text = f"SCANNING{dots}"
        (tw, th), _ = cv2.getTextSize(text, font, 0.6, 1)
        self._put_text(frame, text, ((w - tw) // 2, h // 2), 0.6, self.DIM, 1)

        hint = "Point camera at engine components"
        (hw, _), _ = cv2.getTextSize(hint, font, 0.4, 1)
        self._put_text(frame, hint, ((w - hw) // 2, h // 2 + 28), 0.4, self.DIM, 1)

    # ================================================================
    #  SERVICE MODE (kept for backward compatibility)
    # ================================================================

    def _draw_service_progress(self, frame, w, procedure):
        """Draw service progress bar at top."""
        bar_h = 40
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), self.DARK_BG, -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        bar_x, bar_y, bar_w = 10, 28, w - 20
        bar_fill = int(bar_w * procedure.progress_fraction)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 6), (80, 80, 80), -1)
        if bar_fill > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_fill, bar_y + 6), self.CYAN, -1)
        step = procedure.current_step
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"SEC {step['section_id']}: {step['section_name'].upper()}",
                     (12, 20), font, 0.5, self.CYAN, 1)
        counter = f"{procedure.current_index + 1}/{procedure.total_steps}"
        (cw, _), _ = cv2.getTextSize(counter, font, 0.5, 1)
        cv2.putText(frame, counter, (w - cw - 12, 20), font, 0.5, self.WHITE, 1)

    def _draw_service_panel(self, frame, w, h, step, procedure):
        """Draw service step panel at bottom."""
        panel_h = 160
        panel_y = h - panel_h
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, panel_y), (w, h), self.DARK_BG, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.line(frame, (0, panel_y), (w, panel_y), self.CYAN, 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        check_text = f"#{step['check']}"
        badge_x, badge_y = 15, panel_y + 12
        (bw, bh), _ = cv2.getTextSize(check_text, font, 0.8, 2)
        cv2.rectangle(frame, (badge_x - 4, badge_y - 2),
                       (badge_x + bw + 8, badge_y + bh + 6), self.CYAN, -1)
        cv2.putText(frame, check_text, (badge_x + 2, badge_y + bh + 1), font, 0.8, (0, 0, 0), 2)
        cv2.putText(frame, step["title"].upper(), (badge_x + bw + 20, panel_y + 30),
                     font, 0.7, self.WHITE, 2)

        desc_y = panel_y + 55
        for line in textwrap.wrap(step["description"], width=80)[:2]:
            cv2.putText(frame, line, (15, desc_y), font, 0.45, self.DIM, 1)
            desc_y += 18

        detail = step.get("detail", "")
        if detail:
            detail_y = desc_y + 5
            for line in textwrap.wrap(detail, width=90)[:2]:
                cv2.putText(frame, line, (15, detail_y), font, 0.4, self.WHITE, 1)
                detail_y += 16

        info_y = h - 20
        if step["tools"]:
            cv2.putText(frame, f"TOOLS: {step['tools']}", (15, info_y),
                         font, 0.45, self.AMBER, 1)
        if step["torque"]:
            torque_text = f"TORQUE: {step['torque']}"
            (tw, _), _ = cv2.getTextSize(torque_text, font, 0.45, 1)
            cv2.putText(frame, torque_text, (w - tw - 15, info_y), font, 0.45, self.RED, 1)

        nav_text = "SPACE: Next | BACKSPACE: Prev | 1-7: Section | Q: Quit"
        (nw, _), _ = cv2.getTextSize(nav_text, font, 0.35, 1)
        cv2.putText(frame, nav_text, ((w - nw) // 2, h - 5), font, 0.35, self.DIM, 1)

    # ================================================================
    #  HELPERS
    # ================================================================

    def _put_text(self, frame, text, pos, scale=0.5, color=None, thickness=1):
        """Draw text with dark outline for readability on any background."""
        if color is None:
            color = self.WHITE
        x, y = pos
        # Dark outline
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                     (0, 0, 0), thickness + 2, cv2.LINE_AA)
        # Main text
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                     color, thickness, cv2.LINE_AA)

    def _update_fps(self):
        """Update the FPS counter using exponential moving average."""
        now = time.time()
        dt = now - self.prev_time
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
        self.prev_time = now
