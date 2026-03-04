"""
Mech Training AI - Real-time automotive part detection overlay.

Usage:
    python main.py              Run in standard mode (general object detection)
    python main.py --auto       Run in automotive mode (YOLO-World parts/tools)

Controls:
    q     - Quit the application
    +/-   - Increase/decrease confidence threshold
    m     - Toggle between standard and automotive mode
"""

import cv2
import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.config import load_config
from modules.camera import Camera
from modules.detector import Detector
from modules.overlay import OverlayRenderer


def main():
    config = load_config()

    # Check command-line flag for automotive mode
    mode = "automotive" if "--auto" in sys.argv else "standard"

    print("=" * 50)
    print("  Mech Training AI - Starting up...")
    print("=" * 50)
    print(f"  Mode: {mode.upper()}")
    print(f"  Confidence: {config['confidence_threshold']}")
    print(f"  Resolution: {config['frame_width']}x{config['frame_height']}")
    print("=" * 50)

    # Initialize camera
    print("\nOpening camera...")
    camera = Camera(
        source=config["camera_index"],
        width=config["frame_width"],
        height=config["frame_height"],
    )

    # Initialize detector
    if mode == "automotive":
        print("Loading YOLO-World automotive model (first run downloads ~150MB)...")
    else:
        print("Loading YOLOv8n model (first run downloads ~6MB)...")

    detector = Detector(
        model_path=config["model_path"],
        confidence=config["confidence_threshold"],
        mode=mode,
    )

    # Initialize overlay
    overlay = OverlayRenderer(
        show_fps=config["show_fps"],
        show_confidence=config["show_confidence"],
    )
    overlay.mode_label = f"MODE: {mode.upper()}"

    print("\nReady! Point your camera at automotive parts or tools.")
    print("Controls: 'q' quit | '+'/'-' confidence | 'm' toggle mode\n")

    while True:
        frame = camera.get_frame()
        if frame is None:
            print("Camera disconnected.")
            break

        # Detect objects in the current frame
        detections = detector.detect(frame)

        # Draw overlay on the frame
        annotated = overlay.draw(frame, detections)

        # Display confidence threshold on screen
        conf_text = f"Confidence: {detector.confidence:.0%}"
        cv2.putText(
            annotated, conf_text, (10, annotated.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
        )

        # Show the annotated frame
        cv2.imshow(config["window_name"], annotated)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Quitting...")
            break
        elif key == ord("+") or key == ord("="):
            detector.confidence = min(detector.confidence + 0.05, 0.95)
            print(f"Confidence threshold: {detector.confidence:.0%}")
        elif key == ord("-"):
            detector.confidence = max(detector.confidence - 0.05, 0.10)
            print(f"Confidence threshold: {detector.confidence:.0%}")
        elif key == ord("m"):
            # Toggle mode — requires reloading the model
            camera.release()
            cv2.destroyAllWindows()
            new_mode = "automotive" if detector.mode == "standard" else "standard"
            print(f"\nSwitching to {new_mode.upper()} mode...")
            detector = Detector(
                model_path=config["model_path"],
                confidence=detector.confidence,
                mode=new_mode,
            )
            overlay.mode_label = f"MODE: {new_mode.upper()}"
            camera = Camera(
                source=config["camera_index"],
                width=config["frame_width"],
                height=config["frame_height"],
            )
            print("Ready!\n")

    camera.release()
    cv2.destroyAllWindows()
    print("Shutdown complete.")


if __name__ == "__main__":
    main()
