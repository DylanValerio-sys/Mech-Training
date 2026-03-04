"""
Mech Training AI - Real-time automotive part detection overlay.

Usage:
    python main.py                          Standard mode (general detection)
    python main.py --auto                   Automotive mode (parts/tools)
    python main.py --auto --cam 1           Use camera index 1 (e.g. DroidCam)
    python main.py --auto --mute            Automotive mode without audio

DroidCam setup:
    1. Install DroidCam app on phone + DroidCam Client on PC
    2. Connect them (they'll show "connected")
    3. Run: python main.py --auto --cam 1   (try 1 or 2 for DroidCam)

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
from modules.audio import AudioGuidance


def get_camera_source(config):
    """Determine camera source from command-line args or config."""
    # Check for --cam flag with index number
    if "--cam" in sys.argv:
        idx = sys.argv.index("--cam")
        if idx + 1 < len(sys.argv):
            cam_idx = int(sys.argv[idx + 1])
            print(f"  Using camera index: {cam_idx}")
            return cam_idx
        else:
            print("ERROR: --cam requires a number (e.g. --cam 1)")
            sys.exit(1)

    # Check for --video flag with file path
    if "--video" in sys.argv:
        idx = sys.argv.index("--video")
        if idx + 1 < len(sys.argv):
            path = sys.argv[idx + 1]
            print(f"  Video file: {path}")
            return path
        else:
            print("ERROR: --video requires a file path")
            sys.exit(1)

    return config["camera_index"]


def main():
    config = load_config()
    mode = "automotive" if "--auto" in sys.argv else "standard"
    muted = "--mute" in sys.argv
    camera_source = get_camera_source(config)

    print("=" * 50)
    print("  Mech Training AI - Starting up...")
    print("=" * 50)
    print(f"  Mode: {mode.upper()}")
    print(f"  Confidence: {config['confidence_threshold']}")
    print(f"  Audio: {'OFF' if muted else 'ON'}")
    print("=" * 50)

    # Initialize camera
    print("\nOpening camera...")
    camera = Camera(
        source=camera_source,
        width=config["frame_width"],
        height=config["frame_height"],
    )

    # Initialize detector
    if mode == "automotive":
        print("Loading YOLO-World automotive model...")
    else:
        print("Loading YOLOv8n model...")

    detector = Detector(
        model_path=config["model_path"],
        confidence=config["confidence_threshold"],
        mode=mode,
        infer_size=640,    # Higher res for better part detection
        skip_frames=4,     # Process every 4th frame for smooth video
    )

    # Initialize overlay
    overlay = OverlayRenderer(
        show_fps=config["show_fps"],
        show_confidence=config["show_confidence"],
    )
    overlay.mode_label = f"MODE: {mode.upper()}"

    # Initialize audio guidance
    audio = None
    if not muted:
        audio = AudioGuidance(cooldown=5.0, rate=160)
        print("  Audio guidance: ON (5s cooldown per part)")

    print("\nReady! Point your camera at automotive parts or tools.")
    print("Controls: 'q' quit | '+'/'-' confidence | 'm' toggle mode\n")

    while True:
        frame = camera.get_frame()
        if frame is None:
            print("Camera disconnected.")
            break

        # Detect objects (uses frame skipping internally for performance)
        detections = detector.detect(frame)

        # Announce detected parts via audio
        if audio and detections:
            audio.announce_detections(detections)

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
            detector.confidence = max(detector.confidence - 0.05, 0.05)
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
                infer_size=640,
                skip_frames=4,
            )
            overlay.mode_label = f"MODE: {new_mode.upper()}"
            camera = Camera(
                source=camera_source,
                width=config["frame_width"],
                height=config["frame_height"],
            )
            print("Ready!\n")

    if audio:
        audio.shutdown()
    camera.release()
    cv2.destroyAllWindows()
    print("Shutdown complete.")


if __name__ == "__main__":
    main()
