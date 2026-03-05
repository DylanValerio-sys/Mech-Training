"""
Mech Training AI - Real-time automotive service guidance overlay.

Usage:
    python main.py --service                  Service mode (guided procedure)
    python main.py --service --cam 1          Service mode with camera index 1
    python main.py --service --mute           Service mode without audio
    python main.py --auto                     Automotive detection mode
    python main.py                            Standard detection mode

DroidCam setup:
    1. Install DroidCam app on phone + DroidCam Client on PC
    2. Connect them (they'll show "connected")
    3. Run: python main.py --service --cam 1   (try 1 or 2 for DroidCam)

Service Mode Controls:
    SPACE       - Next step
    BACKSPACE   - Previous step
    1-7         - Jump to section
    q           - Quit

Detection Mode Controls:
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


def run_service_mode(config, camera_source, muted):
    """Run the guided service procedure mode."""
    from modules.procedure import ProcedureGuide

    print("=" * 50)
    print("  Mech Training AI - SERVICE MODE")
    print("=" * 50)

    # Load service procedure
    procedure = ProcedureGuide()
    print(f"  Vehicle: {procedure.vehicle}")
    print(f"  Service: {procedure.service_type}")
    print(f"  Steps: {procedure.total_steps}")
    print(f"  Audio: {'OFF' if muted else 'ON'}")
    print("=" * 50)

    # Initialize camera
    print("\nOpening camera...")
    camera = Camera(
        source=camera_source,
        width=config["frame_width"],
        height=config["frame_height"],
    )

    # Initialize overlay
    overlay = OverlayRenderer(
        show_fps=config["show_fps"],
        show_confidence=config["show_confidence"],
    )

    # Initialize audio
    audio = None
    if not muted:
        audio = AudioGuidance(cooldown=5.0, rate=150)
        print("  Audio guidance: ON")

    # Optionally load detector for background detection
    detector = None
    if "--detect" in sys.argv:
        from modules.detector import Detector
        print("  Loading YOLO detector for background detection...")
        detector = Detector(
            model_path=config["model_path"],
            confidence=config["confidence_threshold"],
            mode="automotive",
            infer_size=640,
            skip_frames=4,
        )

    print("\nReady! Navigate through service steps with SPACE/BACKSPACE.")
    print("Controls: SPACE next | BACKSPACE prev | 1-7 section | Q quit\n")

    # Announce first step
    if audio:
        audio.announce_step(procedure.get_announcement_text())

    while True:
        frame = camera.get_frame()
        if frame is None:
            print("Camera disconnected.")
            break

        # Optional background detection
        detections = []
        if detector:
            detections = detector.detect(frame)

        # Draw service procedure overlay
        annotated = overlay.draw_service(frame, procedure, detections)

        # Show the annotated frame
        cv2.imshow(config["window_name"], annotated)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Quitting...")
            break
        elif key == 32:  # SPACE - next step
            if procedure.next_step():
                step = procedure.current_step
                print(f"  Step {step['check']}: {step['title']}")
                if audio:
                    audio.announce_step(procedure.get_announcement_text())
            else:
                print("  Already at the last step!")
                if audio:
                    audio.announce_step("Service procedure complete. All steps finished.")
        elif key == 8:  # BACKSPACE - previous step
            if procedure.prev_step():
                step = procedure.current_step
                print(f"  Step {step['check']}: {step['title']}")
                if audio:
                    audio.announce_step(procedure.get_announcement_text())
            else:
                print("  Already at the first step!")
        elif key in range(ord("1"), ord("8")):  # Keys 1-7: jump to section
            section_id = key - ord("0")
            if procedure.jump_to_section(section_id):
                step = procedure.current_step
                print(f"\n  Jumped to Section {section_id}: {step['section_name']}")
                print(f"  Step {step['check']}: {step['title']}")
                if audio:
                    section_name = step["section_name"]
                    audio.announce_step(
                        f"Section {section_id}. {section_name}. "
                        + procedure.get_announcement_text()
                    )

    if audio:
        audio.shutdown()
    camera.release()
    cv2.destroyAllWindows()
    print("Shutdown complete.")


def run_detection_mode(config, camera_source, muted):
    """Run the standard or automotive detection mode."""
    from modules.detector import Detector

    mode = "automotive" if "--auto" in sys.argv else "standard"

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
        infer_size=640,
        skip_frames=4,
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

        detections = detector.detect(frame)

        if audio and detections:
            audio.announce_detections(detections)

        annotated = overlay.draw(frame, detections)

        conf_text = f"Confidence: {detector.confidence:.0%}"
        cv2.putText(
            annotated, conf_text, (10, annotated.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
        )

        cv2.imshow(config["window_name"], annotated)

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


def main():
    config = load_config()
    muted = "--mute" in sys.argv
    camera_source = get_camera_source(config)

    if "--service" in sys.argv:
        run_service_mode(config, camera_source, muted)
    else:
        run_detection_mode(config, camera_source, muted)


if __name__ == "__main__":
    main()
