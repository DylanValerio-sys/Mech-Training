"""
Mech Training AI - Real-time automotive service guidance overlay.

An AI-powered mechanic training system that detects engine components
through a camera, highlights them with modern AR-style overlays,
and provides fault/error guidance based on service documentation.

Usage:
    python main.py --auto                     Automotive detection mode
    python main.py --auto --cam 2             Detection with DroidCam
    python main.py --auto --mute              Detection without audio
    python main.py --service                  Service procedure mode
    python main.py --service --cam 2          Service mode with DroidCam
    python main.py                            Standard YOLO detection

Detection Mode Controls:
    TAB         - Cycle through detected parts
    I           - Toggle info/knowledge panel
    SPACE       - Read part info aloud
    +/-         - Increase/decrease confidence
    Q           - Quit

Service Mode Controls:
    SPACE       - Next step
    BACKSPACE   - Previous step
    1-7         - Jump to section
    Q           - Quit
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
from modules.knowledge import KnowledgeBase


def get_camera_source(config):
    """Determine camera source from command-line args or config."""
    if "--cam" in sys.argv:
        idx = sys.argv.index("--cam")
        if idx + 1 < len(sys.argv):
            cam_idx = int(sys.argv[idx + 1])
            print(f"  Using camera index: {cam_idx}")
            return cam_idx
        else:
            print("ERROR: --cam requires a number (e.g. --cam 2)")
            sys.exit(1)

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


def run_detection_mode(config, camera_source, muted):
    """Run the automotive part detection mode with AR overlay."""
    from modules.detector import Detector

    mode = "automotive" if "--auto" in sys.argv else "standard"

    print("=" * 56)
    print("   MECH TRAINING AI")
    print("   Real-Time Engine Part Detection & Fault Guidance")
    print("=" * 56)
    print(f"  Mode       : {mode.upper()}")
    print(f"  Confidence : {config['confidence_threshold']}")
    print(f"  Audio      : {'OFF' if muted else 'ON'}")

    # Load knowledge base
    print("\nLoading knowledge base...")
    knowledge = KnowledgeBase()

    # Initialize camera
    print("\nOpening camera...")
    camera = Camera(
        source=camera_source,
        width=config["frame_width"],
        height=config["frame_height"],
    )

    # Initialize detector
    print("\nLoading detection model...")
    detector = Detector(
        model_path=config["model_path"],
        confidence=config["confidence_threshold"],
        mode=mode,
        infer_size=640,
        skip_frames=4,
    )

    # Initialize overlay with knowledge base
    overlay = OverlayRenderer(
        show_fps=config["show_fps"],
        show_confidence=config["show_confidence"],
        knowledge_db=knowledge,
    )
    if detector.using_custom:
        overlay.mode_label = "ENGINE PARTS"
    else:
        overlay.mode_label = f"{mode.upper()}"

    # Initialize audio guidance
    audio = None
    if not muted:
        audio = AudioGuidance(cooldown=5.0, rate=150)

    print("\n" + "=" * 56)
    print("  READY - Point camera at engine components")
    print("=" * 56)
    print("  [TAB]   Cycle detected parts")
    print("  [I]     Toggle info panel (faults, tools, safety)")
    print("  [SPACE] Read part info aloud")
    print("  [+/-]   Adjust confidence threshold")
    print("  [Q]     Quit")
    print("=" * 56 + "\n")

    show_info = False
    last_announced = None

    while True:
        frame = camera.get_frame()
        if frame is None:
            print("Camera disconnected.")
            break

        # Detect parts
        detections = detector.detect(frame)

        # Auto-announce new primary detection
        if audio and detections:
            sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
            idx = min(overlay._selected_idx, len(sorted_dets) - 1)
            primary = sorted_dets[idx]
            primary_name = primary["class_name"]

            if primary_name != last_announced:
                last_announced = primary_name
                # Use knowledge base for richer announcement
                announcement = knowledge.get_announcement_text(primary_name)
                audio.announce_step(announcement)

        # Draw overlay
        annotated = overlay.draw(frame, detections, show_info=show_info)

        # Show confidence in bottom-left
        conf_text = f"Conf: {detector.confidence:.0%}"
        cv2.putText(annotated, conf_text, (10, annotated.shape[0] - 32),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Display
        cv2.imshow(config["window_name"], annotated)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Quitting...")
            break

        elif key == 9:  # TAB - cycle parts
            overlay.cycle_selection(1)
            last_announced = None  # Force re-announce
            print("  [Cycled to next part]")

        elif key == ord("i"):  # I - toggle info panel
            show_info = overlay.toggle_info()
            state = "ON" if show_info else "OFF"
            print(f"  Info panel: {state}")

        elif key == 32:  # SPACE - read aloud
            if audio and detections:
                sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
                idx = min(overlay._selected_idx, len(sorted_dets) - 1)
                primary = sorted_dets[idx]
                info = knowledge.lookup(primary["class_name"])
                if info:
                    name = info.get("name", primary["class_name"])
                    desc = info.get("description", "")
                    faults = info.get("faults", [])
                    text = f"{name}. {desc}."
                    if faults:
                        text += f" Watch for: "
                        text += ". ".join(f["name"] for f in faults[:3]) + "."
                    safety = info.get("safety", "")
                    if safety:
                        text += f" Safety note: {safety}"
                    audio.announce_step(text)
                    print(f"  Reading aloud: {name}")
                else:
                    audio.announce_step(primary["class_name"].replace("_", " "))

        elif key == ord("+") or key == ord("="):
            detector.confidence = min(detector.confidence + 0.05, 0.95)
            print(f"  Confidence: {detector.confidence:.0%}")

        elif key == ord("-"):
            detector.confidence = max(detector.confidence - 0.05, 0.05)
            print(f"  Confidence: {detector.confidence:.0%}")

        elif key == ord("m"):
            # Mode toggle
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
            overlay.mode_label = f"{new_mode.upper()}"
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


def run_service_mode(config, camera_source, muted):
    """Run the guided service procedure mode."""
    from modules.procedure import ProcedureGuide

    print("=" * 56)
    print("   MECH TRAINING AI - SERVICE MODE")
    print("=" * 56)

    procedure = ProcedureGuide()
    print(f"  Vehicle : {procedure.vehicle}")
    print(f"  Service : {procedure.service_type}")
    print(f"  Steps   : {procedure.total_steps}")
    print(f"  Audio   : {'OFF' if muted else 'ON'}")
    print("=" * 56)

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
