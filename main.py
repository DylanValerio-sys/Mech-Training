"""
Mech Training AI - Real-time automotive service guidance with voice agent.

An AI-powered mechanic training system that detects engine components
through a camera and provides voice-interactive guidance via "Muddy",
an AI assistant that listens, speaks, and guides through service procedures.

Usage:
    python main.py --auto --cam 2             Muddy agent + detection
    python main.py --auto --cam 2 --mute      Detection only, no voice
    python main.py --service --cam 2           Service procedure mode
    python main.py                             Standard YOLO detection

Voice Commands (say "Hey Muddy" first):
    "What am I looking at?"
    "Next step" / "Previous step"
    "How do I check the radiator cap?"
    "What faults should I look for?"
    "What tools do I need?"
    Any automotive question!

Keyboard:
    +/-     Adjust confidence threshold
    Q       Quit
"""

import cv2
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.config import load_config
from modules.camera import Camera
from modules.overlay import OverlayRenderer
from modules.knowledge import KnowledgeBase


def get_camera_source(config):
    """Determine camera source from CLI args or config."""
    if "--cam" in sys.argv:
        idx = sys.argv.index("--cam")
        if idx + 1 < len(sys.argv):
            return int(sys.argv[idx + 1])
        print("ERROR: --cam requires a number (e.g. --cam 2)")
        sys.exit(1)

    if "--video" in sys.argv:
        idx = sys.argv.index("--video")
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
        print("ERROR: --video requires a file path")
        sys.exit(1)

    return config["camera_index"]


def load_api_key():
    """Load the Anthropic API key from .env file or environment variable."""
    # Check environment variable first
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key

    # Check .env file
    project_root = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(project_root, ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("ANTHROPIC_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if key:
                        return key

    return None


def run_detection_mode(config, camera_source, muted):
    """Run automotive part detection with Muddy voice agent."""
    from modules.detector import Detector
    from modules.procedure import ProcedureGuide

    mode = "automotive" if "--auto" in sys.argv else "standard"

    print("=" * 56)
    print("   MECH TRAINING AI")
    print("   Real-Time Engine Part Detection & Voice Guidance")
    print("=" * 56)

    # Load knowledge base
    print("\nLoading knowledge base...")
    knowledge = KnowledgeBase()

    # Load service procedure (for Muddy's context)
    print("Loading service procedure...")
    procedure = ProcedureGuide()

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

    # Initialize overlay
    overlay = OverlayRenderer(
        show_fps=config["show_fps"],
        show_confidence=config["show_confidence"],
        knowledge_db=knowledge,
    )

    # Initialize Muddy voice agent
    muddy = None
    current_detections = []  # Shared mutable list for Muddy to see

    if not muted:
        api_key = load_api_key()
        if api_key:
            print("\nInitializing Muddy voice agent...")
            from modules.muddy import MuddyAgent
            muddy = MuddyAgent(
                knowledge_base=knowledge,
                procedure_guide=procedure,
                api_key=api_key,
                detections_ref=current_detections,
                voice=config.get("muddy_voice", "en-AU-WilliamNeural"),
                model=config.get("muddy_model", "claude-3-5-haiku-latest"),
            )
            overlay.muddy = muddy
            if not muddy.start():
                print("  Muddy failed to start. Running without voice agent.")
                muddy = None
                overlay.muddy = None
        else:
            print("\n  No ANTHROPIC_API_KEY found.")
            print("  To enable Muddy voice agent:")
            print("  1. Get a key at https://console.anthropic.com")
            print("  2. Create a .env file in C:\\Mech Training with:")
            print("     ANTHROPIC_API_KEY=sk-ant-api03-your-key-here")
            print("\n  Running without voice agent (detection only).\n")

    print("\n" + "=" * 56)
    if muddy:
        print('  READY — Say "Hey Muddy" to ask a question')
    else:
        print("  READY — Point camera at engine components")
    print("=" * 56)
    print("  [+/-]   Adjust confidence threshold")
    print("  [Q]     Quit")
    print("=" * 56 + "\n")

    while True:
        frame = camera.get_frame()
        if frame is None:
            print("Camera disconnected.")
            break

        # Detect parts
        detections = detector.detect(frame)

        # Update shared detections list for Muddy
        current_detections.clear()
        if detections:
            current_detections.extend(detections)

        # Draw overlay
        annotated = overlay.draw(frame, detections)

        # Show confidence readout
        conf_text = f"Conf: {detector.confidence:.0%}"
        cv2.putText(annotated, conf_text, (10, annotated.shape[0] - 32),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        cv2.imshow(config["window_name"], annotated)

        # Keyboard input
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Quitting...")
            break
        elif key == ord("+") or key == ord("="):
            detector.confidence = min(detector.confidence + 0.05, 0.95)
            print(f"  Confidence: {detector.confidence:.0%}")
        elif key == ord("-"):
            detector.confidence = max(detector.confidence - 0.05, 0.05)
            print(f"  Confidence: {detector.confidence:.0%}")

    # Shutdown
    if muddy:
        muddy.shutdown()
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
    knowledge = KnowledgeBase()
    print(f"  Vehicle : {procedure.vehicle}")
    print(f"  Service : {procedure.service_type}")
    print(f"  Steps   : {procedure.total_steps}")

    camera = Camera(
        source=camera_source,
        width=config["frame_width"],
        height=config["frame_height"],
    )

    overlay = OverlayRenderer(
        show_fps=config["show_fps"],
        show_confidence=config["show_confidence"],
        knowledge_db=knowledge,
    )

    # Initialize Muddy for service mode
    muddy = None
    current_detections = []

    if not muted:
        api_key = load_api_key()
        if api_key:
            print("\nInitializing Muddy voice agent...")
            from modules.muddy import MuddyAgent
            muddy = MuddyAgent(
                knowledge_base=knowledge,
                procedure_guide=procedure,
                api_key=api_key,
                detections_ref=current_detections,
            )
            overlay.muddy = muddy
            if not muddy.start():
                muddy = None
                overlay.muddy = None

    # Optionally load detector
    detector = None
    if "--detect" in sys.argv:
        from modules.detector import Detector
        detector = Detector(
            model_path=config["model_path"],
            confidence=config["confidence_threshold"],
            mode="automotive", infer_size=640, skip_frames=4,
        )

    print("\nReady! Say 'Hey Muddy' for voice guidance, or use keyboard:")
    print("  SPACE next | BACKSPACE prev | 1-7 section | Q quit\n")

    while True:
        frame = camera.get_frame()
        if frame is None:
            break

        detections = []
        if detector:
            detections = detector.detect(frame)
            current_detections.clear()
            current_detections.extend(detections)

        annotated = overlay.draw_service(frame, procedure, detections)
        cv2.imshow(config["window_name"], annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == 32:  # SPACE
            procedure.next_step()
        elif key == 8:  # BACKSPACE
            procedure.prev_step()
        elif key in range(ord("1"), ord("8")):
            procedure.jump_to_section(key - ord("0"))

    if muddy:
        muddy.shutdown()
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
