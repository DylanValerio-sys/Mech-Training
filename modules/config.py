import json
import os

DEFAULT_CONFIG = {
    "camera_index": 0,
    "frame_width": 1280,
    "frame_height": 720,
    "confidence_threshold": 0.5,
    "model_path": "yolov8n.pt",
    "window_name": "Mech Training AI",
    "show_fps": True,
    "show_confidence": True,
}


def load_config(config_path="config.json"):
    """Load configuration from JSON file, falling back to defaults."""
    config = DEFAULT_CONFIG.copy()

    # Resolve path relative to the project root
    if not os.path.isabs(config_path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, config_path)

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            user_config = json.load(f)
        config.update(user_config)

    return config
