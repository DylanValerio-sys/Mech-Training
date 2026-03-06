"""Knowledge base for engine parts — fault detection, service mapping, and guidance.

Loads parts_knowledge.json and provides lookup methods to retrieve
part information, common faults, related service checks, and tools needed.
"""

import json
import os


class KnowledgeBase:
    """Provides part information, fault data, and service check references."""

    # Category display names and colors (BGR for OpenCV)
    CATEGORIES = {
        "engine":      {"label": "ENGINE",      "color": (0, 140, 255)},    # Orange
        "cooling":     {"label": "COOLING",     "color": (255, 180, 0)},    # Cyan-blue
        "electrical":  {"label": "ELECTRICAL",  "color": (0, 220, 255)},    # Yellow
        "brake":       {"label": "BRAKE",       "color": (50, 50, 255)},    # Red
        "fuel":        {"label": "FUEL",        "color": (0, 200, 100)},    # Green
        "drivetrain":  {"label": "DRIVETRAIN",  "color": (200, 100, 255)},  # Purple
        "suspension":  {"label": "SUSPENSION",  "color": (200, 200, 0)},    # Teal
        "exhaust":     {"label": "EXHAUST",     "color": (100, 100, 200)},  # Muted red
        "body":        {"label": "BODY",        "color": (180, 180, 180)},  # Grey
    }

    def __init__(self, knowledge_path=None):
        if knowledge_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            knowledge_path = os.path.join(project_root, "data", "parts_knowledge.json")

        self._data = {}
        if os.path.exists(knowledge_path):
            with open(knowledge_path, "r") as f:
                raw = json.load(f)
            # Build case-insensitive lookup
            for key, value in raw.items():
                if key.startswith("_"):
                    continue  # Skip meta keys
                self._data[key.upper()] = value
                self._data[key.lower()] = value
                self._data[key] = value
            print(f"  Knowledge base loaded: {len(raw) - 1} parts")  # -1 for _info
        else:
            print(f"  WARNING: Knowledge base not found at {knowledge_path}")

    def lookup(self, class_name):
        """Look up a part by its YOLO class name. Returns dict or None."""
        # Try exact match first, then case variations
        for key in [class_name, class_name.upper(), class_name.lower(),
                    class_name.replace("_", " ").upper()]:
            if key in self._data:
                return self._data[key]
        return None

    def get_faults(self, class_name):
        """Get the list of common faults for a part. Returns list of dicts."""
        info = self.lookup(class_name)
        if info and "faults" in info:
            return info["faults"]
        return []

    def get_service_checks(self, class_name):
        """Get related service check numbers for a part."""
        info = self.lookup(class_name)
        if info:
            return info.get("service_checks", [])
        return []

    def get_category_info(self, class_name):
        """Get category label and color for a part."""
        info = self.lookup(class_name)
        if info:
            cat = info.get("category", "body")
            return self.CATEGORIES.get(cat, self.CATEGORIES["body"])
        return self.CATEGORIES["body"]

    def get_safety(self, class_name):
        """Get safety warnings for a part."""
        info = self.lookup(class_name)
        if info:
            return info.get("safety", "")
        return ""

    def get_difficulty(self, class_name):
        """Get service difficulty level for a part."""
        info = self.lookup(class_name)
        if info:
            return info.get("difficulty", "unknown")
        return "unknown"

    def get_announcement_text(self, class_name):
        """Get text suitable for audio announcement when a part is detected."""
        info = self.lookup(class_name)
        if not info:
            return class_name.replace("_", " ").title()

        name = info.get("name", class_name)
        desc = info.get("description", "")
        faults = info.get("faults", [])

        text = f"{name}. {desc}."
        if faults:
            top_fault = faults[0]
            text += f" Check for: {top_fault['name']}."
        return text

    def get_short_tips(self, class_name, max_tips=3):
        """Get short inspection tips for display on overlay."""
        info = self.lookup(class_name)
        if not info:
            return []
        faults = info.get("faults", [])
        tips = []
        for fault in faults[:max_tips]:
            tips.append(f"Check: {fault['name']} — {fault['visual']}")
        return tips

    def get_full_knowledge_text(self):
        """Get the entire knowledge base as formatted text for an AI system prompt."""
        lines = []
        seen = set()
        for key, info in self._data.items():
            # Only process each part once (we have multiple case variants)
            name = info.get("name", key)
            if name in seen:
                continue
            seen.add(name)

            category = info.get("category", "unknown").upper()
            lines.append(f"PART: {name} [{category}]")
            if info.get("description"):
                lines.append(f"  Description: {info['description']}")
            if info.get("location"):
                lines.append(f"  Location: {info['location']}")
            faults = info.get("faults", [])
            if faults:
                lines.append("  Common Faults:")
                for f in faults:
                    sev = f.get("severity", "medium").upper()
                    lines.append(f"    - {f['name']} ({sev}): {f['visual']}")
                    lines.append(f"      Action: {f['action']}")
            checks = info.get("service_checks", [])
            if checks:
                lines.append(f"  Service Checks: {', '.join(f'#{c}' for c in checks)}")
            if info.get("safety"):
                lines.append(f"  Safety: {info['safety']}")
            if info.get("tools"):
                lines.append(f"  Tools: {', '.join(info['tools'])}")
            if info.get("difficulty"):
                lines.append(f"  Difficulty: {info['difficulty']}")
            lines.append("")

        return "\n".join(lines)
