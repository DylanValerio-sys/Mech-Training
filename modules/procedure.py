import json
import os


class ProcedureGuide:
    """Manages step-by-step navigation through a service procedure.

    Loads a service procedure JSON file and tracks the mechanic's
    progress through each section and step.
    """

    def __init__(self, procedure_path=None):
        if procedure_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            procedure_path = os.path.join(project_root, "data", "service_procedure.json")

        with open(procedure_path, "r") as f:
            self.data = json.load(f)

        self.vehicle = self.data["vehicle"]
        self.service_type = self.data["service_type"]
        self.sections = self.data["sections"]
        self.notes = self.data.get("notes", [])

        # Build flat list of all steps for easy navigation
        self._steps = []
        for section in self.sections:
            for step in section["steps"]:
                self._steps.append({
                    "section_id": section["id"],
                    "section_name": section["name"],
                    **step,
                })

        self.total_steps = len(self._steps)
        self._current_index = 0

    @property
    def current_step(self):
        """Get the current step dict."""
        return self._steps[self._current_index]

    @property
    def current_index(self):
        return self._current_index

    @property
    def progress_fraction(self):
        """Progress as a float 0.0 to 1.0."""
        return self._current_index / max(self.total_steps - 1, 1)

    def next_step(self):
        """Advance to the next step. Returns True if moved, False if at end."""
        if self._current_index < self.total_steps - 1:
            self._current_index += 1
            return True
        return False

    def prev_step(self):
        """Go back to the previous step. Returns True if moved, False if at start."""
        if self._current_index > 0:
            self._current_index -= 1
            return True
        return False

    def jump_to_section(self, section_id):
        """Jump to the first step of a section by section ID (1-7)."""
        for i, step in enumerate(self._steps):
            if step["section_id"] == section_id:
                self._current_index = i
                return True
        return False

    def jump_to_check(self, check_num):
        """Jump to a specific check number (1-83)."""
        for i, step in enumerate(self._steps):
            if step["check"] == check_num:
                self._current_index = i
                return True
        return False

    def get_section_names(self):
        """Return list of (section_id, section_name) tuples."""
        return [(s["id"], s["name"]) for s in self.sections]

    def get_announcement_text(self):
        """Get text suitable for audio announcement of the current step."""
        step = self.current_step
        text = f"Step {step['check']}. {step['title']}. {step['description']}."
        if step["tools"]:
            text += f" Tools needed: {step['tools']}."
        if step["torque"]:
            text += f" Torque: {step['torque']}."
        return text

    def get_detail_text(self):
        """Get the detailed work outline for the current step."""
        return self.current_step.get("detail", "")

    def get_current_step_context(self):
        """Get a short context string about the current position."""
        step = self.current_step
        return (
            f"Currently on Step {step['check']}/{self.total_steps}: "
            f"{step['title']} (Section {step['section_id']}: {step['section_name']})"
        )

    def get_full_procedure_text(self):
        """Get the entire procedure as formatted text for an AI system prompt."""
        lines = []
        lines.append(f"Vehicle: {self.vehicle}")
        lines.append(f"Service Type: {self.service_type}")
        lines.append(f"Total Checks: {self.total_steps}")
        lines.append("")

        for section in self.sections:
            lines.append(f"SECTION {section['id']}: {section['name'].upper()}")
            for step in section["steps"]:
                lines.append(f"  Check {step['check']}: {step['title']}")
                lines.append(f"    Description: {step['description']}")
                if step.get("detail"):
                    lines.append(f"    Detail: {step['detail']}")
                if step.get("tools"):
                    lines.append(f"    Tools: {step['tools']}")
                if step.get("torque"):
                    lines.append(f"    Torque: {step['torque']}")
            lines.append("")

        if self.notes:
            lines.append("NOTES:")
            for note in self.notes:
                lines.append(f"  - {note}")

        return "\n".join(lines)
