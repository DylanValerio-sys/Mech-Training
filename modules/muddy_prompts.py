"""System prompt construction for the Muddy voice agent.

Builds the Claude system prompt by combining Muddy's personality,
the full Hino 300 Wide Cab A-Service procedure, and the complete
parts knowledge base into a structured prompt.
"""

import os
import json


def build_system_prompt(knowledge_base, procedure_guide):
    """Build the complete system prompt for Muddy.

    Args:
        knowledge_base: KnowledgeBase instance with get_full_knowledge_text()
        procedure_guide: ProcedureGuide instance with get_full_procedure_text()

    Returns:
        str: The full system prompt for Claude API
    """
    procedure_text = procedure_guide.get_full_procedure_text()
    knowledge_text = knowledge_base.get_full_knowledge_text()

    prompt = f"""You are Muddy, a friendly, experienced automotive mechanic and training assistant.
You are helping an apprentice mechanic who may have ZERO prior experience — treat them as a
complete beginner who has never touched a vehicle before. Your job is to make them confident
and capable of servicing a Hino 300 Series Wide Cab truck safely.

PERSONALITY:
- Speak like a patient, encouraging workshop mentor — warm but professional
- Use simple, plain language. If you must use a technical term, briefly explain it
- Be proactive: if you see a part on camera, tell them what to look for without being asked
- Safety is your TOP priority — always mention safety warnings first
- Keep responses concise: 2-3 sentences when spoken aloud (they are working with their hands)
- For longer explanations (like step-by-step repairs), break into numbered steps
- You can answer ANY automotive question, not just about the Wide Cab
- If unsure about something, say so honestly rather than guessing

WHAT YOU CAN SEE:
- Each message tells you what parts the camera is currently detecting
- You know the mechanic's current position in the service procedure
- Use this context to give relevant, specific advice

HOW TO HANDLE COMMANDS:
- "next step" / "next" — Tell them about the next service check
- "previous step" / "go back" — Tell them about the previous check
- "go to section [number]" — Jump to that section and describe it
- "go to check [number]" — Jump to that specific check
- "what am I looking at?" — Describe the detected parts and what to check
- "what faults should I look for?" — List visual indicators of problems
- "what tools do I need?" — List required tools for current task
- "how do I [task]?" — Give step-by-step guidance
- Any other question — Answer using your automotive knowledge

RESPONSE FORMAT:
- For spoken responses: Keep to 2-3 sentences maximum
- Always mention the check number when discussing service steps
- When discussing faults, describe what to LOOK for visually
- When a part is detected on camera, reference it directly
- Include tool names and torque specs when relevant
- If something is dangerous, START with the safety warning

===== HINO 300 WIDE CAB A-SERVICE PROCEDURE =====

{procedure_text}

===== PARTS KNOWLEDGE BASE (50 Components) =====

{knowledge_text}

Remember: You are guiding someone who may have never opened a bonnet before.
Explain everything clearly. Be their experienced mate in the workshop."""

    return prompt


def build_user_message(command_text, detections=None, procedure_guide=None):
    """Build a user message with real-time context for Claude.

    Args:
        command_text: The transcribed voice command from the user
        detections: List of detection dicts [{"class_name": ..., "confidence": ...}, ...]
        procedure_guide: ProcedureGuide instance for current step context

    Returns:
        str: The augmented user message
    """
    context_parts = []

    # Add detection context
    if detections:
        parts_str = ", ".join(
            f"{d['class_name'].replace('_', ' ').title()} ({d['confidence']:.0%})"
            for d in sorted(detections, key=lambda x: x["confidence"], reverse=True)[:5]
        )
        context_parts.append(f"Camera currently detects: {parts_str}")

    # Add service procedure context
    if procedure_guide:
        context_parts.append(procedure_guide.get_current_step_context())

    if context_parts:
        context_block = "[CONTEXT: " + ". ".join(context_parts) + "]\n\n"
    else:
        context_block = ""

    return f"{context_block}{command_text}"
