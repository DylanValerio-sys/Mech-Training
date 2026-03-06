"""Muddy - Conversational AI voice agent for Mech Training.

Provides hands-free voice interaction for mechanic training:
- Wake word detection ("Hey Muddy")
- Speech-to-text via Google Speech Recognition
- AI responses via Anthropic Claude API
- Natural text-to-speech via edge-tts + pygame

State machine: IDLE -> LISTENING -> THINKING -> SPEAKING -> IDLE

All voice processing runs in background threads so the video loop
is NEVER blocked.
"""

import threading
import time
import os
import re
import tempfile
import asyncio

# These imports are checked at runtime with helpful error messages
_MISSING = []

try:
    import speech_recognition as sr
except ImportError:
    sr = None
    _MISSING.append("SpeechRecognition")

try:
    import anthropic
except ImportError:
    anthropic = None
    _MISSING.append("anthropic")

try:
    import edge_tts
except ImportError:
    edge_tts = None
    _MISSING.append("edge-tts")

try:
    import pygame
except ImportError:
    pygame = None
    _MISSING.append("pygame")

from modules.muddy_prompts import build_system_prompt, build_user_message


class MuddyAgent:
    """Conversational AI voice agent for mechanic guidance.

    Runs entirely in background threads. The main video loop reads
    the agent's state and subtitle properties for overlay rendering.
    """

    # States
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    ERROR = "ERROR"

    # Wake word variations to match
    WAKE_PHRASES = ["hey muddy", "hey muddie", "hey money", "hey buddy",
                    "hey madi", "hey maddie", "a muddy", "hey modi"]

    def __init__(self, knowledge_base, procedure_guide, api_key,
                 detections_ref=None, voice="en-AU-WilliamNeural",
                 model="claude-3-5-haiku-latest", max_history=10):
        """Initialize the Muddy agent.

        Args:
            knowledge_base: KnowledgeBase instance
            procedure_guide: ProcedureGuide instance
            api_key: Anthropic API key string
            detections_ref: Mutable list shared with main loop (updated each frame)
            voice: edge-tts voice name
            model: Claude model name
            max_history: Max conversation turns to keep
        """
        self.knowledge_base = knowledge_base
        self.procedure_guide = procedure_guide
        self.detections_ref = detections_ref or []
        self.voice = voice
        self.model = model
        self.max_history = max_history

        # State (read by overlay on main thread)
        self._state = self.IDLE
        self.subtitle_question = ""
        self.subtitle_answer = ""
        self.error_msg = ""
        self._subtitle_time = 0  # When the last subtitle was set

        # Conversation history for multi-turn context
        self._history = []

        # System prompt
        self._system_prompt = build_system_prompt(knowledge_base, procedure_guide)
        prompt_tokens = len(self._system_prompt.split())
        print(f"  Muddy system prompt: ~{prompt_tokens} words")

        # Claude client
        self._client = None
        if anthropic and api_key:
            self._client = anthropic.Anthropic(api_key=api_key)
        elif not api_key:
            self.error_msg = "No API key — set ANTHROPIC_API_KEY in .env"

        # Speech recognition
        self._recognizer = None
        self._stop_listening = None
        self._listening_start = 0
        self._listening_timeout = 10  # seconds

        # Audio
        self._mixer_ready = False
        self._chime_path = None

        # Threading
        self._lock = threading.Lock()
        self._running = True

    @property
    def state(self):
        """Current agent state (thread-safe read)."""
        return self._state

    @property
    def subtitle_age(self):
        """Seconds since the last subtitle was shown."""
        if self._subtitle_time == 0:
            return 999
        return time.time() - self._subtitle_time

    def start(self):
        """Initialize microphone and begin background listening."""
        # Check dependencies
        if _MISSING:
            self.error_msg = f"Missing: pip install {' '.join(_MISSING)}"
            self._state = self.ERROR
            print(f"\n  Muddy ERROR: {self.error_msg}")
            return False

        if not self._client:
            self._state = self.ERROR
            print(f"\n  Muddy ERROR: {self.error_msg or 'Claude API not available'}")
            return False

        # Initialize pygame mixer for audio playback
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=2048)
            self._mixer_ready = True
            print("  Muddy audio playback: READY")
        except Exception as e:
            print(f"  Muddy audio playback FAILED: {e}")
            self._mixer_ready = False

        # Generate wake word chime (short beep)
        self._generate_chime()

        # Initialize speech recognition
        try:
            self._recognizer = sr.Recognizer()
            self._recognizer.energy_threshold = 300
            self._recognizer.dynamic_energy_threshold = True
            self._recognizer.pause_threshold = 1.5

            mic = sr.Microphone()
            # Adjust for ambient noise briefly
            with mic as source:
                print("  Muddy calibrating microphone (2 sec)...")
                self._recognizer.adjust_for_ambient_noise(source, duration=2)
                print(f"  Muddy energy threshold: {self._recognizer.energy_threshold:.0f}")

            # Start background listening
            self._stop_listening = self._recognizer.listen_in_background(
                mic, self._audio_callback, phrase_time_limit=8
            )
            print("  Muddy microphone: LISTENING")
            print("\n  Say 'Hey Muddy' to start a conversation!\n")
            return True

        except OSError as e:
            self.error_msg = "No microphone found"
            self._state = self.ERROR
            print(f"\n  Muddy ERROR: {self.error_msg}")
            print("  Make sure a microphone is connected.")
            print("  If using DroidCam, the phone mic may not be available as PC input.")
            return False

        except Exception as e:
            self.error_msg = str(e)
            self._state = self.ERROR
            print(f"\n  Muddy ERROR: {e}")
            return False

    def _audio_callback(self, recognizer, audio):
        """Called by SpeechRecognition's background thread for each audio chunk."""
        if not self._running:
            return

        current_state = self._state

        # --- IDLE: Listen for wake word ---
        if current_state == self.IDLE:
            try:
                text = recognizer.recognize_google(audio).lower().strip()
                if any(wake in text for wake in self.WAKE_PHRASES):
                    print("  [Muddy] Wake word detected!")
                    self._play_chime()
                    with self._lock:
                        self._state = self.LISTENING
                        self._listening_start = time.time()
                        self.subtitle_question = ""
                        self.subtitle_answer = ""
            except sr.UnknownValueError:
                pass  # No speech detected — normal
            except sr.RequestError as e:
                print(f"  [Muddy] Google STT error: {e}")
            except Exception:
                pass

        # --- LISTENING: Capture the user's command ---
        elif current_state == self.LISTENING:
            # Check timeout
            if time.time() - self._listening_start > self._listening_timeout:
                print("  [Muddy] Listening timed out")
                with self._lock:
                    self._state = self.IDLE
                return

            try:
                text = recognizer.recognize_google(audio).strip()
                if not text:
                    return

                # Ignore if it's just the wake word again
                text_lower = text.lower()
                if any(wake in text_lower for wake in self.WAKE_PHRASES):
                    # Reset timeout
                    self._listening_start = time.time()
                    return

                print(f"  [Muddy] Heard: \"{text}\"")
                self.subtitle_question = text

                with self._lock:
                    self._state = self.THINKING

                # Process command in a separate thread
                thread = threading.Thread(
                    target=self._process_command, args=(text,), daemon=True
                )
                thread.start()

            except sr.UnknownValueError:
                pass  # Silence or unclear speech — keep listening
            except sr.RequestError as e:
                print(f"  [Muddy] Google STT error: {e}")
            except Exception as e:
                print(f"  [Muddy] Listen error: {e}")

        # --- THINKING / SPEAKING: Ignore audio ---
        # (don't process input while generating/speaking a response)

    def _process_command(self, command_text):
        """Process a voice command through Claude and speak the response."""
        try:
            # Handle navigation commands locally first
            nav_context = self._handle_navigation(command_text)

            # Build the user message with real-time context
            user_msg = build_user_message(
                command_text,
                detections=list(self.detections_ref),  # Copy to avoid thread issues
                procedure_guide=self.procedure_guide,
            )

            # Add navigation context if a step change happened
            if nav_context:
                user_msg += f"\n\n[SYSTEM: {nav_context}]"

            # Add to history
            self._history.append({"role": "user", "content": user_msg})

            # Trim history
            if len(self._history) > self.max_history * 2:
                self._history = self._history[-(self.max_history * 2):]

            # Call Claude API
            print("  [Muddy] Thinking...")
            response = self._client.messages.create(
                model=self.model,
                max_tokens=250,
                system=self._system_prompt,
                messages=self._history,
            )

            answer = response.content[0].text.strip()
            print(f"  [Muddy] Response: \"{answer}\"")

            # Store response
            self._history.append({"role": "assistant", "content": answer})
            self.subtitle_answer = answer
            self._subtitle_time = time.time()

            # Speak the response
            with self._lock:
                self._state = self.SPEAKING

            self._speak(answer)

        except Exception as e:
            print(f"  [Muddy] Error processing command: {e}")
            self.subtitle_answer = "Sorry, I had trouble with that. Try again."
            self._subtitle_time = time.time()

            with self._lock:
                self._state = self.SPEAKING
            self._speak("Sorry, I had trouble thinking about that. Please try again.")

        finally:
            with self._lock:
                self._state = self.IDLE

    def _handle_navigation(self, command_text):
        """Check for navigation commands and advance the procedure.

        Returns a context string if navigation happened, or None.
        """
        text = command_text.lower().strip()

        # Next step
        if text in ("next step", "next", "go to next", "go to the next step",
                     "move on", "continue", "next check"):
            if self.procedure_guide.next_step():
                step = self.procedure_guide.current_step
                return (f"Advanced to Check {step['check']}: {step['title']}. "
                        f"{step['description']}")
            return "Already at the last step of the service procedure."

        # Previous step
        if text in ("previous step", "previous", "go back", "back",
                     "last step", "previous check"):
            if self.procedure_guide.prev_step():
                step = self.procedure_guide.current_step
                return (f"Went back to Check {step['check']}: {step['title']}. "
                        f"{step['description']}")
            return "Already at the first step of the service procedure."

        # Jump to section (e.g., "go to section 3", "section 5")
        match = re.search(r'(?:go to )?section (\d+)', text)
        if match:
            section_id = int(match.group(1))
            if self.procedure_guide.jump_to_section(section_id):
                step = self.procedure_guide.current_step
                return (f"Jumped to Section {section_id}: {step['section_name']}. "
                        f"First check is #{step['check']}: {step['title']}")
            return f"Section {section_id} not found. Valid sections are 1-7."

        # Jump to check (e.g., "go to check 45", "check 12")
        match = re.search(r'(?:go to )?check (\d+)', text)
        if match:
            check_num = int(match.group(1))
            if self.procedure_guide.jump_to_check(check_num):
                step = self.procedure_guide.current_step
                return (f"Jumped to Check {check_num}: {step['title']}. "
                        f"{step['description']}")
            return f"Check {check_num} not found. Valid checks are 1-83."

        return None  # Not a navigation command

    def _speak(self, text):
        """Generate speech using edge-tts and play with pygame."""
        if not self._mixer_ready or not edge_tts:
            print(f"  [Muddy would say]: {text}")
            time.sleep(1)  # Brief pause to simulate speech
            return

        temp_path = os.path.join(tempfile.gettempdir(), "muddy_response.mp3")

        try:
            # Generate speech audio with edge-tts (async)
            async def _generate():
                communicate = edge_tts.Communicate(text, voice=self.voice)
                await communicate.save(temp_path)

            asyncio.run(_generate())

            # Play with pygame
            if os.path.exists(temp_path):
                pygame.mixer.music.load(temp_path)
                pygame.mixer.music.play()

                # Wait for playback to finish
                while pygame.mixer.music.get_busy() and self._running:
                    time.sleep(0.1)

                pygame.mixer.music.unload()

        except Exception as e:
            print(f"  [Muddy] TTS error: {e}")

        finally:
            # Clean up temp file
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except OSError:
                pass

    def _generate_chime(self):
        """Generate a short chime sound for wake word acknowledgment."""
        if not self._mixer_ready:
            return

        try:
            import struct
            import wave

            self._chime_path = os.path.join(tempfile.gettempdir(), "muddy_chime.wav")
            sample_rate = 44100
            duration = 0.25
            freq = 880  # A5 note

            samples = int(sample_rate * duration)
            data = []
            for i in range(samples):
                t = i / sample_rate
                # Sine wave with fade out
                amplitude = 0.4 * (1.0 - t / duration)
                value = amplitude * __import__("math").sin(2 * 3.14159 * freq * t)
                data.append(int(value * 32767))

            with wave.open(self._chime_path, "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(struct.pack(f"<{len(data)}h", *data))

            print("  Muddy chime: READY")
        except Exception as e:
            print(f"  Muddy chime generation failed: {e}")
            self._chime_path = None

    def _play_chime(self):
        """Play the wake word acknowledgment chime."""
        if not self._mixer_ready or not self._chime_path:
            return
        try:
            chime = pygame.mixer.Sound(self._chime_path)
            chime.play()
        except Exception:
            pass

    def shutdown(self):
        """Clean shutdown of all agent resources."""
        print("  [Muddy] Shutting down...")
        self._running = False

        if self._stop_listening:
            try:
                self._stop_listening(wait_for_stop=False)
            except Exception:
                pass

        if self._mixer_ready:
            try:
                pygame.mixer.music.stop()
                pygame.mixer.quit()
            except Exception:
                pass

        # Clean up temp files
        for path in [self._chime_path]:
            if path:
                try:
                    os.remove(path)
                except OSError:
                    pass

        print("  [Muddy] Shutdown complete.")
