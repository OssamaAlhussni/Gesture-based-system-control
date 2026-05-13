import threading
import speech_recognition as sr
import pyttsx3
import webbrowser
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from actions.open_maps import OpenGoogleMapsHome
import app.shared_state as shared_state


class VoiceAssistant:

    def __init__(self, action_manager):
        self.action_manager = action_manager   # shared instance from live_predict

    def trigger(self):
        t = threading.Thread(target=self._listen_and_execute, daemon=True)
        t.start()

    def _speak(self, text):
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 160)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"[Voice] TTS error: {e}")

    def _listen_and_execute(self):
        recognizer = sr.Recognizer()
        self._speak("Listening")
        shared_state.update_state(voice_heard="Listening...", voice_action="--")

        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                print("[Voice] Mic open, waiting for command...")
                audio = recognizer.listen(source, timeout=6, phrase_time_limit=6)

            command = recognizer.recognize_google(audio).lower()
            print(f"[Voice] Heard: '{command}'")
            shared_state.update_state(voice_heard=command)
            self._execute_command(command)

        except sr.WaitTimeoutError:
            print("[Voice] Timeout")
            self._speak("No command heard")
            shared_state.update_state(voice_heard="--", voice_action="No command heard")
        except sr.UnknownValueError:
            self._speak("Sorry, I didn't catch that")
            shared_state.update_state(voice_heard="--", voice_action="Could not understand")
        except sr.RequestError as e:
            self._speak("Speech service unavailable")
            shared_state.update_state(voice_action="Speech service unavailable")
        except Exception as e:
            print(f"[Voice] Error: {e}")
            self._speak("Something went wrong")
            shared_state.update_state(voice_action="Error")

    def _execute_command(self, command):

        # --- Navigation ---
        if any(w in command for w in ["maps", "navigate", "directions", "location"]):
            self._speak("Opening Google Maps")
            shared_state.update_state(voice_action="Opening Google Maps")
            OpenGoogleMapsHome().trigger()

        # --- Mute / Unmute ---
        elif any(w in command for w in ["mute", "unmute", "silence", "quiet"]):
            self._speak("Toggling mute")
            shared_state.update_state(voice_action="Toggling Mute")
            self.action_manager.handle("fist")

        # --- PowerPoint + enter PPT mode ---
        elif any(w in command for w in ["powerpoint", "presentation", "slides", "slideshow"]):
            self._speak("Launching PowerPoint and entering presentation mode")
            shared_state.update_state(voice_action="Launching PowerPoint")
            self.action_manager.handle("peace")   # launches PPT and sets ppt_mode = True

        # --- Volume Up ---
        elif any(w in command for w in ["volume up", "louder", "increase volume", "turn up"]):
            self._speak("Volume up")
            shared_state.update_state(voice_action="Volume Up")
            import pyautogui
            for _ in range(5): pyautogui.press('volumeup')

        # --- Volume Down ---
        elif any(w in command for w in ["volume down", "quieter", "decrease volume", "turn down"]):
            self._speak("Volume down")
            shared_state.update_state(voice_action="Volume Down")
            import pyautogui
            for _ in range(5): pyautogui.press('volumedown')

        # --- Open Browser ---
        elif any(w in command for w in ["browser", "internet", "chrome", "search"]):
            self._speak("Opening browser")
            shared_state.update_state(voice_action="Opening Browser")
            webbrowser.open("https://www.google.com")

        # --- Unknown ---
        else:
            self._speak(f"I heard {command}, but I don't know that command yet")
            shared_state.update_state(voice_action="Unknown command")
            print(f"[Voice] Unrecognized: '{command}'")