# actions/open_fist.py
import pyautogui
from actions.base import Action

class FistAction(Action):
    """Mute/unmute system audio when fist gesture is detected."""
    
    def __init__(self):
        super().__init__(cooldown=3.0)  # 3s cooldown to prevent spamming

    def run(self):
        try:
            pyautogui.press("volumemute")  # Toggles system mute/unmute
            print("[ACTION] Toggled system mute/unmute")
        except Exception as e:
            print(f"[ERROR] Could not toggle mute: {e}")
