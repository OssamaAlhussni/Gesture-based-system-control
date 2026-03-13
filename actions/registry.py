# actions/registry.py
import subprocess
import pyautogui
import os
from actions.open_maps import OpenGoogleMapsHome
from actions.fist import FistAction
from actions.open_palm import OpenPalmAction
import time

# ---------- PPT dual-mode setup ----------
ppt_path = r"C:\Users\ossam\Desktop\Desktop\University\Sem 7\Grad project\final_ppt_gp.pptx"

class PeaceAction:
    """Open PowerPoint presentation and enter PPT mode."""
    def __init__(self, manager=None):
        self.manager = manager  # reference to ActionManager to toggle ppt_mode

    def trigger(self):
        if self.manager and not self.manager.ppt_mode:
            subprocess.Popen(['start', 'powerpnt', ppt_path], shell=True)
            self.manager.ppt_mode = True

# ---------- Action Manager ----------
class ActionManager:
    """Maps gesture labels to actions and manages PPT mode."""
    def __init__(self):
        self.ppt_mode = False
        self.actions = {
            "index_point": OpenGoogleMapsHome(),
            "fist": FistAction(),
            "open_palm": OpenPalmAction(),
            "peace": PeaceAction(manager=self),
            "thumbs_up": None,
        }

    def handle(self, label):
        # PPT mode overrides gestures
        if self.ppt_mode:
            if label == "index_point":
                pyautogui.press('right')  # next slide
                return
            elif label == "thumbs_up":
                pyautogui.press('left')   # previous slide
                return
            elif label == "open_palm":
                # exit ppt mode

                pyautogui.hotkey('alt', 'f4')     # exit full-screen PPT
                self.ppt_mode = False
                return "reset_lock"

        # Normal action mapping
        action = self.actions.get(label)
        if action:
            action.trigger()