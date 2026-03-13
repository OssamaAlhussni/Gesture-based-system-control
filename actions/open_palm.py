# actions/open_palm.py
from .base import Action
import cv2

class OpenPalmAction(Action):
    """Display gesture instructions on screen when open palm is held."""

    def __init__(self, cooldown=0.5):
        # small cooldown so it doesn’t spam, but can be repeated if needed
        super().__init__(cooldown)

    def run(self):
        # This action itself doesn’t “do” anything in system terms,
        # the overlay text will be drawn in the main loop based on label detection.
        # We can optionally log or print for debug
        print("Open palm detected: showing gesture instructions")