import sys
import os

# Add the Gesture-based-system-control folder to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from actions.open_maps import OpenGoogleMapsHome

if __name__ == "__main__":
    print("[TEST] Triggering OpenGoogleMapsHome action...")
    OpenGoogleMapsHome().trigger()
    print("[TEST] Done.")
