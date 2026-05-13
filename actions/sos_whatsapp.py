import webbrowser
import urllib.parse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import app.shared_state as shared_state

SOS_MESSAGE = "SOS! I need help. This is an emergency. Please contact me immediately."


class SosWhatsApp:

    def trigger(self):
        phone   = shared_state.get_state().get("sos_phone", "").strip()
        if not phone:
            print("[SOS] No phone number set.")
            return
        encoded = urllib.parse.quote(SOS_MESSAGE)
        url     = f"https://wa.me/{phone}?text={encoded}"
        print(f"[SOS] Opening WhatsApp to {phone}")
        webbrowser.open(url)