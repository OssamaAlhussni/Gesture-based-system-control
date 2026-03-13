# actions/open_google_maps.py
import webbrowser
from actions.base import Action

class OpenGoogleMapsHome(Action):
    def __init__(self):
        super().__init__(cooldown=5.0)

    def run(self):
        url = (
            "https://www.google.com/maps/dir/"
            "?api=1"
            #"&destination=Home"
            "&destination=Dubai mall"
            "&travelmode=driving"
        )
        webbrowser.open(url)
        print("[ACTION] Opening Google Maps → Home")

