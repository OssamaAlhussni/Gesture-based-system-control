import threading

# --- live frame shared between live_predict and Flask video feed ---
latest_frame = None
frame_lock   = threading.Lock()

# --- gesture / action state polled by dashboard ---
state = {
    "current_gesture" : "No hand",
    "confidence"      : 0.0,
    "last_action"     : "--",
    "voice_heard"     : "--",
    "voice_action"    : "--",
    "sos_phone"       : "201XXXXXXXXX",   # edit default here
    "hold_time"       : 3,
}
state_lock = threading.Lock()


def update_state(**kwargs):
    with state_lock:
        state.update(kwargs)


def get_state():
    with state_lock:
        return dict(state)