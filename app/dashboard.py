from flask import Flask, Response, jsonify, request, render_template
import cv2
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import app.shared_state as shared_state

app = Flask(__name__)


def gen_frames():
    """Yield frames from shared_state for the MJPEG stream."""
    while True:
        with shared_state.frame_lock:
            frame = shared_state.latest_frame
        if frame is None:
            time.sleep(0.033)
            continue
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.033)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/state')
def api_state():
    return jsonify(shared_state.get_state())


@app.route('/api/settings', methods=['POST'])
def api_settings():
    data = request.get_json()
    updates = {}
    if 'sos_phone' in data and str(data['sos_phone']).strip():
        updates['sos_phone'] = str(data['sos_phone']).strip()
    if 'hold_time' in data:
        try:
            ht = float(data['hold_time'])
            if 1 <= ht <= 10:
                updates['hold_time'] = ht
        except (ValueError, TypeError):
            pass
    if updates:
        shared_state.update_state(**updates)
    return jsonify({"status": "ok", "updated": updates})


def run():
    app.run(host='0.0.0.0', port=5050, debug=False, use_reloader=False)