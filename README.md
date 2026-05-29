# Gesture-Based System Control

Real-time hand gesture recognition that controls your computer — built with MediaPipe, scikit-learn (KNN), and Flask.

## Gestures

| Gesture | Action |
|---|---|
| ☝️ Index Point | Open Google Maps |
| ✊ Fist | Mute / Unmute |
| 🖐 Open Palm | Send SOS via WhatsApp |
| ✌️ Peace | Launch PowerPoint |
| 👍 Thumbs Up | Voice Assistant |

Once in **PPT mode**: Index Point → Next Slide, Thumbs Up → Previous Slide, Open Palm → Exit.

---

## Running with Docker

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed
- A webcam connected

### Linux / macOS

```bash
git clone https://github.com/OssamaAlhussni/Gesture-based-system-control
cd Gesture-based-system-control
xhost +local:docker
docker compose up --build
```

The gesture window will open on your desktop and the dashboard will be at **http://localhost:5050**.




