FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libportaudio2 \
        libx11-6 \
        libxext6 \
        libxrender1 \
        xdotool \
        xdg-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN sed -i \
        -e 's/^opencv-python==/opencv-python-headless==/g' \
        -e 's/^opencv-contrib-python==/opencv-contrib-python-headless==/g' \
        requirements.txt \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5050

CMD ["python", "app/app.py"]