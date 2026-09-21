FROM python:3.11-slim

WORKDIR /app

# opencv/mediapipe need these system libs even in "headless" mode
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Pre-download the anti-spoofing (Fasnet) weights into the image so runtime
# stays offline & instant, same as moderation-api does for toxic-bert.
RUN python -c "from deepface import DeepFace; import numpy as np; DeepFace.extract_faces(np.zeros((224, 224, 3), dtype='uint8'), detector_backend='opencv', enforce_detection=False, anti_spoofing=True)"

EXPOSE 9091

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9091"]
