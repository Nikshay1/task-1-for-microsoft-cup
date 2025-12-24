import cv2
import time
import threading
import requests
import os
import json

# =========================
# CONFIG
# =========================

VISION_INTERVAL = 2          # seconds between Azure calls
DECAY_SECONDS = 5            # object memory duration
CAMERA_INDEX = 0

AZURE_ENDPOINT = os.getenv("AZURE_VISION_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_VISION_KEY")

if not AZURE_ENDPOINT or not AZURE_KEY:
    raise RuntimeError("Azure Vision credentials not set in environment variables")

VISION_URL = (
    AZURE_ENDPOINT.rstrip("/")
    + "/computervision/imageanalysis:analyze"
    + "?api-version=2023-10-01"
    + "&features=tags"
)

HEADERS = {
    "Ocp-Apim-Subscription-Key": AZURE_KEY,
    "Content-Type": "application/octet-stream",
}

# =========================
# GLOBAL STATE
# =========================

latest_frame = None
last_vision_call = 0
last_object_state = None
lock = threading.Lock()

# =========================
# CAMERA LOOP
# =========================

def camera_loop():
    global latest_frame

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        with lock:
            latest_frame = frame

# =========================
# AZURE VISION CALL
# =========================

def call_azure_vision(frame):
    _, img_encoded = cv2.imencode(".jpg", frame)
    response = requests.post(
        VISION_URL,
        headers=HEADERS,
        data=img_encoded.tobytes(),
        timeout=5
    )
    response.raise_for_status()
    return response.json()

# =========================
# OBJECT EXTRACTION LOGIC
# =========================

def extract_primary_object(tags):
    """
    Select highest-confidence tag.
    """
    if not tags:
        return None, None

    tags_sorted = sorted(tags, key=lambda t: t.get("confidence", 0), reverse=True)
    best = tags_sorted[0]

    label = best.get("name", "").lower()
    confidence = best.get("confidence", 0)

    # Simple normalization logic (MVP-safe)
    if label == "glass" and any(t["name"] == "liquid" for t in tags):
        label = "glass_of_water"

    return label, confidence

# =========================
# STATE UPDATE
# =========================

def update_object_state(label, confidence):
    global last_object_state
    last_object_state = {
        "object": label,
        "confidence": round(confidence, 2),
        "timestamp": int(time.time())
    }

# =========================
# VISION LOOP
# =========================

def vision_loop():
    global last_vision_call

    while True:
        time.sleep(0.1)

        with lock:
            frame = latest_frame

        if frame is None:
            continue

        if time.time() - last_vision_call < VISION_INTERVAL:
            continue

        try:
            result = call_azure_vision(frame)
            tags = result.get("tags", [])
            label, confidence = extract_primary_object(tags)

            if label and confidence:
                update_object_state(label, confidence)

        except Exception as e:
            # Silent fail — do NOT crash MVP
            pass

        last_vision_call = time.time()

# =========================
# PUBLIC API (USED BY TASK 2)
# =========================

def get_current_object():
    """
    Returns last seen object if within decay window.
    """
    if not last_object_state:
        return None

    if time.time() - last_object_state["timestamp"] > DECAY_SECONDS:
        return None

    return last_object_state

# =========================
# START SYSTEM
# =========================

def start_vision_system():
    cam_thread = threading.Thread(target=camera_loop, daemon=True)
    vision_thread = threading.Thread(target=vision_loop, daemon=True)

    cam_thread.start()
    vision_thread.start()

# =========================
# DEBUG MODE
# =========================

if __name__ == "__main__":
    start_vision_system()
    print("Vision system running...")

    while True:
        obj = get_current_object()
        if obj:
            print(obj)
        time.sleep(1)
