"""
gaze_vision.py

Two-camera gaze->object module for NeuroBridge MVP (Task 1 extension).

Features:
- Calibration routine: map eye-camera pupil coords -> world-camera image coords using an affine fit.
- Runtime: detect pupil in eye camera, map to world pixel, call Azure Vision Objects API
  periodically, pick the bounding-box under gaze (or nearest box), expose get_current_object().
- Decay buffer (5s) for stability.

Requirements:
pip install opencv-python numpy requests

Set these env vars:
AZURE_VISION_ENDPOINT, AZURE_VISION_KEY

Usage:
python gaze_vision.py
"""

import cv2
import time
import threading
import numpy as np
import requests
import os
import math

# ---------------------------
# CONFIG
# ---------------------------
WORLD_CAM_INDEX = 0   # change if your world camera is not 0
EYE_CAM_INDEX = 1     # change if your eye camera is not 1
CAL_SAMPLES = 9       # number of calibration points (3x3 grid)
VISION_INTERVAL = 2.0  # seconds between Azure object detections
DECAY_SECONDS = 5.0
DEBUG = True

AZURE_ENDPOINT = os.getenv("AZURE_VISION_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_VISION_KEY")
if not AZURE_ENDPOINT or not AZURE_KEY:
    raise RuntimeError("Set AZURE_VISION_ENDPOINT and AZURE_VISION_KEY environment variables")

# ensure objects feature in query
VISION_URL = (
    AZURE_ENDPOINT.rstrip("/")
    + "/computervision/imageanalysis:analyze"
    + "?api-version=2023-10-01"
    + "&features=objects,tags"
)

HEADERS = {
    "Ocp-Apim-Subscription-Key": AZURE_KEY,
    "Content-Type": "application/octet-stream",
}

# ---------------------------
# GLOBAL STATE (thread-safe)
# ---------------------------
latest_world_frame = None
latest_eye_frame = None
lock = threading.Lock()

last_object_state = None
last_vision_call = 0.0
last_azure_response = None

affine_M = None  # 2x3 mapping from [eye_x, eye_y, 1] -> [world_x, world_y]

# ---------------------------
# UTIL: pupil detection in eye frame
# ---------------------------
def detect_pupil_center(eye_frame):
    """
    Simple dark-blob method to find pupil center.
    Input: eye_frame (BGR)
    Returns: (x, y) pixel coordinate in eye_frame or None
    """
    gray = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # equalize and blur
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # adaptive threshold or Otsu to find dark pupil
    # invert to make pupil bright for contour detection
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # choose largest contour with reasonable size
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (h*w)*0.001:  # ignore tiny
            continue
        # centroid
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        # sanity: pupil near center of eye frame
        return (cx, cy)

    return None

# ---------------------------
# AFFINE FIT (calibration)
# ---------------------------
def fit_affine(eye_points, world_points):
    """
    eye_points: list of (x_e, y_e)
    world_points: list of (x_w, y_w)
    Fit world = M * [eye_x, eye_y, 1]  where M is 2x3
    Returns M (2x3) numpy array
    """
    E = np.hstack([np.array(eye_points, dtype=np.float32), np.ones((len(eye_points),1), dtype=np.float32)])  # Nx3
    W = np.array(world_points, dtype=np.float32)  # Nx2

    # Solve for M using least squares: E @ M.T = W  => M.T = pinv(E) @ W
    pinvE = np.linalg.pinv(E)                            # 3xN -> 3xN?
    M_t = pinvE.dot(W)                                  # 3x2
    M = M_t.T                                           # 2x3
    return M

def apply_affine(M, eye_xy):
    """
    M: 2x3
    eye_xy: (x, y)
    returns world_xy (x_w, y_w) floats
    """
    v = np.array([eye_xy[0], eye_xy[1], 1.0], dtype=np.float32)
    out = M.dot(v)  # 2-vector
    return (float(out[0]), float(out[1]))

# ---------------------------
# AZURE OBJECT DETECTION
# ---------------------------
def call_azure_objects(frame):
    """
    Sends a JPEG-encoded frame to Azure and returns parsed objects.
    Returns: list of objects: [{name, confidence, rect: {x,y,w,h}}]
    """
    _, jpg = cv2.imencode(".jpg", frame)
    b = jpg.tobytes()
    try:
        resp = requests.post(VISION_URL, headers=HEADERS, data=b, timeout=6)
        resp.raise_for_status()
        j = resp.json()
    except Exception as e:
        # network/timeout -> return None
        return None

    objs = []
    # Azure returns 'objects' list with 'rectangle' and 'object' keys
    # defensive parsing:
    for o in j.get("objects", []):
        rect = o.get("rectangle", {})
        name = o.get("object") or o.get("name") or o.get("tag") or "unknown"
        conf = float(o.get("confidence", 0.0))
        # rectangle fields may differ in keys; normalize
        x = int(rect.get("x", rect.get("left", 0)))
        y = int(rect.get("y", rect.get("top", 0)))
        w = int(rect.get("w", rect.get("width", 0)))
        h = int(rect.get("h", rect.get("height", 0)))
        objs.append({"name": name, "confidence": conf, "rect": {"x":x, "y":y, "w":w, "h":h}})
    return objs

# ---------------------------
# OBJECT SELECTION FROM GAZE
# ---------------------------
def choose_object_from_gaze(objects, gaze_xy):
    """
    objects: list of objects from Azure (with rect)
    gaze_xy: (xw, yw) world pixel
    returns: (name, confidence) or (None, None)
    """
    if not objects or gaze_xy is None:
        return None, None

    xg, yg = gaze_xy
    # check containment
    for obj in objects:
        r = obj["rect"]
        if r["x"] <= xg <= r["x"]+r["w"] and r["y"] <= yg <= r["y"]+r["h"]:
            return obj["name"], obj["confidence"]

    # else, choose nearest bbox center (distance)
    best = None
    best_dist = None
    for obj in objects:
        r = obj["rect"]
        cx = r["x"] + r["w"]/2.0
        cy = r["y"] + r["h"]/2.0
        d = math.hypot(cx - xg, cy - yg)
        if best is None or d < best_dist:
            best = obj
            best_dist = d
    if best:
        return best["name"], best["confidence"]
    return None, None

# ---------------------------
# CAMERA THREADS
# ---------------------------
def world_camera_loop(cam_index=WORLD_CAM_INDEX):
    global latest_world_frame
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open world camera index {cam_index}")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        with lock:
            latest_world_frame = frame

def eye_camera_loop(cam_index=EYE_CAM_INDEX):
    global latest_eye_frame
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open eye camera index {cam_index}")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        with lock:
            latest_eye_frame = frame

# ---------------------------
# CALIBRATION ROUTINE
# ---------------------------
def calibration_routine():
    """
    Shows a simple 3x3 grid of targets on the live world feed and records
    eye pupil centers when user presses SPACE to accept each target. After
    collecting CAL_SAMPLES samples, fits the affine transform.
    """
    global affine_M

    print("CALIBRATION: We will collect samples. You (or caregiver) should look at each target and press SPACE to capture.")
    print("Press ESC to abort calibration.")

    points_eye = []
    points_world = []

    grid = []
    # will generate a 3x3 grid inside the world frame dynamically
    # Wait until a world frame is available
    while True:
        with lock:
            wframe = latest_world_frame.copy() if latest_world_frame is not None else None
        if wframe is not None:
            break
        time.sleep(0.1)

    h, w = wframe.shape[:2]
    margin = 0.15
    xs = np.linspace(int(w*margin), int(w*(1-margin)), 3)
    ys = np.linspace(int(h*margin), int(h*(1-margin)), 3)
    for yy in ys:
        for xx in xs:
            grid.append( (int(xx), int(yy)) )

    # select CAL_SAMPLES points from grid (first N)
    chosen = grid[:CAL_SAMPLES]

    idx = 0
    while idx < len(chosen):
        target = chosen[idx]
        tx, ty = target

        # show live world feed with target marker
        while True:
            with lock:
                wframe = latest_world_frame.copy() if latest_world_frame is not None else None
                eframe = latest_eye_frame.copy() if latest_eye_frame is not None else None

            if wframe is None or eframe is None:
                time.sleep(0.05)
                continue

            vis = wframe.copy()
            # draw circle at target
            cv2.circle(vis, (tx,ty), 20, (0,0,255), thickness=3)
            cv2.putText(vis, f"Target {idx+1}/{len(chosen)} - Press SPACE when user is looking", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            # show eye frame in small window
            eh, ew = eframe.shape[:2]
            small_eye = cv2.resize(eframe, (int(ew*0.6), int(eh*0.6)))
            # compose for debugging
            vis_small = cv2.hconcat([vis, cv2.copyMakeBorder(small_eye, 0, vis.shape[0]-small_eye.shape[0], 0,0, cv2.BORDER_CONSTANT, value=[0,0,0])])
            cv2.imshow("Calibration (world + eye)", vis_small)
            key = cv2.waitKey(30) & 0xFF
            if key == 27:  # ESC abort
                cv2.destroyAllWindows()
                raise RuntimeError("Calibration aborted")
            if key == 32:  # SPACE capture
                # detect pupil
                pupil = detect_pupil_center(eframe)
                if pupil is None:
                    print("Pupil not detected; try again.")
                    continue
                points_eye.append(pupil)
                points_world.append((tx,ty))
                print(f"Captured sample {idx+1}: eye={pupil}, world={(tx,ty)}")
                idx += 1
                break

    cv2.destroyAllWindows()
    # Fit affine
    M = fit_affine(points_eye, points_world)
    affine_M = M
    print("Calibration complete. Affine matrix:")
    print(M)
    return M

# ---------------------------
# VISION & GAZE LOOP (integration)
# ---------------------------
def integration_loop():
    global last_vision_call, last_azure_response, last_object_state

    while True:
        time.sleep(0.05)
        with lock:
            wframe = latest_world_frame.copy() if latest_world_frame is not None else None
            eframe = latest_eye_frame.copy() if latest_eye_frame is not None else None

        if wframe is None or eframe is None or affine_M is None:
            continue

        # detect pupil
        pupil = detect_pupil_center(eframe)
        gaze_world = None
        if pupil is not None:
            gaze_world = apply_affine(affine_M, pupil)  # float coordinates

        # call Azure periodically
        if time.time() - last_vision_call > VISION_INTERVAL:
            objs = call_azure_objects(wframe)
            last_azure_response = objs
            last_vision_call = time.time()
        else:
            objs = last_azure_response

        name, conf = choose_object_from_gaze(objs, gaze_world) if objs is not None else (None, None)

        if name and conf:
            last_object_state = {"object": name, "confidence": round(float(conf), 2), "timestamp": int(time.time())}
        # debug draw
        if DEBUG:
            dbg = wframe.copy()
            # draw boxes
            if objs:
                for o in objs:
                    r = o["rect"]
                    cv2.rectangle(dbg, (r["x"], r["y"]), (r["x"]+r["w"], r["y"]+r["h"]), (0,255,0), 2)
                    cv2.putText(dbg, f'{o["name"]} {o["confidence"]:.2f}', (r["x"], r["y"]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            # draw gaze point
            if gaze_world:
                gx, gy = int(gaze_world[0]), int(gaze_world[1])
                cv2.circle(dbg, (gx, gy), 10, (0,0,255), -1)
            # overlay text of current object
            if last_object_state:
                cv2.putText(dbg, f'Looked: {last_object_state["object"]} ({last_object_state["confidence"]})', (10, dbg.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.imshow("Gaze->World Debug", dbg)
            if cv2.waitKey(1) & 0xFF == 27:
                break

# ---------------------------
# PUBLIC API
# ---------------------------
def get_current_object():
    global last_object_state
    if last_object_state is None:
        return None
    if time.time() - last_object_state["timestamp"] > DECAY_SECONDS:
        return None
    return last_object_state

def start_system(world_cam=WORLD_CAM_INDEX, eye_cam=EYE_CAM_INDEX):
    # spawn camera threads and integration thread
    t_world = threading.Thread(target=world_camera_loop, args=(world_cam,), daemon=True)
    t_eye = threading.Thread(target=eye_camera_loop, args=(eye_cam,), daemon=True)
    t_integ = threading.Thread(target=integration_loop, daemon=True)

    t_world.start()
    t_eye.start()
    t_integ.start()
    return t_world, t_eye, t_integ

# ---------------------------
# MAIN (script mode)
# ---------------------------
if __name__ == "__main__":
    print("Starting cameras...")
    t1, t2, t3 = start_system()
    # wait for a bit to let cameras warm up
    time.sleep(1.0)
    try:
        print("Run calibration now.")
        calibration_routine()
    except Exception as e:
        print("Calibration failed:", e)
        print("Exiting.")
        raise

    print("Calibration done. Running gaze->object loop. Press ESC in debug window to exit.")
    # Keep main thread alive while integration loop and camera threads run
    while True:
        try:
            time.sleep(1.0)
            obj = get_current_object()
            if obj:
                print("Current looked-object:", obj)
        except KeyboardInterrupt:
            print("Stopping.")
            break

    cv2.destroyAllWindows()
