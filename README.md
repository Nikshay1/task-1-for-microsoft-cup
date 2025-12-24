# Gaze Vision – Module Hand-off

This module provides **gaze-based object selection** for NeuroBridge.

Using a dual-camera setup:
- an **eye-facing camera** to track pupil movement
- a **world-facing camera** to observe the environment

the system determines **which real-world object the user is looking at**, even if it is not directly in front of their head.

This module outputs a clean, structured object context for downstream intent
and language-generation components.

---

## Requirements & Assumptions

- Python 3.10+
- Two connected cameras:
  - 1 × eye-facing camera
  - 1 × world/environment-facing camera (wide FOV recommended)
- Azure Computer Vision resource
- Linux / macOS

Camera indices can be configured inside `gaze_vision.py`:
```python
WORLD_CAM_INDEX = 0
EYE_CAM_INDEX = 1
```


## Setup (using uv)
```python
uv venv .venv
source .venv/bin/activate
uv sync
``` 

## Azure Configuration

This module uses Azure Computer Vision (Image Analysis).

Set the following environment variables before running:
```bash
export AZURE_VISION_ENDPOINT="https://<your-resource>.cognitiveservices.azure.com/"
export AZURE_VISION_KEY="<your-key>"
```

## System Start & Calibration

Calibration is mandatory and must be performed once per session.

```python
from gaze_vision import start_system, calibration_routine

start_system()
calibration_routine()
```

Calibration flow
- A target appears in the world-camera view
- The user looks at the target
- Operator/caregiver presses SPACE to capture a sample
- Repeat until calibration completes
- This maps eye pupil coordinates → world image coordinates.


## Runtime Usage (Hand-off)
### On user trigger (Task 2)

```python 
from gaze_vision import get_current_object

obj = get_current_object()
if obj:
    print(obj)
```

Example output:
```json
{
  "object": "glass_of_water",
  "confidence": 0.92,
  "timestamp": 1703445566
}
```
get_current_object() returns the last valid gaze-selected object
within a short decay window (to prevent flicker).

## Integration Note (Task 2)

The trigger system (blink, button, EEG, etc.) must call
get_current_object() at trigger time to lock the visual context.

Downstream modules should treat this object as immutable input.