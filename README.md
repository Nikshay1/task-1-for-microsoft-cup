### How your teammates will use this

In Task 2, they will simply do:

```python
from vision_context import get_current_object, start_vision_system

start_vision_system()

# Later, on SPACEBAR press:
context = get_current_object()
```

That’s it. Clean contract. No coupling.

### How YOU should test it (important)
Run:
```bash
python vision_context.py
```
Put a glass / phone / book in front of webcam.
You should see:
```json
{
  'object': 'glass_of_water',
  'confidence': 0.91,
  'timestamp': 1703...
}
```
Remove object → it should persist for ~5 seconds → then disappear.
