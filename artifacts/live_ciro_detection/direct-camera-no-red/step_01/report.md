# Step 1: find the red block and avoid unsafe colored markers

Frame: `artifacts/live_ciro_detection/direct-camera-no-red/step_01/camera_frame.jpg`
Target found: `True` confidence `0.0506`
Unsafe color risk: `0.4121`

## Candidates
- `turn_left` score `0.6754`: target is not left enough
- `turn_right` score `0.8788`: turns target toward image center
- `forward_small` score `0.362`: forward progress penalized by unsafe color near path
- `stop_capture` score `0.4346`: only useful once target is centered and close

Selected: `turn_right`
Executed: `False`
Execution result: `dry-run: selected action not executed`
