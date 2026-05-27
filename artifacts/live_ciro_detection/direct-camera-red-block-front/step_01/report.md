# Step 1: find the red block and avoid unsafe colored markers

Frame: `artifacts/live_ciro_detection/direct-camera-red-block-front/step_01/camera_frame.jpg`
Target found: `True` confidence `0.0464`
Unsafe color risk: `0.3302`

## Candidates
- `turn_left` score `0.5995`: turns target toward image center
- `turn_right` score `0.551`: target is not right enough
- `forward_small` score `0.7465`: safe forward progress when target is centered
- `stop_capture` score `0.4346`: only useful once target is centered and close

Selected: `forward_small`
Executed: `False`
Execution result: `dry-run: selected action not executed`
