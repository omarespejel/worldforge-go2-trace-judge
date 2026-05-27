# Step 1: find the red block and avoid unsafe colored markers

Frame: `artifacts/live_ciro_detection/direct-camera-red-block-left/step_01/camera_frame.jpg`
Target found: `True` confidence `0.0422`
Unsafe color risk: `0.404`

## Candidates
- `turn_left` score `0.9115`: turns target toward image center
- `turn_right` score `0.6875`: target is not right enough
- `forward_small` score `0.3632`: forward progress penalized by unsafe color near path
- `stop_capture` score `0.4346`: only useful once target is centered and close

Selected: `turn_left`
Executed: `False`
Execution result: `dry-run: selected action not executed`
