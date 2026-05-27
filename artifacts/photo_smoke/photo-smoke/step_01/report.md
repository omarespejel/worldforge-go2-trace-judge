# Step 1: find the green block and avoid unsafe colored markers

Frame: `artifacts/photo_smoke/photo-smoke/step_01/camera_frame.jpg`
Target found: `True` confidence `0.3362`
Unsafe color risk: `0.6381`

## Candidates
- `turn_left` score `0.5914`: turns target toward image center
- `turn_right` score `0.5474`: target is not right enough
- `forward_small` score `0.7021`: forward progress penalized by unsafe color near path
- `stop_capture` score `0.4346`: only useful once target is centered and close

Selected: `forward_small`
Executed: `False`
Execution result: `dry-run: selected action not executed`
