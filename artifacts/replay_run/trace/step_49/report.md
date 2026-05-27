# Step 49: find the green block and avoid unsafe colored markers

Frame: `artifacts/replay_run/trace/step_49/camera_frame.jpg`
Target found: `True` confidence `0.3373`
Unsafe color risk: `0.0`

## Candidates
- `turn_left` score `0.9115`: turns target toward image center
- `turn_right` score `0.6875`: target is not right enough
- `forward_small` score `0.4288`: safe forward progress when target is centered
- `stop_capture` score `0.4346`: only useful once target is centered and close

Selected: `turn_left`
Executed: `False`
Execution result: `replay-only: no robot command executed`
