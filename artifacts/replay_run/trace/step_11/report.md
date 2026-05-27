# Step 11: find the green block and avoid unsafe colored markers

Frame: `artifacts/replay_run/trace/step_11/camera_frame.jpg`
Target found: `True` confidence `0.094`
Unsafe color risk: `0.0`

## Candidates
- `turn_left` score `0.751`: turns target toward image center
- `turn_right` score `0.6185`: target is not right enough
- `forward_small` score `0.51`: safe forward progress when target is centered
- `stop_capture` score `0.4346`: only useful once target is centered and close

Selected: `turn_left`
Executed: `False`
Execution result: `replay-only: no robot command executed`
