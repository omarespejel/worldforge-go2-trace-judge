# Step 47: find the green block and avoid unsafe colored markers

Frame: `artifacts/replay_run/trace/step_47/camera_frame.jpg`
Target found: `True` confidence `0.1604`
Unsafe color risk: `0.0`

## Candidates
- `turn_left` score `0.6031`: turns target toward image center
- `turn_right` score `0.5526`: target is not right enough
- `forward_small` score `0.793`: safe forward progress when target is centered
- `stop_capture` score `0.4346`: only useful once target is centered and close

Selected: `forward_small`
Executed: `False`
Execution result: `replay-only: no robot command executed`
