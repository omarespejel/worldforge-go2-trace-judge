# Hackathon Runbook

## If The Robot Is Available

1. Put the robot on the floor with clear space around it.
2. Connect the operator host to the Go2 Wi-Fi.
3. If the operator copy is stale, copy the latest scripts from this repo:

```bash
scp ./scripts/go2_shared_runtime.py \
  ./scripts/go2_find_colored_target.py \
  <operator>@<venue-host>:warehouse_inspect_dimos/
```

4. Start the lightweight runtime:

```bash
cd warehouse_inspect_dimos
ROBOT_IP=192.168.12.1 nohup ./.venv/bin/python go2_shared_runtime.py > go2_shared_runtime.log 2>&1 &
echo $! > go2_shared_runtime.pid
```

5. Check MCP readiness:

```bash
./.venv/bin/python - <<'PY'
from dimos.agents.mcp.mcp_adapter import McpAdapter
a = McpAdapter("http://localhost:9990/mcp")
print("READY", a.wait_for_ready(timeout=8))
print([tool["name"] for tool in a.list_tools()])
PY
```

6. Capture calibration frames before moving:

```bash
./.venv/bin/python go2_find_colored_target.py --target red --unsafe-colors green,yellow --max-steps 1 --run-id calib-red-center
```

7. Run live only after dry-run output looks sane:

```bash
./.venv/bin/python go2_find_colored_target.py \
  --target red \
  --unsafe-colors green,yellow \
  --max-steps 8 \
  --execute \
  --balance-first \
  --celebrate \
  --run-id live-red-block-01
```

## If The Robot Is Not Available

Use the replay artifact:

```bash
cd .
make all
open artifacts/replay_run/report.html
```

Submission assets:

```text
artifacts/replay_run/worldforge_trace_replay.mp4
artifacts/replay_run/summary.json
artifacts/human_review/human_review.html
artifacts/human_review/contact_sheet.jpg
artifacts/dataset_audit/audit_report.html
dataset/go2_trace_candidates.jsonl
artifacts/ranker_smoke/model.json
artifacts/replay_run/trace/step_01/score_info.json
artifacts/replay_run/trace/step_01/candidate_scores.json
artifacts/replay_run/trace/step_01/selected_action.json
```

## 90-Second Demo Script

0-10s:
> This is a Unitree Go2 camera trace. Instead of just showing motion, we show the decision evidence.

10-30s:
> The goal is to move toward the target color. In the live block setup, the same loop can also reject calibrated unsafe color markers.

30-60s:
> Each frame becomes observation_summary. The host proposes candidate actions: turn left, turn right, move forward, or stop. The scorer picks the best candidate and logs why.

60-80s:
> These JSON artifacts are the important part: score_info, candidate_scores, selected_action, outcome. They are the dataset shape needed for future learned world-model scoring.

80-90s:
> WorldForge is the trace judge. DimOS controls the robot. The separation makes robot autonomy inspectable and swappable.

If asked about the model:
> We did not pretend to train a Go2 foundation model overnight. The included tiny ranker is
> a smoke test over the trace dataset. The serious learned-scorer path starts once live
> traces contain measured outcomes.

## Safety Boundary

- Do not claim WorldForge controls robot safety.
- Do not run `--execute` without a human operator ready to stop the robot.
- Keep movement small: the script caps joystick values and durations.
- Keep demo claims to "transparent scorer now, learned scorer later."
