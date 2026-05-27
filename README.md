# WorldForge Go2 Trace Judge

Inspectable decision traces for a Unitree Go2. The robot remains host-controlled;
this repo records what the robot saw, which candidate moves were considered, why one
was selected, and what evidence was left behind.

This is a hackathon prototype, not a claim of solved Go2 autonomy. The useful
contribution is the trace/evidence contract around robot decisions.

## Thesis

One prompt should become an inspectable robot loop:

```text
camera observation
-> visual target/distractor detector
-> candidate robot moves
-> transparent score/evidence trail
-> optional host-owned robot execution
```

WorldForge does not own physical control or safety. The host runtime owns the Go2 connection,
velocity limits, operator supervision, and emergency stop path.

## What Is In This Repo

- `scripts/go2_shared_runtime.py`
  - Lightweight DimOS host runtime: MCP server + safe direct Go2 skills + local `humancli` router.
- `scripts/go2_find_colored_target.py`
  - Live or single-frame target loop. Uses MCP for camera/motion only when not in dry-run mode.
- `scripts/go2_trace_replay.py`
  - Offline replay builder from Go2 camera video. Produces annotated video and per-frame evidence.
- `scripts/build_human_review_pack.py`
  - Contact sheet + CSV template for human review of selected-action labels.
- `scripts/collect_trace_dataset.py`
  - Flattens per-step artifacts into candidate-level JSONL rows for later scorer training.
- `scripts/audit_trace_dataset.py`
  - Adds human/auto audit fields and writes a reviewed dataset.
- `scripts/train_tiny_ranker.py`
  - Trains a tiny scorer-distillation smoke test from the flattened trace dataset.
- `artifacts/replay_run/worldforge_trace_replay.mp4`
  - Current fallback demo video from the supplied Go2 camera recording.
- `artifacts/replay_run/report.html`
  - Browser-openable report with replay video, decision distribution, and evidence links.
- `artifacts/replay_run/trace/`
  - Per-frame `score_info.json`, `candidate_scores.json`, `selected_action.json`,
    `outcome_after_execution.json`, and `venue_input.json`.
- `data/go2_camera_recording.mp4`
  - Raw Go2 camera recording used by the offline replay.

## What Worked / What Did Not

Worked:

- Real Go2 camera frames were captured from the robot perspective.
- The prototype detected target-colored blocks and unsafe-marker colors in selected frames.
- It generated WorldForge-shaped decision traces for each scored candidate action.
- It sent bounded stand, turn, move, stop, and sit Sport commands to the Go2.

Did not fully work:

- Reliable closed-loop walking to a colored block was not completed during the robot battery window.
- Live WebRTC camera capture became unstable under repeated reconnects and rate limits.
- The included ranker is a scorer-distillation smoke test, not a trained Go2 world model.

## Replay Demo

Use this when the robot is unavailable or battery/network access is unstable.

```bash
python3 scripts/go2_trace_replay.py \
  --input-video data/go2_camera_recording.mp4 \
  --output-dir artifacts/replay_run \
  --run-id go2-camera-replay \
  --target green \
  --unsafe-colors "" \
  --fps 2 \
  --width 960
```

Output:

```text
artifacts/replay_run/worldforge_trace_replay.mp4
artifacts/replay_run/report.html
artifacts/replay_run/summary.json
artifacts/replay_run/trace/step_*/score_info.json
artifacts/replay_run/trace/step_*/candidate_scores.json
artifacts/replay_run/trace/step_*/selected_action.json
```

Build the static report:

```bash
python3 scripts/build_replay_report.py --replay-dir artifacts/replay_run
```

Or run the full offline pipeline:

```bash
make all
```

That checks scripts, rebuilds replay artifacts, writes the static report, creates the
human review pack, flattens the trace dataset, trains the tiny ranker smoke test, and
refreshes the zip package.

## Human Label Review

Use this before trusting the replay dataset:

```bash
make review
open artifacts/human_review/human_review.html
```

Outputs:

```text
artifacts/human_review/contact_sheet.jpg
artifacts/human_review/human_review.html
artifacts/human_review/human_labels_template.csv
```

Mark each sampled frame as `correct`, `wrong`, or `unsure`. This is the right
human-in-the-loop step because the current labels come from the transparent scorer,
not measured execution outcomes. For replay video, keep `UNSAFE_COLORS` empty unless
the unsafe markers were actually calibrated in that video; otherwise colored lighting
can create false unsafe detections.

Audit the dataset after review:

```bash
make audit
open artifacts/dataset_audit/audit_report.html
```

The reviewed JSONL is:

```text
artifacts/dataset_audit/go2_trace_candidates_reviewed.jsonl
```

## Trace Dataset And Tiny Ranker

Flatten replay/live traces:

```bash
make dataset
```

Output:

```text
dataset/go2_trace_candidates.jsonl
dataset/go2_trace_dataset_summary.json
```

Train the smoke-test ranker:

```bash
make ranker
```

Output:

```text
artifacts/ranker_smoke/model.json
artifacts/ranker_smoke/predictions_sample.json
```

This is not a new Go2 world model. It is a production sanity check that the evidence
contract can feed a learned scorer later. Current labels are `transparent_score_label`;
future labels should come from measured outcomes after execution.

## Live Demo

Run this only with robot on floor, operator supervision, and emergency stop ready.

On the venue/operator host:

```bash
cd warehouse_inspect_dimos

ROBOT_IP=192.168.12.1 \
nohup ./.venv/bin/python go2_shared_runtime.py > go2_shared_runtime.log 2>&1 &
echo $! > go2_shared_runtime.pid
```

Check tools:

```bash
./.venv/bin/python - <<'PY'
from dimos.agents.mcp.mcp_adapter import McpAdapter
a = McpAdapter("http://localhost:9990/mcp")
print("READY", a.wait_for_ready(timeout=8))
print([tool["name"] for tool in a.list_tools()])
PY
```

Dry run with a saved frame:

```bash
./.venv/bin/python go2_find_colored_target.py \
  --target red \
  --unsafe-colors green,yellow \
  --dry-run-frame fixtures/red_center.jpg \
  --max-steps 1
```

Live autonomous run:

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

## Pitch

> We built an inspectable robot decision layer. The Go2 sees a target, compares possible
> next moves, rejects risky or low-value actions, selects a bounded action, and leaves a
> replayable evidence trail. The same trace shape can later train or swap in a learned
> world-model scorer.

## Useful Docs

- `docs/RUNBOOK.md`: robot-time commands and fallback path.
- `docs/JUDGES_SCRIPT.md`: 90-second narration.
- `docs/TRACE_SCHEMA.md`: artifact contract and dataset shape.
- `docs/TRAINING_RESEARCH.md`: why the model-training claim stays honest.
- `docs/SUBMISSION.md`: project copy for submission.
