# WorldForge Go2 Trace Judge

Inspectable candidate scoring for a Unitree Go2 hackathon demo.

This repo turns real robot-view material into:

```text
real Go2 camera frames
-> label-safe real-photo-edit counterfactual dataset
-> small micro world scorer
-> WorldForge-style decision evidence
-> final judge video
```

It is not a claim of solved Go2 autonomy. The contribution is the decision/evidence
layer around robot actions: what the robot saw, which candidate moves were scored,
why one won, and what artifacts were saved for replay or later training.

## Final Artifacts

- `artifacts/showcase/final_hackathon_video.mp4`
  - 58-second final video.
- `artifacts/showcase/sim_decision_trace_video.mp4`
  - 23-second supporting clip: candidate scoring, trace JSON, and DimOS MCP
    simulation handoff.
- `artifacts/micro_world_demo/latest/micro_world_trace.mp4`
  - One-command scorer trace from a real Go2 frame.
- `artifacts/micro_world_demo/latest/`
  - `score_info.json`, `candidate_scores.json`, `selected_action.json`,
    `outcome_after_execution.json`, `run_manifest.json`, `annotated_image.jpg`.
- `artifacts/real_photo_edit_dataset/`
  - Regenerable intermediate build metadata/contact sheet for 480 accepted examples.
- `hf_dataset/`
  - Hugging Face-ready dataset package with the compact image/mask copy.
- `artifacts/micro_world_scorer/` and `hf_model/`
  - Small scorer model, eval report, model card.
- `artifacts/micro_jepa_scorer/` and `hf_model_jepa/`
  - Optional action-conditioned latent scorer.
- `artifacts/dinov2_scorer/` and `hf_model_dinov2/`
  - Frozen-DINOv2 hybrid scorer ablation.
- `artifacts/model_audit/` and `docs/MODEL_AUDIT.md`
  - Honesty audit: shuffled labels, plate holdout, and model comparison.
- `hf_dataset_dimos_replay/` and `hf_model_dimos_replay_latent/`
  - Hugging Face card/summary/provenance plus a frozen-DINOv2 residual dynamics
    model package. The full replay-derived image/jsonl dataset is published on
    Hugging Face, not duplicated in GitHub. See `docs/DIMOS_REPLAY_WORLD_MODEL.md`.
- `artifacts/replay_mpc_demo/`
  - No-robot replay-MPC demo: real DimOS Go2 replay frame, six candidate
    egomotion actions, latent future scoring, selected action, and JSON trace.
- `artifacts/replay_mpc_arena/`
  - Multi-scene Replay-MPC arena: 12 held-out DimOS Go2 replay decisions,
    per-decision WorldForge-style traces, contact sheet, and MP4.
- `artifacts/dimos_mcp_bridge_plan/`
  - Dry-run DimOS MCP command proposal generated from the replay-MPC selected
    action. This is the bridge between WorldForge scoring evidence and a
    DimOS `relative_move` / `wait` tool call.
- `submission_bundle/`
  - Copy-ready hackathon bundle.

Published HF artifacts:

- Dataset: https://huggingface.co/datasets/espejelomar/worldforge-go2-dimos-replay-world-pairs
- Model: https://huggingface.co/espejelomar/go2-dimos-replay-latent-dynamics

## Claim Boundary

Accurate wording:

- micro world scorer
- latent action scorer
- WorldForge-style decision trace evidence
- host-owned robot execution boundary

Avoid claiming:

- trained Go2 foundation world model
- trained V-JEPA
- solved autonomous Go2 navigation
- safety-certified robot controller

The model is a small pure-NumPy scorer head trained on transparent labels from real
Go2 frames and label-safe counterfactual edits.

## Core Idea

The robot should not just output an action. The useful interface is:

```text
observation_summary + task + candidate actions
-> score_info
-> candidate_scores
-> selected_action
-> outcome_after_execution
```

WorldForge does not directly drive the Go2. The host runtime owns robot connection,
velocity limits, operator supervision, and emergency stop. The trace judge owns
scoring, evidence, and replayability.

## Scripts

- `scripts/go2_shared_runtime.py`
  - Lightweight DimOS/MCP host runtime used during robot access.
- `scripts/go2_find_colored_target.py`
  - Live or single-frame colored-target scorer loop.
- `scripts/build_real_photo_edit_dataset.py`
  - Generates label-safe counterfactuals from real Go2 plates and real cube cutouts.
- `scripts/build_hf_decision_trace_dataset.py`
  - Builds the HF-ready dataset with real seed and real-photo-edit splits.
- `scripts/train_micro_world_scorer.py`
  - Trains the small NumPy scorer head.
- `scripts/train_micro_jepa_scorer.py`
  - Trains a small JEPA-style latent predictor.
- `scripts/train_dinov2_scorer.py`
  - Trains a frozen-DINOv2 hybrid scorer head.
- `scripts/build_dimos_replay_world_dataset.py`
  - Builds action-conditioned future-frame pairs from public DimOS Go2 replay DBs.
- `scripts/train_dimos_replay_latent_dynamics.py`
  - Trains a frozen-DINOv2 residual latent dynamics head on the replay pairs.
- `scripts/run_replay_mpc_demo.py`
  - Runs replay-time candidate scoring without robot access and writes MP4 plus
    WorldForge-style trace JSON.
- `scripts/run_replay_mpc_arena.py`
  - Renders multiple held-out replay decisions as an arena video and writes
    per-decision `score_info`, `candidate_scores`, `selected_action`, and
    `outcome_after_execution` traces.
- `scripts/dimos_mcp_bridge_plan.py`
  - Converts replay-MPC `selected_action.json` into a conservative DimOS MCP
    command plan. Execution is disabled by default and requires an explicit
    confirmation string plus `WORLDFORGE_DIMOS_ENABLE_EXECUTE=1`.
- `scripts/dimos_simulation_probe.py`
  - Safely inspects local DimOS replay/simulation readiness and writes the next
    no-hardware commands without starting MuJoCo or moving a robot.
- `scripts/dimos_smoke.py`
  - Runs bounded no-robot DimOS CLI/replay/simulation checks with process-group
    cleanup and artifact logging.
- `scripts/upload_hf_artifacts.py`
  - Uploads the HF-ready replay dataset/model folders using `HF_TOKEN`.
- `scripts/audit_model_honesty.py`
  - Runs shuffled-label and plate-holdout controls.
- `scripts/run_micro_world_scorer_demo.py`
  - Runs one real frame through the model and writes trace artifacts plus MP4.
- `scripts/build_final_showcase_video.py`
  - Renders the final 78-second hackathon video.
- `scripts/build_submission_bundle.py`
  - Packages the key video, dataset, model, and evidence files.

Older replay/audit scripts are still kept for provenance:

- `scripts/go2_trace_replay.py`
- `scripts/build_replay_report.py`
- `scripts/build_human_review_pack.py`
- `scripts/collect_trace_dataset.py`
- `scripts/audit_trace_dataset.py`
- `scripts/train_tiny_ranker.py`

## Rebuild Everything Important

```bash
make hackathon-final
```

Equivalent explicit commands:

```bash
python3 -m py_compile scripts/*.py
python3 scripts/build_real_photo_edit_dataset.py --count 480 --clean
python3 scripts/build_hf_decision_trace_dataset.py --clean
python3 scripts/train_micro_world_scorer.py --dataset-dir hf_dataset --output-dir artifacts/micro_world_scorer
python3 scripts/run_micro_world_scorer_demo.py \
  --image artifacts/live_ciro/direct_camera_unsafe_path.jpg \
  --model artifacts/micro_world_scorer/model.json \
  --run-id latest \
  --clean
python3 scripts/build_final_showcase_video.py
python3 scripts/run_replay_mpc_demo.py \
  --dataset-dir hf_dataset_dimos_replay \
  --model-dir hf_model_dimos_replay_latent \
  --output-dir artifacts/replay_mpc_demo \
  --clean
python3 scripts/run_replay_mpc_arena.py \
  --dataset-dir hf_dataset_dimos_replay \
  --model-dir hf_model_dimos_replay_latent \
  --output-dir artifacts/replay_mpc_arena \
  --examples 12 \
  --clean
python3 scripts/dimos_mcp_bridge_plan.py --clean
python3 scripts/build_submission_bundle.py
```

## Dataset

Build:

```bash
make real-photo-edit
make hf-dataset
```

Current split counts:

```text
train: 336
validation: 96
test: 48
real_seed: 8
```

Dataset rows include:

```text
image
mask/bbox
target_color
unsafe_colors
observation_summary
action_candidates
candidate_scores
selected_candidate_id
score_info
limitations
```

`real_photo_edit` rows are real Go2 camera plates with real cube cutouts moved to
new positions. They are useful for counterfactual scoring, not as measured robot
outcome labels.

## Model

Train:

```bash
make micro-world-scorer
```

Output:

```text
artifacts/micro_world_scorer/model.json
artifacts/micro_world_scorer/eval_report.json
artifacts/micro_world_scorer/predictions_sample.json
hf_model/
```

Current local evaluation against transparent labels:

```text
test selection accuracy: 97.9%
test MAE: 0.0234
test R2: 0.944
random baseline: 25%
always-forward baseline: 31.2%
```

These metrics show the small model learned the trace scoring boundary. They do not
prove real-world long-horizon navigation success.

## ML Stretch And Honesty Audit

Run:

```bash
make ml-stretch
```

This adds:

```text
artifacts/micro_jepa_scorer/
artifacts/dinov2_scorer/
artifacts/model_audit/
hf_model_jepa/
hf_model_dinov2/
```

Current comparison:

```text
geometry micro scorer: 97.9% selection accuracy, R2 0.9438
micro JEPA-style scorer: 97.9% selection accuracy, R2 0.9514
DINOv2 hybrid scorer: 97.9% selection accuracy, R2 0.9443
shuffled-label control: 21.8% mean selection accuracy, R2 -0.0119
```

Interpretation: the JEPA-style latent scorer is architecturally cleaner, but all
models are still distilling transparent labels. The frozen-DINOv2 ablation is a
useful negative/neutral result: visual foundation features do not materially
improve labels that were generated from geometry/risk traces.

## DimOS Replay World-Model Stretch

Run:

```bash
make dimos-replay-stretch
```

This pulls public DimOS Unitree Go2 replay archives into an ignored raw cache and
exports a Hugging Face-ready dataset. The full image/jsonl payload is ignored in
GitHub because it is published on Hugging Face:

```text
hf_dataset_dimos_replay/
  data/{train,validation,test}.jsonl
  imagefolder/{train,validation,test}/metadata.jsonl
  images/frames/
  images/pair_previews/
```

Current derived replay dataset:

```text
pairs: 2557
train/validation/test: 1791 / 383 / 383
usable source frames: 20918
usable replays: go2_short, markers_go2, go2_bigoffice, go2_hongkong_office,
  go2_slamabuse1, go2_slamabuse2
skipped replay: go2_china_office, missing usable pose fields
```

Current frozen-DINOv2 residual dynamics head:

```text
validation lift vs no-motion baseline: +0.050662 cosine
test lift vs no-motion baseline: +0.018193 cosine
validation candidate-ranking accuracy: 28.5%
test candidate-ranking accuracy: 25.8%
random among six candidates: 16.7%
```

This is deliberately framed as a small action-conditioned world-model head, not a
trained Go2 foundation model or V-JEPA model.

Rejected ablations:

```text
DINOv2-small MLP head: rejected, negative test lift despite slightly better ranking.
DINOv2-base ridge head: rejected, lower test lift than DINOv2-small ridge.
```

No-robot replay-MPC demo:

```bash
make replay-mpc-demo
```

Output:

```text
artifacts/replay_mpc_demo/replay_mpc_demo.mp4
artifacts/replay_mpc_demo/predicted_vs_actual_future.jpg
artifacts/replay_mpc_demo/score_info.json
artifacts/replay_mpc_demo/candidate_scores.json
artifacts/replay_mpc_demo/selected_action.json
artifacts/replay_mpc_demo/outcome_after_execution.json
```

Build the DimOS MCP bridge proposal from that trace:

```bash
make dimos-mcp-bridge-plan
```

Output:

```text
artifacts/dimos_mcp_bridge_plan/bridge_plan.json
artifacts/dimos_mcp_bridge_plan/selected_mcp_command.sh
artifacts/dimos_mcp_bridge_plan/run_plan.sh
```

This is dry-run only by default. To execute against a running MCP-enabled DimOS
blueprint, start `unitree-go2-agentic`, confirm `dimos mcp list-tools`, then run:

```bash
WORLDFORGE_DIMOS_ENABLE_EXECUTE=1 \
python3 scripts/dimos_mcp_bridge_plan.py \
  --execute \
  --confirm LIVE_DIMOS_MCP_EXECUTE \
  --allow-pose-derived-replay-command
```

The extra flag is required when the chosen candidate came from replay pose
deltas, because those deltas are evidence for planning, not raw live velocity
commands. Use it in simulation first.

## DimOS Simulation Roadmap

Run the safe probe first:

```bash
make dimos-sim-probe
```

Output:

```text
artifacts/dimos_simulation_probe/probe.json
artifacts/dimos_simulation_probe/next_commands.sh
```

This does not start simulation. It checks the local DimOS checkout, Go2
blueprints, simulation/replay docs, and optional CLI availability. The detailed
research and execution path is in `docs/DIMOS_SIMULATION_WORLD_MODEL_ROADMAP.md`.

After the probe, run bounded no-robot smoke checks:

```bash
make dimos-cli-smoke
make dimos-replay-smoke
make dimos-replay-smoke-bypass
make dimos-sim-smoke
```

Output:

```text
artifacts/dimos_simulation_smoke/smoke_report.json
artifacts/dimos_simulation_smoke/smoke_summary.json
```

The replay/sim smoke checks run in the foreground and are killed at timeout, so
they are safe to use while iterating on DimOS dependencies.

On macOS, DimOS may require sudo host prep before replay/simulation can run:

```bash
make dimos-macos-host-prep
```

`make dimos-replay-smoke-bypass` skips DimOS host configurators only for
structural debugging. Do not use that bypass as evidence that a real DimOS run is
properly configured.

## One-Command Demo

```bash
make micro-world-demo
```

Output:

```text
artifacts/micro_world_demo/latest/annotated_image.jpg
artifacts/micro_world_demo/latest/micro_world_trace.mp4
artifacts/micro_world_demo/latest/score_info.json
artifacts/micro_world_demo/latest/candidate_scores.json
artifacts/micro_world_demo/latest/selected_action.json
artifacts/micro_world_demo/latest/outcome_after_execution.json
```

## Final Video

```bash
make final-video
```

Output:

```text
artifacts/showcase/final_hackathon_video.mp4
```

Video arc:

1. Real Go2 material from the venue.
2. Curated robot POV frames with target and unsafe-marker examples.
3. Label-safe counterfactual dataset from real photos.
4. Micro world scorer metrics and claim boundary.
5. One-command scorer demo.
6. Evidence trail and package.

## Live Robot Notes

The live robot path was used during the hackathon, but the final build is designed
to run offline after battery/network access is gone.

Only run live commands with the Go2 on the floor, operator supervision, and emergency
stop ready. The host runtime executes robot commands; this repo does not remove the
human safety boundary.

## Pitch

> We used real Unitree Go2 robot-view data, generated label-preserving
> counterfactual scenes, trained a small action scorer, and wrapped every decision
> in a WorldForge-style evidence trail. It is not a black-box robot demo; it is an
> inspectable decision layer for embodied AI.
