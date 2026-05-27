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
  - 78-second final video.
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
- `submission_bundle/`
  - Copy-ready hackathon bundle.

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
