# Submission Notes

## Project Name

WorldForge Go2 Trace Judge

## One-Liner

An inspectable decision layer for Unitree Go2 autonomy: real robot-view frames,
counterfactual candidate scenes, a small learned scorer, and replayable evidence.

## Final Deliverables

```text
artifacts/showcase/final_hackathon_video.mp4
artifacts/replay_mpc_arena/replay_mpc_arena.mp4
artifacts/replay_mpc_arena/decision_traces/
artifacts/dimos_mcp_sim_motion_take2/dimos_mcp_sim_motion_proof.mp4
artifacts/micro_world_demo/latest/micro_world_trace.mp4
artifacts/micro_world_demo/latest/*.json
artifacts/real_photo_edit_dataset/
hf_dataset/
hf_dataset_dimos_replay/
hf_model/
hf_model_dimos_replay_latent/
submission_bundle/
```

## Problem

Most robot demos show motion while hiding the decision process. That makes autonomy
hard to debug, trust, compare, or improve.

## Solution

Keep robot execution host-owned, but make decision scoring inspectable:

```text
observation_summary
task / goal
candidate actions
score_info
candidate_scores
selected_action
outcome_after_execution
```

The Go2 frame becomes a set of candidate futures. The scorer ranks `turn_left`,
`turn_right`, `forward_small`, and `stop_capture`. The trace records both the
selected action and rejected alternatives.

## What We Actually Built

- Curated real Go2 robot-view frames from the venue.
- Label-safe real-photo-edit counterfactual dataset from real Go2 plates and real
  cube cutouts.
- Hugging Face-ready dataset package with train/validation/test/real_seed splits.
- Small pure-NumPy micro world scorer trained on the decision traces.
- Optional micro JEPA-style latent scorer and frozen-DINOv2 hybrid scorer
  ablation.
- DimOS replay world-model dataset: 2,557 action-conditioned Go2 current/future
  frame pairs from six usable public DimOS replay sources.
- Small action-conditioned latent world model head:
  `DINOv2(current_frame) + egomotion_delta -> future DINOv2 latent`.
- Replay-MPC arena: 12 held-out replay decisions rendered with per-decision
  WorldForge-style trace files.
- DimOS MCP/MuJoCo bridge proof showing selected movement commands reaching the
  simulated Go2 runtime.
- Model honesty audit with shuffled-label and plate-holdout controls.
- One-command demo that writes annotated image, MP4, and evidence JSON.
- 58-second final video.

## Model Boundary

We did **not** train a Go2 foundation world model or V-JEPA model.

The included model is a small micro world scorer:

```text
cube geometry + unsafe risk + candidate action token -> candidate score
```

The replay model is also small and explicitly bounded:

```text
frozen DINOv2 current-frame latent + egomotion candidate
-> predicted future-frame latent
-> candidate score by similarity to the goal future latent
```

Current local metrics against transparent labels:

```text
test selection accuracy: 97.9%
test MAE: 0.0234
test R2: 0.944
random baseline: 25%
always-forward baseline: 31.2%
```

These metrics prove the scorer learned the trace scoring boundary. They do not
prove long-horizon real-world navigation success.

Additional audit:

```text
micro JEPA-style scorer: 97.9% selection accuracy, R2 0.951
DINOv2 hybrid scorer: 97.9% selection accuracy, R2 0.944
shuffled-label control: 21.8% mean selection accuracy, R2 -0.012
plate-holdout minimum selection accuracy: 92.9%
```

The DINOv2 result is deliberately framed as an ablation, not a win: frozen
full-frame DINOv2 features do not materially improve this dataset because the
cube is tiny and the labels are geometry-derived.

Replay-MPC arena:

```text
test rows scored: 383
arena decisions rendered: 12
sources shown: 6 DimOS replay sources
selected observed replay action in arena set: 12/12
```

## Why It Matters

This separates:

- candidate generation: what could the robot do?
- scoring/world model boundary: which candidate is best under the goal?
- host execution: how does the robot safely perform the selected action?
- evidence: why was this action selected?

That separation is the useful WorldForge layer.

## Runbook

```bash
make hackathon-final
```

Or inspect the final artifacts directly:

```bash
open artifacts/showcase/final_hackathon_video.mp4
open artifacts/replay_mpc_arena/replay_mpc_arena.mp4
open artifacts/micro_world_demo/latest/annotated_image.jpg
cat artifacts/micro_world_demo/latest/candidate_scores.json
```

## Future Work

- Replace transparent labels with measured outcome labels from longer Go2 runs.
- Add DINOv2/V-JEPA-style frozen visual embeddings as an optional scorer feature.
- Add Rerun visualization for candidate path overlays.
- Extend the same trace contract to SO-101 / LeRobot manipulation.
