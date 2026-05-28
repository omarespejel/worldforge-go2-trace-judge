# WorldForge Go2 Trace Judge Submission Bundle

## Main Demo

- `final_hackathon_video.mp4`: final judge video shell for external voiceover.
- `sim_decision_trace_video.mp4`: supporting clip showing candidate scoring,
  trace JSON, and DimOS MCP simulation handoff.
- `micro_world_trace.mp4`: one-command micro world scorer trace from a real Go2 frame.
- `micro_world_annotated.jpg`: annotated frame with candidate scores.
- `replay_mpc_demo.mp4`: no-robot replay-MPC demo from public DimOS Go2 replay data.
- `replay_mpc_arena.mp4`: multi-scene replay-MPC arena over held-out DimOS Go2
  replay rows.
- `replay_mpc_summary.jpg`: white-background summary frame showing current view,
  actual replay future, ranked candidate futures, and selected action.
- `dimos_mcp_sim_motion_proof.mp4`: simulation proof that selected MCP movement
  commands execute through DimOS/MuJoCo.

## Evidence

- `score_info.json`
- `candidate_scores.json`
- `selected_action.json`
- `outcome_after_execution.json`
- `run_manifest.json`
- `replay_mpc_score_info.json`
- `replay_mpc_candidate_scores.json`
- `replay_mpc_selected_action.json`
- `replay_mpc_outcome_after_execution.json`
- `replay_mpc_run_manifest.json`
- `replay_mpc_arena_summary.json`
- `replay_mpc_arena_contact_sheet.jpg`
- `dimos_mcp_sim_motion_report.json`

These show the WorldForge-style boundary:

```text
observation_summary + task + candidate actions
-> candidate_scores
-> selected_action
-> outcome
```

## Dataset

- `hf_dataset_data/*.jsonl`
- `hf_dataset_summary.json`
- `hf_dataset_card.md`
- `real_photo_edit_contact_sheet.jpg`
- `dimos_replay_dataset_card.md`
- `dimos_replay_dataset_summary.json`
- `dimos_replay_dataset_provenance.json`

Rows are built from curated real Go2 seed frames and label-safe real-photo-edit
counterfactuals. The labels are transparent scorer labels, not measured long-horizon
outcome labels.

The DimOS replay package is a separate action-conditioned world-model dataset:
2,557 current/future Go2 frame pairs from six usable public DimOS replay DBs.

## Model

- `micro_world_model.json`
- `micro_world_eval.json`
- `micro_world_predictions.json`
- `micro_jepa_model.json`
- `micro_jepa_eval.json`
- `dinov2_hybrid_model.json`
- `dinov2_hybrid_eval.json`
- `model_honesty_report.json`
- `model_honesty_report_md.md`
- `hf_model/README.md`
- `hf_model_jepa/README.md`
- `hf_model_dinov2/README.md`
- `hf_model_dimos_replay_latent/README.md`

The headline model is a small micro world scorer head. The bundle also includes
an optional JEPA-style latent scorer, a frozen-DINOv2 hybrid ablation, and an
honesty audit. The DimOS replay model is a small action-conditioned latent world
model head on top of frozen DINOv2 features. None of these are Go2 foundation
models or trained V-JEPA models.
