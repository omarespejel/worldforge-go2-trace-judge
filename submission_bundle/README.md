# WorldForge Go2 Trace Judge Submission Bundle

## Main Demo

- `final_hackathon_video.mp4`: 88-second final judge video.
- `micro_world_trace.mp4`: one-command micro world scorer trace from a real Go2 frame.
- `micro_world_annotated.jpg`: annotated frame with candidate scores.

## Evidence

- `score_info.json`
- `candidate_scores.json`
- `selected_action.json`
- `outcome_after_execution.json`
- `run_manifest.json`

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

Rows are built from curated real Go2 seed frames and label-safe real-photo-edit
counterfactuals. The labels are transparent scorer labels, not measured long-horizon
outcome labels.

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

The headline model is a small micro world scorer head. The bundle also includes
an optional JEPA-style latent scorer, a frozen-DINOv2 hybrid ablation, and an
honesty audit. None of these are Go2 foundation models or trained V-JEPA models.
