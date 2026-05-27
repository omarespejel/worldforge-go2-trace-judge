# WorldForge Go2 Trace Judge Submission Bundle

## Main Demo

- `demo_video.mp4`: annotated replay from real Unitree Go2 camera video.
- `demo_report.html`: browser report with selected action distribution and evidence links.

## Evidence

- `step_01_score_info.json`
- `step_01_candidate_scores.json`
- `step_01_selected_action.json`

These show the WorldForge-shaped boundary:

```text
observation_summary + task + candidate actions
-> candidate_scores
-> selected_action
-> outcome
```

## Dataset

- `trace_dataset.jsonl`: candidate-level trace rows.
- `reviewed_dataset.jsonl`: same rows with audit fields.
- `audit_report.html`: human/audit view of label quality.

Current labels are transparent-scorer labels. They are not measured outcome labels yet.

## Model Smoke Test

- `ranker_model.json`: tiny ranker that distills the transparent scorer.

This is a smoke test proving the trace can feed a model, not a new Go2 foundation world model.
