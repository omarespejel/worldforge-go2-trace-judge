from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _copy(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise RuntimeError(f"Missing source artifact: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.is_dir():
        raise RuntimeError(f"Missing source directory: {src}")
    shutil.copytree(src, dst, dirs_exist_ok=True)


def run(args: argparse.Namespace) -> int:
    root = Path.cwd()
    out = Path(args.output_dir).expanduser()
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "final_hackathon_video": root / "artifacts/showcase/final_hackathon_video.mp4",
        "submission_notes": root / "docs/SUBMISSION.md",
        "micro_world_trace": root / "artifacts/micro_world_demo/latest/micro_world_trace.mp4",
        "micro_world_annotated": root / "artifacts/micro_world_demo/latest/annotated_image.jpg",
        "real_photo_edit_contact_sheet": root / "artifacts/real_photo_edit_dataset/contact_sheet.jpg",
        "media_review_contact_sheet": root / "artifacts/media_review/batch_review_kept_after_deletes.jpg",
        "hf_dataset_summary": root / "hf_dataset/dataset_summary.json",
        "hf_dataset_card": root / "hf_dataset/README.md",
        "micro_world_model": root / "artifacts/micro_world_scorer/model.json",
        "micro_world_eval": root / "artifacts/micro_world_scorer/eval_report.json",
        "micro_world_predictions": root / "artifacts/micro_world_scorer/predictions_sample.json",
        "micro_jepa_model": root / "artifacts/micro_jepa_scorer/model.json",
        "micro_jepa_eval": root / "artifacts/micro_jepa_scorer/eval_report.json",
        "dinov2_hybrid_model": root / "artifacts/dinov2_scorer/model.json",
        "dinov2_hybrid_eval": root / "artifacts/dinov2_scorer/eval_report.json",
        "model_honesty_report": root / "artifacts/model_audit/honesty_report.json",
        "model_honesty_report_md": root / "artifacts/model_audit/honesty_report.md",
        "dimos_replay_dataset_card": root / "hf_dataset_dimos_replay/README.md",
        "dimos_replay_dataset_summary": root / "hf_dataset_dimos_replay/dataset_summary.json",
        "dimos_replay_dataset_provenance": root / "hf_dataset_dimos_replay/provenance.json",
        "replay_mpc_demo": root / "artifacts/replay_mpc_demo/replay_mpc_demo.mp4",
        "replay_mpc_summary": root / "artifacts/replay_mpc_demo/predicted_vs_actual_future.jpg",
        "replay_mpc_score_info": root / "artifacts/replay_mpc_demo/score_info.json",
        "replay_mpc_candidate_scores": root / "artifacts/replay_mpc_demo/candidate_scores.json",
        "replay_mpc_selected_action": root / "artifacts/replay_mpc_demo/selected_action.json",
        "replay_mpc_outcome_after_execution": root / "artifacts/replay_mpc_demo/outcome_after_execution.json",
        "replay_mpc_run_manifest": root / "artifacts/replay_mpc_demo/run_manifest.json",
        "replay_mpc_arena": root / "artifacts/replay_mpc_arena/replay_mpc_arena.mp4",
        "replay_mpc_arena_contact_sheet": root / "artifacts/replay_mpc_arena/arena_contact_sheet.jpg",
        "replay_mpc_arena_summary": root / "artifacts/replay_mpc_arena/arena_summary.json",
        "dimos_mcp_sim_motion_proof": root / "artifacts/dimos_mcp_sim_motion_take2/dimos_mcp_sim_motion_proof.mp4",
        "dimos_mcp_sim_motion_report": root / "artifacts/dimos_mcp_sim_motion_take2/motion_take2_report.json",
        "score_info": root / "artifacts/micro_world_demo/latest/score_info.json",
        "candidate_scores": root / "artifacts/micro_world_demo/latest/candidate_scores.json",
        "selected_action": root / "artifacts/micro_world_demo/latest/selected_action.json",
        "outcome_after_execution": root / "artifacts/micro_world_demo/latest/outcome_after_execution.json",
        "run_manifest": root / "artifacts/micro_world_demo/latest/run_manifest.json",
    }
    for name, src in artifacts.items():
        _copy(src, out / f"{name}{src.suffix}")

    data_out = out / "hf_dataset_data"
    data_out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test", "real_seed"):
        _copy(root / "hf_dataset/data" / f"{split}.jsonl", data_out / f"{split}.jsonl")
    _copy_tree(root / "hf_model", out / "hf_model")
    _copy_tree(root / "hf_model_jepa", out / "hf_model_jepa")
    _copy_tree(root / "hf_model_dinov2", out / "hf_model_dinov2")
    _copy_tree(root / "hf_model_dimos_replay_latent", out / "hf_model_dimos_replay_latent")
    _copy_tree(root / "artifacts/replay_mpc_arena/decision_traces", out / "replay_mpc_arena_decision_traces")

    readme = """# WorldForge Go2 Trace Judge Submission Bundle

## Main Demo

- `final_hackathon_video.mp4`: final judge video shell for external voiceover.
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
"""
    (out / "README.md").write_text(readme)
    manifest = {
        "schema_version": 1,
        "bundle_dir": str(out),
        "artifacts": sorted(p.name for p in out.iterdir()),
        "claim_boundary": "Small scorers over real Go2 frames, label-safe counterfactual traces, and DimOS replay latent dynamics. Not a Go2 foundation model.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(out)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the hackathon submission artifact bundle.")
    parser.add_argument("--output-dir", default="submission_bundle")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
