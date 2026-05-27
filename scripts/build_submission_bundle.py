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

    readme = """# WorldForge Go2 Trace Judge Submission Bundle

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
"""
    (out / "README.md").write_text(readme)
    manifest = {
        "schema_version": 1,
        "bundle_dir": str(out),
        "artifacts": sorted(p.name for p in out.iterdir()),
        "claim_boundary": "Small micro world scorer over real Go2 frames and label-safe counterfactual traces. Not a Go2 foundation model.",
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
