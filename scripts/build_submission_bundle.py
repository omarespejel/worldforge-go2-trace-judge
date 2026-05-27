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


def run(args: argparse.Namespace) -> int:
    root = Path.cwd()
    out = Path(args.output_dir).expanduser()
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "demo_video": root / "artifacts/replay_run/worldforge_trace_replay.mp4",
        "demo_report": root / "artifacts/replay_run/report.html",
        "human_review": root / "artifacts/human_review/human_review.html",
        "contact_sheet": root / "artifacts/human_review/contact_sheet.jpg",
        "audit_report": root / "artifacts/dataset_audit/audit_report.html",
        "audit_summary": root / "artifacts/dataset_audit/audit_summary.json",
        "trace_dataset": root / "dataset/go2_trace_candidates.jsonl",
        "reviewed_dataset": root / "artifacts/dataset_audit/go2_trace_candidates_reviewed.jsonl",
        "ranker_model": root / "artifacts/ranker_smoke/model.json",
        "step_01_score_info": root / "artifacts/replay_run/trace/step_01/score_info.json",
        "step_01_candidate_scores": root / "artifacts/replay_run/trace/step_01/candidate_scores.json",
        "step_01_selected_action": root / "artifacts/replay_run/trace/step_01/selected_action.json",
    }
    for name, src in artifacts.items():
        suffix = src.suffix
        _copy(src, out / f"{name}{suffix}")

    readme = """# WorldForge Go2 Trace Judge Submission Bundle

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
"""
    (out / "README.md").write_text(readme)
    manifest = {"schema_version": 1, "bundle_dir": str(out), "artifacts": sorted(p.name for p in out.iterdir())}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(out)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the hackathon submission artifact bundle.")
    parser.add_argument("--output-dir", default="submission_bundle")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
