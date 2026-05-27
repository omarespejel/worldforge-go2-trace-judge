from __future__ import annotations

import argparse
import csv
import html
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


JSON = dict[str, Any]
VALID_HUMAN_LABELS = {"correct", "wrong", "unsure", ""}


def _read_json(path: Path) -> JSON:
    if not path.is_file():
        raise RuntimeError(f"Missing required file: {path}")
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[JSON]:
    if not path.is_file():
        raise RuntimeError(f"Dataset not found: {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _rel(path: str | Path, base: Path) -> str:
    return os.path.relpath(Path(path).resolve(), base.resolve())


def _read_human_labels(path: Path | None) -> dict[int, JSON]:
    if path is None or not path.is_file():
        return {}
    labels: dict[int, JSON] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("frame_index"):
                continue
            label = row.get("human_label", "").strip().lower()
            if label not in VALID_HUMAN_LABELS:
                raise RuntimeError(
                    f"Invalid human_label {label!r} for frame {row['frame_index']}. "
                    f"Use one of {sorted(VALID_HUMAN_LABELS)}."
                )
            if not label:
                continue
            labels[int(row["frame_index"])] = {
                "human_label": label,
                "human_note": row.get("human_note", "").strip(),
            }
    return labels


def _auto_flags(step: JSON, min_confidence: float, max_abs_center_x: float) -> list[str]:
    flags: list[str] = []
    target = step["target"]
    confidence = float(target["confidence"])
    center_x = float(target["center_x"])
    score = float(step["score"])
    if confidence < min_confidence:
        flags.append("low_target_confidence")
    if abs(center_x) > max_abs_center_x and step["selected_candidate_id"] == "forward_small":
        flags.append("forward_selected_while_target_off_center")
    if score < 0.62:
        flags.append("low_selected_score")
    return flags


def _step_rows(rows: list[JSON], step_index: int) -> list[JSON]:
    return [row for row in rows if int(row["step_index"]) == step_index]


def _write_reviewed_jsonl(rows: list[JSON], output_path: Path, step_reviews: dict[int, JSON]) -> None:
    reviewed: list[JSON] = []
    for row in rows:
        step_index = int(row["step_index"])
        review = step_reviews[step_index]
        item = dict(row)
        item["human_label"] = review.get("human_label")
        item["human_note"] = review.get("human_note")
        item["auto_flags"] = review["auto_flags"]
        item["usable_for_training"] = review["usable_for_training"]
        reviewed.append(item)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in reviewed) + "\n")


def _write_html(output_path: Path, summary: JSON, reviewed_steps: list[JSON]) -> None:
    cards = []
    base = output_path.parent
    for step in reviewed_steps:
        flags = ", ".join(step["auto_flags"]) or "none"
        human = step.get("human_label") or "unreviewed"
        candidate_rows = "".join(
            f"<li><strong>{html.escape(row['candidate_id'])}</strong>: {row['transparent_score_label']} - {html.escape(row['reason'])}</li>"
            for row in step["candidate_rows"]
        )
        cards.append(
            f"""
            <article class="{html.escape(step['status_class'])}">
              <img src="{html.escape(_rel(step['annotated_frame'], base))}" alt="frame {step['frame_index']}">
              <div>
                <h2>Frame {step['frame_index']:02d}: {html.escape(step['selected_candidate_id'])}</h2>
                <p>human <strong>{html.escape(human)}</strong> | usable <strong>{step['usable_for_training']}</strong> | flags <strong>{html.escape(flags)}</strong></p>
                <p>target confidence <strong>{step['target']['confidence']}</strong> | center_x <strong>{step['target']['center_x']}</strong> | score <strong>{step['score']}</strong></p>
                <ul>{candidate_rows}</ul>
                <p>{html.escape(step.get('human_note') or '')}</p>
              </div>
            </article>
            """
        )

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>WorldForge Go2 Dataset Audit</title>
  <style>
    :root {{ color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    body {{ margin: 0; background: #101113; color: #f5f3ed; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px 18px 48px; }}
    h1 {{ margin: 0 0 8px; font-size: 36px; letter-spacing: 0; }}
    p {{ color: #d8d2c8; line-height: 1.55; }}
    .facts {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 18px 0; }}
    .fact {{ border: 1px solid #34363a; background: #18191c; padding: 12px; }}
    .fact span {{ display: block; color: #a9a195; font-size: 12px; text-transform: uppercase; }}
    .fact strong {{ font-size: 24px; }}
    article {{ display: grid; grid-template-columns: 420px 1fr; gap: 16px; padding: 14px; margin: 14px 0; border: 1px solid #34363a; background: #191a1e; }}
    article.flagged {{ border-left: 4px solid #f4c542; }}
    article.rejected {{ border-left: 4px solid #ff6666; }}
    article.accepted {{ border-left: 4px solid #65d683; }}
    img {{ width: 100%; aspect-ratio: 16 / 9; object-fit: cover; background: #000; }}
    li {{ margin: 4px 0; color: #e7e0d5; }}
    @media (max-width: 820px) {{ .facts {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} article {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
<main>
  <h1>WorldForge Go2 Dataset Audit</h1>
  <p>This report separates trace plumbing from training-quality labels. A row is training-usable only if it passes auto checks and is not marked wrong by a human.</p>
  <section class="facts">
    <div class="fact"><span>Steps</span><strong>{summary['step_count']}</strong></div>
    <div class="fact"><span>Usable</span><strong>{summary['usable_step_count']}</strong></div>
    <div class="fact"><span>Flagged</span><strong>{summary['flagged_step_count']}</strong></div>
    <div class="fact"><span>Human Reviewed</span><strong>{summary['human_reviewed_step_count']}</strong></div>
  </section>
  {''.join(cards)}
</main>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page)


def run(args: argparse.Namespace) -> int:
    replay_dir = Path(args.replay_dir).expanduser()
    dataset_jsonl = Path(args.dataset_jsonl).expanduser()
    labels_path = Path(args.human_labels).expanduser() if args.human_labels else None
    output_dir = Path(args.output_dir).expanduser()

    summary = _read_json(replay_dir / "summary.json")
    rows = _read_jsonl(dataset_jsonl)
    human_labels = _read_human_labels(labels_path)

    step_reviews: dict[int, JSON] = {}
    reviewed_steps: list[JSON] = []
    for step in summary["steps"]:
        step_index = int(step["frame_index"])
        labels = human_labels.get(step_index, {})
        flags = _auto_flags(step, args.min_confidence, args.max_abs_center_x)
        human_label = labels.get("human_label")
        usable = not flags and human_label != "wrong"
        if human_label == "correct":
            usable = True
        if human_label == "unsure":
            usable = False
        status_class = "accepted" if usable else ("rejected" if human_label == "wrong" else "flagged")
        review = {
            "frame_index": step_index,
            "selected_candidate_id": step["selected_candidate_id"],
            "score": step["score"],
            "target": step["target"],
            "annotated_frame": step["annotated_frame"],
            "auto_flags": flags,
            "human_label": human_label,
            "human_note": labels.get("human_note", ""),
            "usable_for_training": usable,
            "status_class": status_class,
            "candidate_rows": _step_rows(rows, step_index),
        }
        step_reviews[step_index] = review
        reviewed_steps.append(review)

    reviewed_jsonl = output_dir / "go2_trace_candidates_reviewed.jsonl"
    _write_reviewed_jsonl(rows, reviewed_jsonl, step_reviews)

    flag_counts = Counter(flag for review in reviewed_steps for flag in review["auto_flags"])
    human_counts = Counter(review.get("human_label") or "unreviewed" for review in reviewed_steps)
    selected_counts = Counter(review["selected_candidate_id"] for review in reviewed_steps)
    audit_summary = {
        "schema_version": 1,
        "replay_dir": str(replay_dir),
        "dataset_jsonl": str(dataset_jsonl),
        "human_labels": str(labels_path) if labels_path else None,
        "reviewed_jsonl": str(reviewed_jsonl),
        "step_count": len(reviewed_steps),
        "candidate_row_count": len(rows),
        "usable_step_count": sum(1 for review in reviewed_steps if review["usable_for_training"]),
        "flagged_step_count": sum(1 for review in reviewed_steps if review["auto_flags"]),
        "human_reviewed_step_count": sum(1 for review in reviewed_steps if review.get("human_label")),
        "flag_counts": dict(flag_counts),
        "human_label_counts": dict(human_counts),
        "selected_candidate_counts": dict(selected_counts),
        "thresholds": {
            "min_confidence": args.min_confidence,
            "max_abs_center_x": args.max_abs_center_x,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit_summary.json").write_text(json.dumps(audit_summary, indent=2, sort_keys=True) + "\n")
    _write_html(output_dir / "audit_report.html", audit_summary, reviewed_steps)
    print(json.dumps(audit_summary, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Go2 trace labels with optional human labels.")
    parser.add_argument("--replay-dir", default="artifacts/replay_run")
    parser.add_argument("--dataset-jsonl", default="dataset/go2_trace_candidates.jsonl")
    parser.add_argument("--human-labels", default="artifacts/human_review/human_labels_template.csv")
    parser.add_argument("--output-dir", default="artifacts/dataset_audit")
    parser.add_argument("--min-confidence", type=float, default=0.05)
    parser.add_argument("--max-abs-center-x", type=float, default=0.75)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
