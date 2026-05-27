from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


JSON = dict[str, Any]


def _read_json(path: Path) -> JSON:
    if not path.is_file():
        raise RuntimeError(f"Missing required file: {path}")
    return json.loads(path.read_text())


def _rel(path: str | Path, base: Path) -> str:
    return os.path.relpath(Path(path).resolve(), base.resolve())


def _sample_steps(steps: list[JSON], max_frames: int) -> list[JSON]:
    if len(steps) <= max_frames:
        return steps

    selected: dict[int, JSON] = {}
    for step in steps[:3]:
        selected[int(step["frame_index"])] = step
    for step in steps[-3:]:
        selected[int(step["frame_index"])] = step

    previous_action: str | None = None
    for step in steps:
        action = str(step["selected_candidate_id"])
        if action != previous_action:
            selected[int(step["frame_index"])] = step
            previous_action = action

    stride = max(1, len(steps) // max(1, max_frames - len(selected)))
    for step in steps[::stride]:
        selected[int(step["frame_index"])] = step
        if len(selected) >= max_frames:
            break
    return [selected[index] for index in sorted(selected)[:max_frames]]


def _load_scores(step: JSON) -> list[JSON]:
    path = Path(step["step_dir"]) / "candidate_scores.json"
    return _read_json(path)["scores"]


def _draw_contact_sheet(sample_steps: list[JSON], output_path: Path) -> None:
    thumb_w = 480
    thumb_h = 270
    text_h = 108
    cols = 2
    rows = (len(sample_steps) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + text_h)), (18, 19, 22))
    draw = ImageDraw.Draw(sheet)

    for index, step in enumerate(sample_steps):
        col = index % cols
        row = index // cols
        x0 = col * thumb_w
        y0 = row * (thumb_h + text_h)
        frame = Image.open(step["annotated_frame"]).convert("RGB")
        frame.thumbnail((thumb_w, thumb_h))
        sheet.paste(frame, (x0, y0))

        scores = _load_scores(step)
        top = sorted(scores, key=lambda item: item["score"], reverse=True)[:2]
        lines = [
            f"frame {step['frame_index']:02d} selected={step['selected_candidate_id']} score={step['score']}",
            f"target_conf={step['target']['confidence']} x={step['target']['center_x']} unsafe={step['unsafe_risk']}",
            f"top: {top[0]['candidate_id']}={top[0]['score']} | {top[1]['candidate_id']}={top[1]['score']}",
            "human: correct / wrong / unsure?",
        ]
        for line_index, line in enumerate(lines):
            draw.text((x0 + 12, y0 + thumb_h + 10 + line_index * 22), line, fill=(244, 240, 230))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def _write_csv(sample_steps: list[JSON], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_index",
                "selected_candidate_id",
                "score",
                "target_confidence",
                "target_center_x",
                "unsafe_risk",
                "human_label",
                "human_note",
            ],
        )
        writer.writeheader()
        for step in sample_steps:
            writer.writerow(
                {
                    "frame_index": step["frame_index"],
                    "selected_candidate_id": step["selected_candidate_id"],
                    "score": step["score"],
                    "target_confidence": step["target"]["confidence"],
                    "target_center_x": step["target"]["center_x"],
                    "unsafe_risk": step["unsafe_risk"],
                    "human_label": "",
                    "human_note": "",
                }
            )


def _write_html(sample_steps: list[JSON], output_dir: Path, contact_sheet: Path, labels_csv: Path) -> None:
    cards = []
    for step in sample_steps:
        scores = _load_scores(step)
        score_items = "".join(
            f"<li><strong>{html.escape(row['candidate_id'])}</strong>: {row['score']} - {html.escape(row['reason'])}</li>"
            for row in scores
        )
        cards.append(
            f"""
            <article>
              <img src="{html.escape(_rel(step['annotated_frame'], output_dir))}" alt="frame {step['frame_index']}">
              <div>
                <h2>Frame {step['frame_index']:02d}: {html.escape(step['selected_candidate_id'])}</h2>
                <p>score <strong>{step['score']}</strong> | target confidence <strong>{step['target']['confidence']}</strong> | center_x <strong>{step['target']['center_x']}</strong> | unsafe risk <strong>{step['unsafe_risk']}</strong></p>
                <ul>{score_items}</ul>
                <p class="review">Human review: correct / wrong / unsure. Put notes in <code>{html.escape(_rel(labels_csv, output_dir))}</code>.</p>
              </div>
            </article>
            """
        )

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>WorldForge Go2 Human Review</title>
  <style>
    :root {{ color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    body {{ margin: 0; background: #111215; color: #f5f3ed; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px 18px 48px; }}
    h1 {{ margin: 0 0 8px; font-size: 36px; letter-spacing: 0; }}
    p {{ color: #d8d2c8; line-height: 1.55; }}
    a {{ color: #8cc8ff; }}
    .warning {{ border-left: 4px solid #f4c542; padding: 10px 14px; background: #25200f; }}
    article {{ display: grid; grid-template-columns: 420px 1fr; gap: 16px; padding: 14px; margin: 14px 0; border: 1px solid #34363a; background: #191a1e; }}
    img {{ width: 100%; aspect-ratio: 16 / 9; object-fit: cover; background: #000; }}
    h2 {{ margin: 0 0 8px; }}
    li {{ margin: 4px 0; color: #e7e0d5; }}
    code {{ background: #25272d; padding: 2px 4px; }}
    .review {{ color: #f2df97; }}
    @media (max-width: 820px) {{ article {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
<main>
  <h1>WorldForge Go2 Human Review</h1>
  <p class="warning">The labels in this dataset currently come from the transparent scorer, not from measured execution outcomes. Use this review pack to mark whether the selected action makes sense.</p>
  <p>Contact sheet: <a href="{html.escape(_rel(contact_sheet, output_dir))}">{html.escape(contact_sheet.name)}</a></p>
  <p>Label template: <a href="{html.escape(_rel(labels_csv, output_dir))}">{html.escape(labels_csv.name)}</a></p>
  {''.join(cards)}
</main>
</body>
</html>
"""
    (output_dir / "human_review.html").write_text(page)


def run(args: argparse.Namespace) -> int:
    replay_dir = Path(args.replay_dir).expanduser()
    summary = _read_json(replay_dir / "summary.json")
    sample_steps = _sample_steps(summary["steps"], args.max_frames)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    contact_sheet = output_dir / "contact_sheet.jpg"
    labels_csv = output_dir / "human_labels_template.csv"
    _draw_contact_sheet(sample_steps, contact_sheet)
    _write_csv(sample_steps, labels_csv)
    _write_html(sample_steps, output_dir, contact_sheet, labels_csv)

    print(output_dir / "human_review.html")
    print(contact_sheet)
    print(labels_csv)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a human review pack for Go2 trace labels.")
    parser.add_argument("--replay-dir", default="artifacts/replay_run")
    parser.add_argument("--output-dir", default="artifacts/human_review")
    parser.add_argument("--max-frames", type=int, default=14)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
