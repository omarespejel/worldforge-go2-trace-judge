from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path


def _rel(path: str | Path, base: Path) -> str:
    return str(Path(path).resolve().relative_to(base.resolve()))


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.replay_dir).expanduser()
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"Missing summary: {summary_path}")
    summary = json.loads(summary_path.read_text())
    steps = summary["steps"]
    counts = Counter(step["selected_candidate_id"] for step in steps)
    sample_steps = steps[: min(8, len(steps))]

    rows = []
    for step in sample_steps:
        frame = _rel(step["annotated_frame"], output_dir)
        candidate_scores = Path(step["step_dir"]) / "candidate_scores.json"
        score_info = Path(step["step_dir"]) / "score_info.json"
        rows.append(
            f"""
            <article class="step-card">
              <img src="{html.escape(frame)}" alt="Annotated frame {step['frame_index']}">
              <div>
                <h3>Frame {step['frame_index']:02d}: {html.escape(step['selected_candidate_id'])}</h3>
                <p>score <strong>{step['score']}</strong>, target confidence <strong>{step['target']['confidence']}</strong>, unsafe risk <strong>{step['unsafe_risk']}</strong></p>
                <a href="{html.escape(_rel(candidate_scores, output_dir))}">candidate_scores.json</a>
                <a href="{html.escape(_rel(score_info, output_dir))}">score_info.json</a>
              </div>
            </article>
            """
        )

    dist = "".join(
        f"<li><strong>{html.escape(action)}</strong>: {count} frames</li>"
        for action, count in counts.most_common()
    )
    replay_video = _rel(summary["replay_video"], output_dir)
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>WorldForge Go2 Trace Judge</title>
  <style>
    :root {{
      color-scheme: dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #101113;
      color: #f5f3ed;
    }}
    body {{ margin: 0; background: #101113; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 32px 20px 48px; }}
    h1 {{ font-size: 42px; margin: 0 0 8px; letter-spacing: 0; }}
    h2 {{ margin-top: 34px; font-size: 22px; }}
    p {{ line-height: 1.55; color: #d7d2c8; }}
    .hero {{ display: grid; gap: 20px; }}
    video {{ width: 100%; border: 1px solid #3a3a3a; background: #000; }}
    .facts {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-top: 16px; }}
    .fact {{ border: 1px solid #353535; padding: 14px; background: #18191c; }}
    .fact span {{ display: block; color: #a7a093; font-size: 12px; text-transform: uppercase; }}
    .fact strong {{ font-size: 24px; }}
    .step-grid {{ display: grid; gap: 14px; }}
    .step-card {{ display: grid; grid-template-columns: 220px 1fr; gap: 14px; border: 1px solid #343434; background: #18191c; padding: 12px; }}
    .step-card img {{ width: 220px; aspect-ratio: 16 / 9; object-fit: cover; }}
    .step-card h3 {{ margin: 0 0 6px; }}
    .step-card a {{ display: inline-block; margin-right: 12px; color: #8cc8ff; }}
    code, pre {{ background: #1e2024; color: #f1e7d0; }}
    pre {{ overflow: auto; padding: 14px; border: 1px solid #343434; }}
    @media (max-width: 760px) {{
      h1 {{ font-size: 32px; }}
      .facts {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .step-card {{ grid-template-columns: 1fr; }}
      .step-card img {{ width: 100%; }}
    }}
  </style>
</head>
<body>
<main>
  <section class="hero">
    <div>
      <h1>WorldForge Go2 Trace Judge</h1>
      <p>Go2 camera observations become candidate robot moves, transparent scores, selected actions, and replayable evidence. The robot runtime stays host-owned; WorldForge is the trace judge.</p>
    </div>
    <video controls src="{html.escape(replay_video)}"></video>
  </section>

  <section class="facts">
    <div class="fact"><span>Run</span><strong>{html.escape(summary['run_id'])}</strong></div>
    <div class="fact"><span>Frames</span><strong>{summary['frame_count']}</strong></div>
    <div class="fact"><span>Target</span><strong>{html.escape(summary['target'])}</strong></div>
    <div class="fact"><span>Unsafe</span><strong>{html.escape(', '.join(summary['unsafe_colors']))}</strong></div>
  </section>

  <h2>Selected Action Distribution</h2>
  <ul>{dist}</ul>

  <h2>Evidence Contract</h2>
  <pre>observation_summary + task + candidate actions
-> score_info.json
-> candidate_scores.json
-> selected_action.json
-> outcome_after_execution.json</pre>

  <h2>Sample Frames</h2>
  <section class="step-grid">
    {''.join(rows)}
  </section>
</main>
</body>
</html>
"""
    out = output_dir / "report.html"
    out.write_text(page)
    print(out)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build static HTML report for Go2 trace replay.")
    parser.add_argument("--replay-dir", default="artifacts/replay_run")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
