from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


JSON = dict[str, Any]


def _read_json(path: Path) -> JSON:
    if not path.is_file():
        raise RuntimeError(f"Missing required artifact: {path}")
    return json.loads(path.read_text())


def _safe_get(data: JSON, path: list[str], default: Any = None) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _rows_for_step(step_dir: Path) -> list[JSON]:
    candidate_scores = _read_json(step_dir / "candidate_scores.json")
    score_info_artifact = _read_json(step_dir / "score_info.json")
    selected_action = _read_json(step_dir / "selected_action.json")
    outcome = _read_json(step_dir / "outcome_after_execution.json")
    venue_input = _read_json(step_dir / "venue_input.json")

    score_info = score_info_artifact.get("score_info", {})
    observation = score_info.get("observation_summary", {})
    visual = observation.get("visual_summary", {})
    nav = observation.get("navigation_state", {})
    costmap = observation.get("costmap_summary", {})
    candidates_by_id = {candidate["id"]: candidate for candidate in venue_input.get("candidates", [])}
    selected_id = candidate_scores["selected_candidate_id"]

    rows: list[JSON] = []
    for score_row in candidate_scores.get("scores", []):
        candidate_id = score_row["candidate_id"]
        candidate = candidates_by_id.get(candidate_id, {})
        rows.append(
            {
                "schema_version": 1,
                "source": "worldforge-go2-trace-demo",
                "run_id": candidate_scores.get("run_id"),
                "step_index": candidate_scores.get("step_index"),
                "step_dir": str(step_dir),
                "candidate_id": candidate_id,
                "action": candidate.get("action"),
                "params": candidate.get("params", {}),
                "features": score_row.get("features", {}),
                "transparent_score_label": score_row.get("score"),
                "selected": candidate_id == selected_id,
                "selected_candidate_id": selected_id,
                "reason": score_row.get("reason"),
                "goal": _safe_get(score_info, ["task", "human_goal"]),
                "target_label": visual.get("target_label"),
                "target_confidence": visual.get("target_confidence"),
                "target_bearing_degrees": visual.get("target_bearing_degrees"),
                "target_area_ratio": visual.get("target_area_ratio"),
                "unsafe_color_risk": visual.get("unsafe_color_risk"),
                "blocked_ahead": costmap.get("blocked_ahead"),
                "localized": nav.get("localized"),
                "stuck_probability": nav.get("stuck_probability"),
                "executed": outcome.get("executed"),
                "execution_result": outcome.get("execution_result"),
                "manual_intervention": _safe_get(
                    outcome,
                    ["outcome_after_execution", "manual_intervention"],
                    False,
                ),
            }
        )
    return rows


def run(args: argparse.Namespace) -> int:
    trace_dir = Path(args.trace_dir).expanduser()
    if not trace_dir.is_dir():
        raise RuntimeError(f"Trace directory not found: {trace_dir}")

    step_dirs = sorted(path for path in trace_dir.glob("step_*") if path.is_dir())
    if not step_dirs:
        raise RuntimeError(f"No step_* directories found under {trace_dir}")

    rows: list[JSON] = []
    for step_dir in step_dirs:
        rows.extend(_rows_for_step(step_dir))

    output_jsonl = Path(args.output_jsonl).expanduser()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")

    selected_counts = Counter(row["selected_candidate_id"] for row in rows if row["selected"])
    candidate_counts = Counter(row["candidate_id"] for row in rows)
    summary = {
        "schema_version": 1,
        "trace_dir": str(trace_dir),
        "output_jsonl": str(output_jsonl),
        "step_count": len(step_dirs),
        "candidate_row_count": len(rows),
        "selected_candidate_counts": dict(selected_counts),
        "candidate_counts": dict(candidate_counts),
        "label": "transparent_score_label",
        "note": "This is a trace dataset for scorer training or distillation. Labels come from the transparent scorer unless replaced with measured outcomes later.",
    }
    if args.summary_output:
        summary_output = Path(args.summary_output).expanduser()
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flatten WorldForge-shaped Go2 trace artifacts into candidate rows."
    )
    parser.add_argument("--trace-dir", default="artifacts/replay_run/trace")
    parser.add_argument("--output-jsonl", default="dataset/go2_trace_candidates.jsonl")
    parser.add_argument("--summary-output", default="dataset/go2_trace_dataset_summary.json")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
