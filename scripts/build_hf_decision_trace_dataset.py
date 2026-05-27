from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from go2_find_colored_target import (
    _build_candidates,
    _detect_scene,
    _observation_summary,
    _score,
)


ROOT = Path(__file__).resolve().parents[1]
COLORS = {
    "red": (230, 45, 58),
    "green": (22, 170, 82),
    "yellow": (236, 198, 54),
    "blue": (55, 120, 230),
}
SCHEMA_VERSION = "0.1.0"


JSON = dict[str, Any]


def write_jsonl(path: Path, rows: list[JSON]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def read_json(path: Path) -> JSON:
    return json.loads(path.read_text())


def split_for_index(index: int) -> str:
    if index % 10 == 0:
        return "test"
    if index % 10 in {1, 2}:
        return "validation"
    return "train"


def selected_score(candidate_scores: JSON) -> float:
    selected = candidate_scores["selected_candidate_id"]
    for row in candidate_scores["scores"]:
        if row["candidate_id"] == selected:
            return float(row["score"])
    raise KeyError(selected)


def row_from_artifacts(
    *,
    row_id: str,
    source_domain: str,
    image_path: str,
    score_info: JSON,
    candidate_scores: JSON,
    selected_action: JSON,
    outcome: JSON,
    synthetic_config: JSON | None = None,
) -> JSON:
    score_payload = score_info["score_info"]
    observation = score_payload["observation_summary"]
    task = score_payload["task"]
    target_label = observation["visual_summary"]["target_label"]
    unsafe_risk = float(observation["visual_summary"]["unsafe_color_risk"])
    return {
        "schema_version": SCHEMA_VERSION,
        "row_id": row_id,
        "source_domain": source_domain,
        "embodiment": "unitree_go2",
        "task_type": "color_target_decision_trace",
        "goal_text": task["human_goal"],
        "target_label": target_label,
        "image": image_path,
        "observation_summary": observation,
        "action_candidates": score_info["action_candidates"],
        "candidate_scores": candidate_scores["scores"],
        "selected_candidate_id": candidate_scores["selected_candidate_id"],
        "selected_score": selected_score(candidate_scores),
        "selected_action": {
            "action": selected_action["action"],
            "params": selected_action["params"],
            "worldforge_executes_robot": selected_action.get("worldforge_executes_robot", False),
        },
        "label_source": "transparent_heuristic_score",
        "outcome_after_execution": outcome.get("outcome_after_execution", {}),
        "features_summary": {
            "target_found": bool(observation["visual_summary"]["target_confidence"] > 0.0),
            "target_confidence": observation["visual_summary"]["target_confidence"],
            "target_bearing_degrees": observation["visual_summary"]["target_bearing_degrees"],
            "target_area_ratio": observation["visual_summary"]["target_area_ratio"],
            "unsafe_color_risk": unsafe_risk,
            "blocked_ahead": observation["costmap_summary"]["blocked_ahead"],
        },
        "synthetic_config": synthetic_config,
        "limitations": [
            "Labels are scorer outputs, not human outcome labels.",
            "Rows are suitable for scorer smoke tests and schema integration, not policy imitation.",
            "Synthetic rows must be evaluated separately from real Go2 seed rows.",
        ],
    }


def draw_synthetic_scene(path: Path, config: JSON) -> None:
    width, height = 640, 360
    image = Image.new("RGB", (width, height), (205, 209, 201))
    draw = ImageDraw.Draw(image)

    # Background: hallway wall and reflective floor.
    draw.rectangle((0, 0, width, 135), fill=(180, 186, 180))
    draw.rectangle((0, 135, width, height), fill=(220, 222, 214))
    draw.line((0, 135, width, 135), fill=(145, 150, 148), width=3)
    for x in range(70, width, 110):
        draw.line((x, 0, x + 30, 135), fill=(157, 163, 162), width=2)
    for x in range(-120, width, 120):
        draw.line((x, height, x + 250, 135), fill=(200, 203, 198), width=2)

    def block(color_name: str, cx_norm: float, area_scale: float, y_bias: float = 0.0) -> None:
        color = COLORS[color_name]
        size = int(18 + 72 * area_scale)
        cx = int(width * (0.5 + 0.42 * cx_norm))
        floor_y = int(height * (0.78 + y_bias))
        x1 = max(0, min(width - size - 2, cx - size // 2))
        y1 = max(145, min(height - size - 2, floor_y - size // 2))
        draw.rectangle((x1 + 5, y1 + size - 4, x1 + size + 7, y1 + size + 3), fill=(150, 150, 145))
        draw.rectangle((x1, y1, x1 + size, y1 + size), fill=color, outline=(80, 80, 80), width=2)
        draw.polygon(
            [(x1 + size, y1), (x1 + size + 9, y1 + 7), (x1 + size + 9, y1 + size + 7), (x1 + size, y1 + size)],
            fill=tuple(max(0, c - 35) for c in color),
        )

    block(str(config["target_color"]), float(config["target_x"]), float(config["target_scale"]))
    for obstacle in config["unsafe_blocks"]:
        block(str(obstacle["color"]), float(obstacle["x"]), float(obstacle["scale"]), y_bias=float(obstacle.get("y_bias", 0)))

    # Add a little deterministic sensor texture.
    rng = random.Random(config["seed"])
    px = image.load()
    for _ in range(2600):
        x = rng.randrange(width)
        y = rng.randrange(height)
        r, g, b = px[x, y]
        delta = rng.randrange(-7, 8)
        px[x, y] = (max(0, min(255, r + delta)), max(0, min(255, g + delta)), max(0, min(255, b + delta)))

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=90)


def build_synthetic_rows(output_dir: Path, count: int, seed: int) -> dict[str, list[JSON]]:
    rng = random.Random(seed)
    by_split: dict[str, list[JSON]] = {"train": [], "validation": [], "test": []}
    positions = [-0.82, -0.45, -0.12, 0.0, 0.18, 0.48, 0.82]
    target_colors = ["red", "green", "yellow", "blue"]
    unsafe_palette = ["green", "yellow"]

    for index in range(count):
        target_color = target_colors[index % len(target_colors)]
        unsafe_colors = [c for c in unsafe_palette if c != target_color]
        if not unsafe_colors:
            unsafe_colors = ["red"]
        config: JSON = {
            "seed": seed + index,
            "target_color": target_color,
            "target_x": positions[index % len(positions)] + rng.uniform(-0.05, 0.05),
            "target_scale": rng.uniform(0.28, 0.95),
            "unsafe_blocks": [],
        }
        if index % 3 != 0:
            config["unsafe_blocks"].append(
                {
                    "color": unsafe_colors[index % len(unsafe_colors)],
                    "x": rng.choice([-0.25, 0.0, 0.25]) + rng.uniform(-0.05, 0.05),
                    "scale": rng.uniform(0.18, 0.60),
                    "y_bias": rng.uniform(0.00, 0.07),
                }
            )
        if index % 7 == 0:
            config["unsafe_blocks"].append(
                {
                    "color": "blue" if target_color != "blue" else "red",
                    "x": rng.choice([-0.72, 0.72]),
                    "scale": rng.uniform(0.14, 0.40),
                    "y_bias": rng.uniform(-0.03, 0.04),
                }
            )

        split = split_for_index(index)
        rel_image = Path("images") / "synthetic" / split / f"synthetic_{index:04d}.jpg"
        abs_image = output_dir / rel_image
        draw_synthetic_scene(abs_image, config)

        unsafe_set = {str(block["color"]) for block in config["unsafe_blocks"]}
        scene = _detect_scene(abs_image, target_color, unsafe_set)
        candidates = _build_candidates(scene)
        scores = [_score(candidate["features"]) for candidate in candidates]
        best_index = max(range(len(scores)), key=scores.__getitem__)
        selected = candidates[best_index]
        observation = _observation_summary(scene, target_color)
        task = {
            "human_goal": f"find the {target_color} block and avoid unsafe colored markers",
            "goal_representation": {
                "type": "host_interpreted_visual_goal",
                "target_label": target_color,
                "unsafe_markers": sorted(unsafe_set),
            },
        }
        score_info = {
            "provider": "transparent-go2-color-target-score",
            "capability": "score",
            "score_info": {
                "schema_version": 1,
                "run_id": "hf-synthetic",
                "step_index": index,
                "embodiment": "unitree_go2",
                "host_runtime": "offline-synthetic-generator",
                "input_source": "procedural_color_block_scene",
                "task": task,
                "observation_summary": observation,
            },
            "action_candidates": candidates,
        }
        candidate_scores = {
            "schema_version": 1,
            "run_id": "hf-synthetic",
            "step_index": index,
            "scores": [
                {
                    "candidate_id": candidate["id"],
                    "score": score,
                    "features": candidate["features"],
                    "reason": candidate["reason_hint"],
                }
                for candidate, score in zip(candidates, scores)
            ],
            "selected_candidate_id": selected["id"],
            "lower_is_better": False,
        }
        selected_action = {
            "action": selected["action"],
            "params": selected["params"],
            "worldforge_executes_robot": False,
        }
        outcome = {
            "outcome_after_execution": {
                "target_confidence": scene["target"]["confidence"],
                "target_area_ratio": scene["target"]["area_ratio"],
                "unsafe_color_risk": scene["unsafe_risk"],
                "manual_intervention": False,
            }
        }
        row = row_from_artifacts(
            row_id=f"synthetic-{index:04d}",
            source_domain="synthetic_procedural",
            image_path=str(rel_image),
            score_info=score_info,
            candidate_scores=candidate_scores,
            selected_action=selected_action,
            outcome=outcome,
            synthetic_config=config,
        )
        by_split[split].append(row)
    return by_split


def trace_dirs() -> list[tuple[str, Path]]:
    roots = [
        ("real_go2_replay", ROOT / "artifacts/replay_run/trace"),
        ("real_go2_single_frame", ROOT / "artifacts/live_ciro_detection"),
        ("real_go2_control_probe", ROOT / "artifacts/final_rise_move_detection"),
    ]
    result: list[tuple[str, Path]] = []
    for source, root in roots:
        if not root.exists():
            continue
        for step in sorted(root.rglob("selected_action.json")):
            result.append((source, step.parent))
    return result


def build_real_seed_rows(output_dir: Path, max_rows: int | None = None) -> list[JSON]:
    rows: list[JSON] = []
    for index, (source, step_dir) in enumerate(trace_dirs()):
        if max_rows is not None and len(rows) >= max_rows:
            break
        src_frame = step_dir / "camera_frame.jpg"
        if not src_frame.exists():
            continue
        rel_image = Path("images") / "real_seed" / f"{source}_{index:04d}.jpg"
        dst_frame = output_dir / rel_image
        dst_frame.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_frame, dst_frame)
        row = row_from_artifacts(
            row_id=f"real-seed-{index:04d}",
            source_domain=source,
            image_path=str(rel_image),
            score_info=read_json(step_dir / "score_info.json"),
            candidate_scores=read_json(step_dir / "candidate_scores.json"),
            selected_action=read_json(step_dir / "selected_action.json"),
            outcome=read_json(step_dir / "outcome_after_execution.json"),
        )
        row["split"] = "real_seed"
        row["privacy_review_required"] = True
        rows.append(row)
    return rows


def write_dataset_card(output_dir: Path, counts: JSON) -> None:
    card = f"""---
license: mit
task_categories:
- robotics
- reinforcement-learning
tags:
- robotics
- unitree-go2
- world-models
- decision-traces
- counterfactual-evaluation
- synthetic-data
pretty_name: WorldForge Go2 Trace Judge Dataset
size_categories:
- n<1K
---

# WorldForge Go2 Trace Judge Dataset

This is a small decision-trace dataset for inspectable Unitree Go2 autonomy. It is
not an imitation-learning policy dataset. Each row records:

```text
observation + goal + candidate actions + transparent scores + selected action
```

The dataset is designed for scorer integration tests, learned-ranker smoke tests,
and future world-model scoring research.

## Splits

| split | rows | source |
|---|---:|---|
| train | {counts['train']} | procedural synthetic color-block scenes |
| validation | {counts['validation']} | procedural synthetic color-block scenes |
| test | {counts['test']} | procedural synthetic color-block scenes |
| real_seed | {counts['real_seed']} | real Go2 camera/replay trace artifacts |

## Schema

Important fields:

- `image`: relative path to the robot-view or synthetic-view frame.
- `goal_text`: natural-language task description.
- `observation_summary`: compact robot/world state summary.
- `action_candidates`: possible bounded Go2 actions.
- `candidate_scores`: transparent scorer outputs for every candidate.
- `selected_candidate_id`: selected action candidate.
- `label_source`: currently `transparent_heuristic_score`.
- `source_domain`: `synthetic_procedural`, `real_go2_replay`, or real single-frame sources.

## Intended Use

Use this for:

- validating a WorldForge-style decision trace contract,
- scorer-distillation smoke tests,
- candidate-ranking model prototypes,
- debugging dataset pipelines for later learned world models.

Do not use this as evidence that a Go2 navigation policy was trained or validated.

## Synthetic Data Caveat

Synthetic rows are useful because they provide controllable counterfactual scenes:
target left/right/center, close/far, and unsafe colored markers in the path. They
should not be mixed with real rows without keeping `source_domain` as an explicit
feature or split.

## Real Seed Caveat

`real_seed` rows come from hackathon Go2 camera material and scorer traces. They
are small and should be privacy-reviewed before any broad public redistribution.
Labels are transparent scorer labels, not measured long-horizon outcomes.

## Roadmap

The next valuable version would add measured execution outcomes:

```text
candidate action -> actual movement/result -> success/collision/stuck/progress label
```

That would turn this from a scorer-contract dataset into a true learned world-model
or candidate-ranker dataset.
"""
    (output_dir / "README.md").write_text(card)


def build(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and args.clean:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    synthetic = build_synthetic_rows(output_dir, count=args.synthetic_count, seed=args.seed)
    real_rows = build_real_seed_rows(output_dir, max_rows=args.max_real_rows)

    data_dir = output_dir / "data"
    counts: JSON = {}
    for split, rows in synthetic.items():
        for row in rows:
            row["split"] = split
        counts[split] = len(rows)
        write_jsonl(data_dir / f"{split}.jsonl", rows)
    counts["real_seed"] = len(real_rows)
    write_jsonl(data_dir / "real_seed.jsonl", real_rows)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": "worldforge-go2-trace-judge-dataset",
        "row_counts": counts,
        "label_source": "transparent_heuristic_score",
        "source_domains": sorted({row["source_domain"] for rows in synthetic.values() for row in rows} | {row["source_domain"] for row in real_rows}),
        "notes": [
            "Synthetic rows are for scorer/data-pipeline testing.",
            "Real seed rows are small hackathon traces and should be privacy-reviewed before upload.",
            "This is not an imitation policy dataset and not a validated Go2 world model dataset.",
        ],
    }
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_dataset_card(output_dir, counts)
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Hugging Face-ready Go2 decision-trace dataset.")
    parser.add_argument("--output-dir", default="hf_dataset")
    parser.add_argument("--synthetic-count", type=int, default=180)
    parser.add_argument("--max-real-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
