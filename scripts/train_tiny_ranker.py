from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


JSON = dict[str, Any]
CANDIDATE_IDS = ("forward_small", "stop_capture", "turn_left", "turn_right")
FEATURE_KEYS = (
    "goal_alignment",
    "information_gain",
    "progress",
    "obstacle_risk",
    "stuck_risk",
    "execution_cost",
)
OBSERVATION_KEYS = (
    "target_confidence",
    "target_bearing_degrees",
    "target_area_ratio",
    "unsafe_color_risk",
    "stuck_probability",
)


def _load_rows(path: Path) -> list[JSON]:
    if not path.is_file():
        raise RuntimeError(f"Dataset not found: {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) < 20:
        raise RuntimeError(f"Need at least 20 candidate rows for a useful smoke test, got {len(rows)}")
    return rows


def _number(value: Any) -> float:
    if value is None or value is False:
        return 0.0
    if value is True:
        return 1.0
    return float(value)


def _vector(row: JSON) -> list[float]:
    features = row.get("features", {})
    values: list[float] = [1.0]
    values.extend(1.0 if row.get("candidate_id") == candidate_id else 0.0 for candidate_id in CANDIDATE_IDS)
    values.extend(_number(features.get(key)) for key in FEATURE_KEYS)
    values.extend(_number(row.get(key)) for key in OBSERVATION_KEYS)
    values.append(1.0 if row.get("blocked_ahead") else 0.0)
    values.append(1.0 if row.get("localized") else 0.0)
    return values


def _feature_names() -> list[str]:
    return (
        ["bias"]
        + [f"candidate:{candidate_id}" for candidate_id in CANDIDATE_IDS]
        + [f"feature:{key}" for key in FEATURE_KEYS]
        + [f"observation:{key}" for key in OBSERVATION_KEYS]
        + ["observation:blocked_ahead", "observation:localized"]
    )


def _fit_ridge(x_train: np.ndarray, y_train: np.ndarray, alpha: float) -> np.ndarray:
    identity = np.eye(x_train.shape[1])
    identity[0, 0] = 0.0
    return np.linalg.solve(x_train.T @ x_train + alpha * identity, x_train.T @ y_train)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> JSON:
    errors = y_pred - y_true
    mae = float(np.mean(np.abs(errors)))
    rmse = float(math.sqrt(float(np.mean(errors * errors))))
    return {"mae": round(mae, 6), "rmse": round(rmse, 6), "count": int(y_true.shape[0])}


def run(args: argparse.Namespace) -> int:
    dataset_jsonl = Path(args.dataset_jsonl).expanduser()
    rows = _load_rows(dataset_jsonl)
    x = np.asarray([_vector(row) for row in rows], dtype=float)
    y = np.asarray([float(row["transparent_score_label"]) for row in rows], dtype=float)

    # Deterministic split by step index keeps all candidates from the same frame together.
    train_mask = np.asarray([int(row["step_index"]) % 5 != 0 for row in rows], dtype=bool)
    test_mask = ~train_mask
    if int(test_mask.sum()) == 0:
        train_mask[:] = True
        test_mask[-max(4, len(rows) // 5) :] = True
        train_mask[test_mask] = False

    weights = _fit_ridge(x[train_mask], y[train_mask], alpha=args.alpha)
    train_pred = x[train_mask] @ weights
    test_pred = x[test_mask] @ weights

    feature_names = _feature_names()
    top_weights = sorted(
        [
            {"feature": name, "weight": round(float(weight), 6)}
            for name, weight in zip(feature_names, weights)
        ],
        key=lambda item: abs(float(item["weight"])),
        reverse=True,
    )[:10]
    model = {
        "schema_version": 1,
        "model_type": "ridge_regression_transparent_scorer_distillation",
        "dataset_jsonl": str(dataset_jsonl),
        "label": "transparent_score_label",
        "honest_claim": "This tiny ranker distills the transparent scorer from trace artifacts. It is not a learned Go2 world model until labels come from measured outcomes.",
        "alpha": args.alpha,
        "feature_names": feature_names,
        "weights": [round(float(weight), 10) for weight in weights],
        "metrics": {
            "train": _metrics(y[train_mask], train_pred),
            "test": _metrics(y[test_mask], test_pred),
        },
        "top_weights": top_weights,
        "row_count": len(rows),
    }

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model.json").write_text(json.dumps(model, indent=2, sort_keys=True) + "\n")

    prediction_rows = []
    for row, predicted in zip(rows[: min(40, len(rows))], x @ weights):
        prediction_rows.append(
            {
                "step_index": row["step_index"],
                "candidate_id": row["candidate_id"],
                "label": row["transparent_score_label"],
                "prediction": round(float(predicted), 4),
                "selected": row["selected"],
            }
        )
    (output_dir / "predictions_sample.json").write_text(
        json.dumps(prediction_rows, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(model["metrics"], indent=2, sort_keys=True))
    print(output_dir / "model.json")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a tiny ranker smoke test from flattened Go2 trace candidates."
    )
    parser.add_argument("--dataset-jsonl", default="dataset/go2_trace_candidates.jsonl")
    parser.add_argument("--output-dir", default="artifacts/ranker_smoke")
    parser.add_argument("--alpha", type=float, default=0.001)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
