from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


JSON = dict[str, Any]
CANDIDATES = ("turn_left", "turn_right", "forward_small", "stop_capture")


def _read_jsonl(path: Path) -> list[JSON]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _all_rows(dataset_dir: Path) -> list[JSON]:
    rows: list[JSON] = []
    for split in ("train", "validation", "test", "real_seed"):
        for row in _read_jsonl(dataset_dir / "data" / f"{split}.jsonl"):
            row.setdefault("split", split)
            rows.append(row)
    return rows


def _num(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if value is True:
        return 1.0
    if value is False:
        return 0.0
    return float(value)


def _obs_features(row: JSON) -> dict[str, float]:
    observation = row.get("observation_summary", {})
    visual = observation.get("visual_summary", {})
    costmap = observation.get("costmap_summary", {})
    navigation = observation.get("navigation_state", {})
    bearing = _num(visual.get("target_bearing_degrees")) / 45.0
    confidence = _num(visual.get("target_confidence"))
    area = _num(visual.get("target_area_ratio"))
    unsafe = _num(visual.get("unsafe_color_risk"))
    return {
        "target_confidence": confidence,
        "target_bearing_norm": bearing,
        "target_abs_bearing_norm": abs(bearing),
        "target_area_ratio": area,
        "target_area_sqrt": math.sqrt(max(0.0, area)),
        "unsafe_color_risk": unsafe,
        "blocked_ahead": 1.0 if costmap.get("blocked_ahead") else 0.0,
        "localized": 1.0 if navigation.get("localized", True) else 0.0,
        "stuck_probability": _num(navigation.get("stuck_probability"), 0.05),
    }


def _candidate_rows(rows: list[JSON]) -> list[JSON]:
    expanded: list[JSON] = []
    for row in rows:
        obs = _obs_features(row)
        scores = {score["candidate_id"]: score for score in row.get("candidate_scores", [])}
        for candidate in CANDIDATES:
            if candidate not in scores:
                continue
            expanded.append(
                {
                    "row_id": row["row_id"],
                    "split": row.get("split", "train"),
                    "source_domain": row.get("source_domain"),
                    "candidate_id": candidate,
                    "selected": candidate == row.get("selected_candidate_id"),
                    "score": float(scores[candidate]["score"]),
                    "obs": obs,
                }
            )
    return expanded


def _feature_names() -> list[str]:
    names = ["bias"]
    names.extend(f"candidate:{candidate}" for candidate in CANDIDATES)
    obs_names = [
        "target_confidence",
        "target_bearing_norm",
        "target_abs_bearing_norm",
        "target_area_ratio",
        "target_area_sqrt",
        "unsafe_color_risk",
        "blocked_ahead",
        "localized",
        "stuck_probability",
    ]
    names.extend(f"obs:{name}" for name in obs_names)
    for candidate in CANDIDATES:
        for obs in (
            "target_bearing_norm",
            "target_abs_bearing_norm",
            "target_area_sqrt",
            "unsafe_color_risk",
            "blocked_ahead",
        ):
            names.append(f"interaction:{candidate}:{obs}")
    return names


def _vector(item: JSON) -> list[float]:
    obs = item["obs"]
    candidate = item["candidate_id"]
    values: list[float] = [1.0]
    values.extend(1.0 if candidate == candidate_id else 0.0 for candidate_id in CANDIDATES)
    obs_order = [
        "target_confidence",
        "target_bearing_norm",
        "target_abs_bearing_norm",
        "target_area_ratio",
        "target_area_sqrt",
        "unsafe_color_risk",
        "blocked_ahead",
        "localized",
        "stuck_probability",
    ]
    values.extend(float(obs[name]) for name in obs_order)
    for candidate_id in CANDIDATES:
        active = 1.0 if candidate == candidate_id else 0.0
        for obs_name in (
            "target_bearing_norm",
            "target_abs_bearing_norm",
            "target_area_sqrt",
            "unsafe_color_risk",
            "blocked_ahead",
        ):
            values.append(active * float(obs[obs_name]))
    return values


def _ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    identity = np.eye(x.shape[1], dtype=float)
    identity[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + alpha * identity, x.T @ y)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> JSON:
    err = y_pred - y_true
    return {
        "count": int(len(y_true)),
        "mae": round(float(np.mean(np.abs(err))), 6),
        "rmse": round(float(math.sqrt(float(np.mean(err * err)))), 6),
        "r2": round(float(1.0 - (np.sum(err * err) / max(1e-12, np.sum((y_true - y_true.mean()) ** 2)))), 6),
    }


def _selection_accuracy(items: list[JSON], predictions: np.ndarray) -> JSON:
    grouped: dict[str, list[tuple[JSON, float]]] = defaultdict(list)
    for item, pred in zip(items, predictions):
        grouped[str(item["row_id"])].append((item, float(pred)))
    total = 0
    correct = 0
    selected_counts: dict[str, int] = defaultdict(int)
    predicted_counts: dict[str, int] = defaultdict(int)
    for group in grouped.values():
        if not group:
            continue
        gold = max(group, key=lambda pair: 1 if pair[0]["selected"] else 0)[0]["candidate_id"]
        pred_item = max(group, key=lambda pair: pair[1])[0]
        pred = pred_item["candidate_id"]
        selected_counts[str(gold)] += 1
        predicted_counts[str(pred)] += 1
        correct += int(gold == pred)
        total += 1
    return {
        "groups": total,
        "accuracy": round(correct / total, 6) if total else 0.0,
        "selected_counts": dict(sorted(selected_counts.items())),
        "predicted_counts": dict(sorted(predicted_counts.items())),
    }


def _always_candidate_accuracy(items: list[JSON], candidate_id: str) -> float:
    grouped: dict[str, list[JSON]] = defaultdict(list)
    for item in items:
        grouped[str(item["row_id"])].append(item)
    if not grouped:
        return 0.0
    correct = 0
    for group in grouped.values():
        gold = next((item["candidate_id"] for item in group if item["selected"]), None)
        correct += int(gold == candidate_id)
    return round(correct / len(grouped), 6)


def train(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset_dir).resolve()
    rows = _all_rows(dataset_dir)
    items = _candidate_rows(rows)
    if len(items) < 40:
        raise RuntimeError(f"Need at least 40 candidate samples, got {len(items)}")

    train_items = [item for item in items if item["split"] == "train"]
    validation_items = [item for item in items if item["split"] == "validation"]
    test_items = [item for item in items if item["split"] == "test"]
    real_seed_items = [item for item in items if item["split"] == "real_seed"]
    if not train_items:
        raise RuntimeError("No training items found in hf_dataset/data/train.jsonl")

    feature_names = _feature_names()
    x_train = np.asarray([_vector(item) for item in train_items], dtype=float)
    y_train = np.asarray([item["score"] for item in train_items], dtype=float)
    weights = _ridge(x_train, y_train, args.alpha)

    def predict(split_items: list[JSON]) -> np.ndarray:
        if not split_items:
            return np.asarray([], dtype=float)
        x = np.asarray([_vector(item) for item in split_items], dtype=float)
        return x @ weights

    eval_report: JSON = {
        "schema_version": 1,
        "model_type": "micro_world_scorer_ridge_head",
        "claim_boundary": "Small scorer head trained on transparent labels from real Go2 frames and label-safe real-photo-edit counterfactuals. Not a Go2 foundation world model.",
        "dataset_dir": str(dataset_dir),
        "alpha": args.alpha,
        "candidate_ids": list(CANDIDATES),
        "splits": {
            "train_candidate_rows": len(train_items),
            "validation_candidate_rows": len(validation_items),
            "test_candidate_rows": len(test_items),
            "real_seed_candidate_rows": len(real_seed_items),
        },
        "score_metrics": {},
        "selection_metrics": {},
        "baselines": {},
    }
    for name, split_items in (
        ("train", train_items),
        ("validation", validation_items),
        ("test", test_items),
        ("real_seed", real_seed_items),
    ):
        if not split_items:
            continue
        pred = predict(split_items)
        y = np.asarray([item["score"] for item in split_items], dtype=float)
        eval_report["score_metrics"][name] = _metrics(y, pred)
        eval_report["selection_metrics"][name] = _selection_accuracy(split_items, pred)
        eval_report["baselines"][name] = {
            "always_forward_small_accuracy": _always_candidate_accuracy(split_items, "forward_small"),
            "always_turn_left_accuracy": _always_candidate_accuracy(split_items, "turn_left"),
            "random_expected_accuracy": round(1.0 / len(CANDIDATES), 6),
        }

    top_weights = sorted(
        (
            {"feature": feature, "weight": round(float(weight), 8)}
            for feature, weight in zip(feature_names, weights)
        ),
        key=lambda item: abs(float(item["weight"])),
        reverse=True,
    )[:20]
    model = {
        "schema_version": 1,
        "model_type": "go2_cube_micro_world_scorer",
        "feature_names": feature_names,
        "weights": [round(float(weight), 12) for weight in weights],
        "candidate_ids": list(CANDIDATES),
        "alpha": args.alpha,
        "training_summary": eval_report,
        "top_weights": top_weights,
    }

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model.json").write_text(json.dumps(model, indent=2, sort_keys=True) + "\n")
    (output_dir / "eval_report.json").write_text(json.dumps(eval_report, indent=2, sort_keys=True) + "\n")

    sample_items = (test_items or validation_items or train_items)[:80]
    preds = predict(sample_items)
    prediction_rows = [
        {
            "row_id": item["row_id"],
            "candidate_id": item["candidate_id"],
            "selected": item["selected"],
            "label_score": item["score"],
            "predicted_score": round(float(pred), 6),
            "split": item["split"],
        }
        for item, pred in zip(sample_items, preds)
    ]
    (output_dir / "predictions_sample.json").write_text(
        json.dumps(prediction_rows, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(eval_report, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Go2 cube micro world scorer.")
    parser.add_argument("--dataset-dir", default="hf_dataset")
    parser.add_argument("--output-dir", default="artifacts/micro_world_scorer")
    parser.add_argument("--alpha", type=float, default=0.01)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
