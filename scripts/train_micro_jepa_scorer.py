from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from go2_find_colored_target import SCORE_WEIGHTS
from train_micro_world_scorer import CANDIDATES, _all_rows, _obs_features


JSON = dict[str, Any]
LATENT_NAMES = (
    "goal_alignment",
    "information_gain",
    "progress",
    "obstacle_risk",
    "stuck_risk",
    "execution_cost",
)


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


def _vector(candidate_id: str, obs: dict[str, float]) -> list[float]:
    values: list[float] = [1.0]
    values.extend(1.0 if candidate_id == item else 0.0 for item in CANDIDATES)
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
    for item in CANDIDATES:
        active = 1.0 if candidate_id == item else 0.0
        for obs_name in (
            "target_bearing_norm",
            "target_abs_bearing_norm",
            "target_area_sqrt",
            "unsafe_color_risk",
            "blocked_ahead",
        ):
            values.append(active * float(obs[obs_name]))
    return values


def _latent_score(latent: np.ndarray) -> np.ndarray:
    goal = latent[:, LATENT_NAMES.index("goal_alignment")]
    info = latent[:, LATENT_NAMES.index("information_gain")]
    progress = latent[:, LATENT_NAMES.index("progress")]
    obstacle = latent[:, LATENT_NAMES.index("obstacle_risk")]
    stuck = latent[:, LATENT_NAMES.index("stuck_risk")]
    cost = latent[:, LATENT_NAMES.index("execution_cost")]
    return (
        0.10
        + SCORE_WEIGHTS["goal_alignment"] * goal
        + SCORE_WEIGHTS["information_gain"] * info
        + SCORE_WEIGHTS["progress"] * progress
        + SCORE_WEIGHTS["clearance"] * (1.0 - obstacle)
        + SCORE_WEIGHTS["not_stuck"] * (1.0 - stuck)
        + SCORE_WEIGHTS["execution_cost"] * cost
    )


def _rows_to_items(rows: list[JSON]) -> list[JSON]:
    items: list[JSON] = []
    for row in rows:
        obs = _obs_features(row)
        scores = {score["candidate_id"]: score for score in row.get("candidate_scores", [])}
        for candidate in CANDIDATES:
            score_row = scores.get(candidate)
            if not score_row:
                continue
            features = score_row.get("features", {})
            items.append(
                {
                    "row_id": row["row_id"],
                    "split": row.get("split", "train"),
                    "source_domain": row.get("source_domain"),
                    "candidate_id": candidate,
                    "selected": candidate == row.get("selected_candidate_id"),
                    "label_score": float(score_row["score"]),
                    "obs": obs,
                    "latent": [float(features[name]) for name in LATENT_NAMES],
                }
            )
    return items


def _ridge_multi(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    identity = np.eye(x.shape[1], dtype=float)
    identity[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + alpha * identity, x.T @ y)


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> JSON:
    err = y_pred - y_true
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "count": int(len(y_true)),
        "mae": round(float(np.mean(np.abs(err))), 6),
        "rmse": round(float(math.sqrt(float(np.mean(err * err)))), 6),
        "r2": round(float(1.0 - np.sum(err * err) / max(1e-12, denom)), 6),
    }


def _latent_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> JSON:
    return {
        name: _regression_metrics(y_true[:, index], y_pred[:, index])
        for index, name in enumerate(LATENT_NAMES)
    }


def _selection_accuracy(items: list[JSON], scores: np.ndarray) -> JSON:
    grouped: dict[str, list[tuple[JSON, float]]] = defaultdict(list)
    for item, score in zip(items, scores):
        grouped[str(item["row_id"])].append((item, float(score)))
    correct = 0
    total = 0
    selected_counts: dict[str, int] = defaultdict(int)
    predicted_counts: dict[str, int] = defaultdict(int)
    for group in grouped.values():
        gold = next((item["candidate_id"] for item, _ in group if item["selected"]), None)
        pred = max(group, key=lambda pair: pair[1])[0]["candidate_id"]
        if gold is None:
            continue
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


def _split(items: list[JSON], split: str) -> list[JSON]:
    return [item for item in items if item["split"] == split]


def train(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset_dir).resolve()
    rows = _all_rows(dataset_dir)
    items = _rows_to_items(rows)
    train_items = _split(items, "train")
    validation_items = _split(items, "validation")
    test_items = _split(items, "test")
    real_seed_items = _split(items, "real_seed")
    if len(train_items) < 40:
        raise RuntimeError(f"Need at least 40 train candidate samples, got {len(train_items)}")

    feature_names = _feature_names()
    x_train = np.asarray([_vector(item["candidate_id"], item["obs"]) for item in train_items], dtype=float)
    y_train = np.asarray([item["latent"] for item in train_items], dtype=float)
    weights = _ridge_multi(x_train, y_train, args.alpha)

    def predict_latent(split_items: list[JSON]) -> np.ndarray:
        if not split_items:
            return np.zeros((0, len(LATENT_NAMES)), dtype=float)
        x = np.asarray([_vector(item["candidate_id"], item["obs"]) for item in split_items], dtype=float)
        pred = x @ weights
        return np.clip(pred, 0.0, 1.0)

    eval_report: JSON = {
        "schema_version": 1,
        "model_type": "go2_cube_micro_jepa_latent_predictor",
        "architecture": "action_conditioned_latent_ridge_predictor",
        "claim_boundary": "Small JEPA-style latent predictor trained on transparent action-outcome latents from real Go2 frames and label-safe real-photo-edit counterfactuals. Not a trained V-JEPA model or Go2 foundation world model.",
        "dataset_dir": str(dataset_dir),
        "alpha": args.alpha,
        "candidate_ids": list(CANDIDATES),
        "latent_names": list(LATENT_NAMES),
        "splits": {
            "train_candidate_rows": len(train_items),
            "validation_candidate_rows": len(validation_items),
            "test_candidate_rows": len(test_items),
            "real_seed_candidate_rows": len(real_seed_items),
        },
        "latent_metrics": {},
        "score_metrics_from_predicted_latents": {},
        "selection_metrics_from_predicted_latents": {},
        "oracle_latent_selection_metrics": {},
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
        y_latent = np.asarray([item["latent"] for item in split_items], dtype=float)
        pred_latent = predict_latent(split_items)
        pred_scores = _latent_score(pred_latent)
        label_scores = np.asarray([item["label_score"] for item in split_items], dtype=float)
        oracle_scores = _latent_score(y_latent)
        eval_report["latent_metrics"][name] = _latent_metrics(y_latent, pred_latent)
        eval_report["score_metrics_from_predicted_latents"][name] = _regression_metrics(
            label_scores, pred_scores
        )
        eval_report["selection_metrics_from_predicted_latents"][name] = _selection_accuracy(
            split_items, pred_scores
        )
        eval_report["oracle_latent_selection_metrics"][name] = _selection_accuracy(
            split_items, oracle_scores
        )
        eval_report["baselines"][name] = {
            "always_forward_small_accuracy": _always_candidate_accuracy(split_items, "forward_small"),
            "always_turn_left_accuracy": _always_candidate_accuracy(split_items, "turn_left"),
            "random_expected_accuracy": round(1.0 / len(CANDIDATES), 6),
        }

    model = {
        "schema_version": 1,
        "model_type": "go2_cube_micro_jepa_latent_predictor",
        "architecture": "action_conditioned_latent_ridge_predictor",
        "feature_names": feature_names,
        "input_feature_names": feature_names,
        "latent_names": list(LATENT_NAMES),
        "weights": [[round(float(value), 12) for value in row] for row in weights],
        "candidate_ids": list(CANDIDATES),
        "score_weights": SCORE_WEIGHTS,
        "alpha": args.alpha,
        "training_summary": eval_report,
    }

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model.json").write_text(json.dumps(model, indent=2, sort_keys=True) + "\n")
    (output_dir / "eval_report.json").write_text(json.dumps(eval_report, indent=2, sort_keys=True) + "\n")

    sample_items = (test_items or validation_items or train_items)[:80]
    sample_pred_latents = predict_latent(sample_items)
    sample_pred_scores = _latent_score(sample_pred_latents)
    prediction_rows = []
    for item, pred_latent, pred_score in zip(sample_items, sample_pred_latents, sample_pred_scores):
        prediction_rows.append(
            {
                "row_id": item["row_id"],
                "candidate_id": item["candidate_id"],
                "selected": item["selected"],
                "label_score": item["label_score"],
                "predicted_score_from_latent": round(float(pred_score), 6),
                "label_latent": {
                    name: round(float(value), 6)
                    for name, value in zip(LATENT_NAMES, item["latent"])
                },
                "predicted_latent": {
                    name: round(float(value), 6)
                    for name, value in zip(LATENT_NAMES, pred_latent)
                },
                "split": item["split"],
            }
        )
    (output_dir / "predictions_sample.json").write_text(
        json.dumps(prediction_rows, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(eval_report, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a micro JEPA-style latent action scorer.")
    parser.add_argument("--dataset-dir", default="hf_dataset")
    parser.add_argument("--output-dir", default="artifacts/micro_jepa_scorer")
    parser.add_argument("--alpha", type=float, default=0.01)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
