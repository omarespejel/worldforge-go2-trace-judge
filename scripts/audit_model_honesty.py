from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np

from train_micro_world_scorer import (
    CANDIDATES,
    _all_rows,
    _candidate_rows,
    _feature_names,
    _ridge,
    _vector,
)


JSON = dict[str, Any]


def _predict(items: list[JSON], weights: np.ndarray) -> np.ndarray:
    if not items:
        return np.asarray([], dtype=float)
    x = np.asarray([_vector(item) for item in items], dtype=float)
    return x @ weights


def _selection_accuracy(items: list[JSON], predictions: np.ndarray) -> float:
    grouped: dict[str, list[tuple[JSON, float]]] = defaultdict(list)
    for item, pred in zip(items, predictions):
        grouped[str(item["row_id"])].append((item, float(pred)))
    if not grouped:
        return 0.0
    correct = 0
    for group in grouped.values():
        gold = next((item["candidate_id"] for item, _ in group if item["selected"]), None)
        pred = max(group, key=lambda pair: pair[1])[0]["candidate_id"]
        correct += int(gold == pred)
    return correct / len(grouped)


def _score_metrics(items: list[JSON], predictions: np.ndarray) -> JSON:
    if not items:
        return {"count": 0, "mae": None, "rmse": None, "r2": None}
    y = np.asarray([item["score"] for item in items], dtype=float)
    err = predictions - y
    denom = float(np.sum((y - y.mean()) ** 2))
    return {
        "count": int(len(items)),
        "mae": round(float(np.mean(np.abs(err))), 6),
        "rmse": round(float(math.sqrt(float(np.mean(err * err)))), 6),
        "r2": round(float(1.0 - np.sum(err * err) / max(1e-12, denom)), 6),
    }


def _fit(train_items: list[JSON], labels: np.ndarray | None = None, alpha: float = 0.01) -> np.ndarray:
    x = np.asarray([_vector(item) for item in train_items], dtype=float)
    y = labels if labels is not None else np.asarray([item["score"] for item in train_items], dtype=float)
    return _ridge(x, y, alpha)


def _group_source(row: JSON) -> str:
    config = row.get("synthetic_config") or {}
    return str(config.get("base_plate") or row.get("source_domain") or "unknown")


def _items_for_rows(rows: list[JSON]) -> list[JSON]:
    items = _candidate_rows(rows)
    source_by_row = {row["row_id"]: _group_source(row) for row in rows}
    for item in items:
        item["source_group"] = source_by_row.get(item["row_id"], "unknown")
    return items


def _summarize(values: list[float]) -> JSON:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": round(mean(values), 6),
        "std": round(pstdev(values), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def _write_markdown(path: Path, report: JSON) -> None:
    shuffled = report["shuffled_label_control"]
    plate_rows = report["plate_holdout"]
    lines = [
        "# Model Honesty Audit",
        "",
        "This audit is intentionally adversarial. Its purpose is to prevent the",
        "hackathon model from being oversold as a real Go2 foundation world model.",
        "",
        "## Claim Boundary",
        "",
        report["claim_boundary"],
        "",
        "## Main Scorer",
        "",
        f"- test selected-action accuracy: `{report['main_model']['test_selection_accuracy']:.4f}`",
        f"- test MAE: `{report['main_model']['test_score_metrics']['mae']}`",
        f"- test R2: `{report['main_model']['test_score_metrics']['r2']}`",
        "",
        "## Shuffled-Label Control",
        "",
        "A model trained on randomly shuffled training labels should collapse toward",
        "the simple baselines. If it does not, the evaluation is leaking too much.",
        "",
        f"- repeats: `{shuffled['repeats']}`",
        f"- mean test selected-action accuracy: `{shuffled['test_selection_accuracy']['mean']}`",
        f"- min/max: `{shuffled['test_selection_accuracy']['min']}` / `{shuffled['test_selection_accuracy']['max']}`",
        f"- random baseline: `{report['baselines']['random_expected_accuracy']}`",
        f"- always-forward baseline: `{report['baselines']['always_forward_small_accuracy']}`",
        "",
        "## Plate Holdout",
        "",
        "Each row below trains on every real-photo plate except one, then evaluates on",
        "the held-out plate. This is more honest than a random row split because it",
        "tests whether the scorer survives a new camera background.",
        "",
        "| held-out group | groups | selection acc | MAE | R2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in plate_rows:
        lines.append(
            f"| `{item['heldout_group']}` | {item['decision_groups']} | "
            f"{item['selection_accuracy']:.4f} | {item['score_metrics']['mae']} | {item['score_metrics']['r2']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This model is a useful score-provider smoke test and trace distillation.",
            "It is not evidence of learned long-horizon robot control.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def run(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset_dir).resolve()
    rows = _all_rows(dataset_dir)
    items = _items_for_rows(rows)
    train_items = [item for item in items if item["split"] == "train"]
    test_items = [item for item in items if item["split"] == "test"]
    if not train_items or not test_items:
        raise RuntimeError("Need train and test items in the HF dataset")

    main_weights = _fit(train_items, alpha=args.alpha)
    main_test_predictions = _predict(test_items, main_weights)

    rng = random.Random(args.seed)
    shuffled_accuracies: list[float] = []
    shuffled_r2: list[float] = []
    train_labels = [float(item["score"]) for item in train_items]
    for _ in range(args.repeats):
        labels = train_labels[:]
        rng.shuffle(labels)
        weights = _fit(train_items, np.asarray(labels, dtype=float), alpha=args.alpha)
        pred = _predict(test_items, weights)
        shuffled_accuracies.append(_selection_accuracy(test_items, pred))
        shuffled_r2.append(float(_score_metrics(test_items, pred)["r2"]))

    all_photo_items = [item for item in items if item.get("source_domain") == "real_photo_edit"]
    groups = sorted({str(item["source_group"]) for item in all_photo_items})
    plate_holdout = []
    for group in groups:
        holdout = [item for item in all_photo_items if item["source_group"] == group]
        train = [item for item in all_photo_items if item["source_group"] != group]
        if len({item["row_id"] for item in holdout}) < args.min_holdout_groups or not train:
            continue
        weights = _fit(train, alpha=args.alpha)
        pred = _predict(holdout, weights)
        plate_holdout.append(
            {
                "heldout_group": group,
                "candidate_rows": len(holdout),
                "decision_groups": len({item["row_id"] for item in holdout}),
                "selection_accuracy": round(_selection_accuracy(holdout, pred), 6),
                "score_metrics": _score_metrics(holdout, pred),
            }
        )

    grouped_test: dict[str, list[JSON]] = defaultdict(list)
    for item in test_items:
        grouped_test[item["row_id"]].append(item)
    always_forward = 0
    for group in grouped_test.values():
        gold = next((item["candidate_id"] for item in group if item["selected"]), None)
        always_forward += int(gold == "forward_small")

    report: JSON = {
        "schema_version": 1,
        "claim_boundary": "The model distills transparent score labels from real Go2 frames and label-safe counterfactuals. It is not a Go2 foundation world model or validated autonomy policy.",
        "feature_count": len(_feature_names()),
        "main_model": {
            "test_selection_accuracy": round(_selection_accuracy(test_items, main_test_predictions), 6),
            "test_score_metrics": _score_metrics(test_items, main_test_predictions),
        },
        "baselines": {
            "random_expected_accuracy": round(1.0 / len(CANDIDATES), 6),
            "always_forward_small_accuracy": round(always_forward / len(grouped_test), 6),
        },
        "shuffled_label_control": {
            "repeats": args.repeats,
            "test_selection_accuracy": _summarize(shuffled_accuracies),
            "test_r2": _summarize(shuffled_r2),
        },
        "plate_holdout": plate_holdout,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "honesty_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_markdown(output_dir / "honesty_report.md", report)
    print(json.dumps(report, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit scorer overclaiming risk.")
    parser.add_argument("--dataset-dir", default="hf_dataset")
    parser.add_argument("--output-dir", default="artifacts/model_audit")
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--min-holdout-groups", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
