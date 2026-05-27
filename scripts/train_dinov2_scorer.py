from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from train_micro_world_scorer import (
    CANDIDATES,
    _all_rows,
    _feature_names as _geometry_feature_names,
    _obs_features,
    _ridge,
    _vector as _geometry_vector,
)


JSON = dict[str, Any]


def _read_rows(dataset_dir: Path) -> list[JSON]:
    rows = _all_rows(dataset_dir)
    for row in rows:
        row["image_path"] = str(dataset_dir / row["image"])
    return rows


def _candidate_rows(rows: list[JSON]) -> list[JSON]:
    items: list[JSON] = []
    for row in rows:
        obs = _obs_features(row)
        scores = {score["candidate_id"]: score for score in row.get("candidate_scores", [])}
        for candidate in CANDIDATES:
            score = scores.get(candidate)
            if not score:
                continue
            items.append(
                {
                    "row_id": row["row_id"],
                    "split": row.get("split", "train"),
                    "source_domain": row.get("source_domain"),
                    "image_path": row["image_path"],
                    "candidate_id": candidate,
                    "selected": candidate == row.get("selected_candidate_id"),
                    "score": float(score["score"]),
                    "obs": obs,
                }
            )
    return items


def _feature_names(embedding_dim: int) -> list[str]:
    return _geometry_feature_names() + [f"dinov2:{index}" for index in range(embedding_dim)]


def _vector(item: JSON, embeddings: dict[str, list[float]]) -> list[float]:
    image_key = str(item["image_path"])
    values = _geometry_vector(item)
    values.extend(embeddings[image_key])
    return values


def _extract_embeddings(dataset_dir: Path, rows: list[JSON], cache_path: Path, model_name: str, batch_size: int) -> dict[str, list[float]]:
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if cached.get("model_name") == model_name:
            return {str(key): value for key, value in cached["embeddings"].items()}

    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
    except Exception as exc:  # pragma: no cover - dependency guard for hackathon machines.
        raise RuntimeError(
            "DINOv2 scorer requires torch and transformers. Install with: "
            "python3 -m pip install --user torch torchvision transformers safetensors"
        ) from exc

    device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    image_paths = sorted({str(dataset_dir / row["image"]) for row in rows})
    embeddings: dict[str, list[float]] = {}
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        images = [Image.open(path).convert("RGB") for path in batch_paths]
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            pooled = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
        pooled = pooled / np.maximum(1e-12, np.linalg.norm(pooled, axis=1, keepdims=True))
        for path, vec in zip(batch_paths, pooled):
            embeddings[path] = [round(float(value), 8) for value in vec]
        print(f"embedded {min(start + batch_size, len(image_paths))}/{len(image_paths)} images")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "model_name": model_name,
                "embedding_dim": len(next(iter(embeddings.values()))),
                "embeddings": embeddings,
            },
            sort_keys=True,
        )
        + "\n"
    )
    return embeddings


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> JSON:
    err = y_pred - y_true
    return {
        "count": int(len(y_true)),
        "mae": round(float(np.mean(np.abs(err))), 6),
        "rmse": round(float(math.sqrt(float(np.mean(err * err)))), 6),
        "r2": round(float(1.0 - np.sum(err * err) / max(1e-12, np.sum((y_true - y_true.mean()) ** 2))), 6),
    }


def _selection_accuracy(items: list[JSON], pred: np.ndarray) -> JSON:
    grouped: dict[str, list[tuple[JSON, float]]] = defaultdict(list)
    for item, value in zip(items, pred):
        grouped[str(item["row_id"])].append((item, float(value)))
    correct = 0
    selected_counts: dict[str, int] = defaultdict(int)
    predicted_counts: dict[str, int] = defaultdict(int)
    for group in grouped.values():
        gold = next((item["candidate_id"] for item, _ in group if item["selected"]), None)
        guess = max(group, key=lambda pair: pair[1])[0]["candidate_id"]
        selected_counts[str(gold)] += 1
        predicted_counts[str(guess)] += 1
        correct += int(gold == guess)
    total = len(grouped)
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
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(dataset_dir)
    embeddings = _extract_embeddings(
        dataset_dir,
        rows,
        output_dir / "dinov2_embeddings_cache.json",
        args.model_name,
        args.batch_size,
    )
    items = _candidate_rows(rows)
    train_items = [item for item in items if item["split"] == "train"]
    validation_items = [item for item in items if item["split"] == "validation"]
    test_items = [item for item in items if item["split"] == "test"]
    real_seed_items = [item for item in items if item["split"] == "real_seed"]
    if not train_items:
        raise RuntimeError("No training items found")

    embedding_dim = len(next(iter(embeddings.values())))
    feature_names = _feature_names(embedding_dim)
    x_train = np.asarray([_vector(item, embeddings) for item in train_items], dtype=float)
    y_train = np.asarray([item["score"] for item in train_items], dtype=float)
    weights = _ridge(x_train, y_train, args.alpha)

    def predict(split_items: list[JSON]) -> np.ndarray:
        if not split_items:
            return np.asarray([], dtype=float)
        x = np.asarray([_vector(item, embeddings) for item in split_items], dtype=float)
        return x @ weights

    eval_report: JSON = {
        "schema_version": 1,
        "model_type": "dinov2_frozen_embedding_hybrid_action_scorer",
        "claim_boundary": "Frozen DINOv2 visual embeddings plus geometry/action features and a small ridge score head. This is not a fine-tuned foundation world model or validated autonomy policy.",
        "backbone": args.model_name,
        "backbone_frozen": True,
        "candidate_ids": list(CANDIDATES),
        "embedding_dim": embedding_dim,
        "feature_count": len(feature_names),
        "alpha": args.alpha,
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
        labels = np.asarray([item["score"] for item in split_items], dtype=float)
        eval_report["score_metrics"][name] = _metrics(labels, pred)
        eval_report["selection_metrics"][name] = _selection_accuracy(split_items, pred)
        eval_report["baselines"][name] = {
            "always_forward_small_accuracy": _always_candidate_accuracy(split_items, "forward_small"),
            "always_turn_left_accuracy": _always_candidate_accuracy(split_items, "turn_left"),
            "random_expected_accuracy": round(1.0 / len(CANDIDATES), 6),
        }

    model = {
        "schema_version": 1,
        "model_type": "dinov2_frozen_embedding_hybrid_action_scorer",
        "backbone": args.model_name,
        "backbone_frozen": True,
        "feature_names": feature_names,
        "weights": [round(float(value), 12) for value in weights],
        "candidate_ids": list(CANDIDATES),
        "alpha": args.alpha,
        "training_summary": eval_report,
    }
    (output_dir / "model.json").write_text(json.dumps(model, indent=2, sort_keys=True) + "\n")
    (output_dir / "eval_report.json").write_text(json.dumps(eval_report, indent=2, sort_keys=True) + "\n")

    sample_items = (test_items or validation_items or train_items)[:80]
    pred = predict(sample_items)
    rows_out = [
        {
            "row_id": item["row_id"],
            "candidate_id": item["candidate_id"],
            "selected": item["selected"],
            "label_score": item["score"],
            "predicted_score": round(float(value), 6),
            "split": item["split"],
        }
        for item, value in zip(sample_items, pred)
    ]
    (output_dir / "predictions_sample.json").write_text(json.dumps(rows_out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(eval_report, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a frozen-DINOv2 action scorer.")
    parser.add_argument("--dataset-dir", default="hf_dataset")
    parser.add_argument("--output-dir", default="artifacts/dinov2_scorer")
    parser.add_argument("--model-name", default="facebook/dinov2-small")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
