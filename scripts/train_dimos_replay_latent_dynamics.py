from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
JSON = dict[str, Any]


def read_jsonl(path: Path) -> list[JSON]:
    if not path.exists():
        return []
    rows: list[JSON] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_rows(dataset_dir: Path) -> list[JSON]:
    rows: list[JSON] = []
    for split in ("train", "validation", "test"):
        for row in read_jsonl(dataset_dir / "data" / f"{split}.jsonl"):
            row["split"] = split
            rows.append(row)
    if not rows:
        rows = read_jsonl(dataset_dir / "metadata.jsonl")
    return rows


def action_vector(row: JSON) -> list[float]:
    delta = row["egomotion_delta"]
    dyaw = float(delta["dyaw_rad"])
    return [
        float(delta["dx_body_m"]),
        float(delta["dy_body_m"]),
        math.sin(dyaw),
        math.cos(dyaw) - 1.0,
        float(delta["distance_m"]),
        float(row["horizon_s"]),
    ]


def candidate_actions(row: JSON) -> list[JSON]:
    actual = action_vector(row)
    delta = row["egomotion_delta"]
    distance = max(0.05, abs(float(delta["distance_m"])))
    dyaw = max(0.25, abs(float(delta["dyaw_rad"])))
    return [
        {"candidate_id": "actual_egomotion", "action_vector": actual, "selected": True},
        {"candidate_id": "zero_motion", "action_vector": [0.0, 0.0, 0.0, 0.0, 0.0, float(row["horizon_s"])], "selected": False},
        {"candidate_id": "forward_same_distance", "action_vector": [distance, 0.0, 0.0, 0.0, distance, float(row["horizon_s"])], "selected": False},
        {"candidate_id": "rotate_left", "action_vector": [0.0, 0.0, math.sin(dyaw), math.cos(dyaw) - 1.0, 0.0, float(row["horizon_s"])], "selected": False},
        {"candidate_id": "rotate_right", "action_vector": [0.0, 0.0, math.sin(-dyaw), math.cos(-dyaw) - 1.0, 0.0, float(row["horizon_s"])], "selected": False},
        {"candidate_id": "reverse_actual", "action_vector": [-actual[0], -actual[1], -actual[2], actual[3], distance, float(row["horizon_s"])], "selected": False},
    ]


def _normalise_rows(x: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mean is None:
        mean = x.mean(axis=0)
    if std is None:
        std = x.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
    return (x - mean) / std, mean, std


def _ridge_multi(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    xtx = x.T @ x
    reg = np.eye(xtx.shape[0]) * alpha
    reg[-1, -1] = 0.0  # Do not regularize bias.
    return np.linalg.solve(xtx + reg, x.T @ y)


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.maximum(1e-12, np.linalg.norm(a, axis=-1, keepdims=True))
    b_norm = b / np.maximum(1e-12, np.linalg.norm(b, axis=-1, keepdims=True))
    return np.sum(a_norm * b_norm, axis=-1)


def metrics(y_true: np.ndarray, y_pred: np.ndarray, current: np.ndarray) -> JSON:
    err = y_pred - y_true
    cos_pred = _cosine(y_pred, y_true)
    cos_current = _cosine(current, y_true)
    return {
        "count": int(len(y_true)),
        "future_cosine_mean": round(float(cos_pred.mean()), 6),
        "future_cosine_median": round(float(np.median(cos_pred)), 6),
        "no_motion_cosine_mean": round(float(cos_current.mean()), 6),
        "cosine_lift_vs_no_motion": round(float(cos_pred.mean() - cos_current.mean()), 6),
        "latent_mae": round(float(np.mean(np.abs(err))), 6),
        "latent_rmse": round(float(math.sqrt(float(np.mean(err * err)))), 6),
    }


def extract_embeddings(dataset_dir: Path, rows: list[JSON], output_dir: Path, model_name: str, batch_size: int, device_arg: str) -> dict[str, list[float]]:
    cache_path = output_dir / "frame_embeddings_cache.json"
    image_paths = sorted({str(dataset_dir / row[key]) for row in rows for key in ("current_image", "future_image")})
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if cached.get("model_name") == model_name and set(image_paths).issubset(set(cached.get("embeddings", {}))):
            return {str(key): value for key, value in cached["embeddings"].items()}

    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
    except Exception as exc:
        raise RuntimeError(
            "This trainer needs torch and transformers for frozen DINOv2 embeddings."
        ) from exc

    if device_arg == "auto":
        device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    else:
        device = device_arg

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

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
        print(f"embedded {min(start + batch_size, len(image_paths))}/{len(image_paths)} frames")

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


def feature_matrix(rows: list[JSON], embeddings: dict[str, list[float]], dataset_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[float]]]:
    x_parts: list[list[float]] = []
    y_parts: list[list[float]] = []
    current_parts: list[list[float]] = []
    actions: list[list[float]] = []
    for row in rows:
        current_key = str(dataset_dir / row["current_image"])
        future_key = str(dataset_dir / row["future_image"])
        current = embeddings[current_key]
        action = action_vector(row)
        x_parts.append(current + action)
        y_parts.append(embeddings[future_key])
        current_parts.append(current)
        actions.append(action)
    return (
        np.asarray(x_parts, dtype=float),
        np.asarray(y_parts, dtype=float),
        np.asarray(current_parts, dtype=float),
        actions,
    )


def evaluate_selection(rows: list[JSON], embeddings: dict[str, list[float]], dataset_dir: Path, weights: np.ndarray, mean: np.ndarray, std: np.ndarray) -> JSON:
    if not rows:
        return {"groups": 0, "accuracy": 0.0}
    correct = 0
    margins: list[float] = []
    examples: list[JSON] = []
    candidate_counts: dict[str, int] = {}
    for row in rows:
        current = embeddings[str(dataset_dir / row["current_image"])]
        current_latent = np.asarray(current, dtype=float).reshape(1, -1)
        goal = np.asarray(embeddings[str(dataset_dir / row["future_image"])], dtype=float)
        scored: list[tuple[str, float, bool]] = []
        for candidate in candidate_actions(row):
            raw = np.asarray([current + candidate["action_vector"]], dtype=float)
            x_norm = (raw - mean) / std
            x_aug = np.concatenate([x_norm, np.ones((1, 1))], axis=1)
            pred = current_latent + x_aug @ weights
            score = float(_cosine(pred, goal.reshape(1, -1))[0])
            scored.append((candidate["candidate_id"], score, bool(candidate["selected"])))
        scored_sorted = sorted(scored, key=lambda item: item[1], reverse=True)
        guess = scored_sorted[0][0]
        candidate_counts[guess] = candidate_counts.get(guess, 0) + 1
        correct += int(guess == "actual_egomotion")
        actual_score = next(score for candidate_id, score, _ in scored if candidate_id == "actual_egomotion")
        best_decoy = max(score for candidate_id, score, _ in scored if candidate_id != "actual_egomotion")
        margins.append(actual_score - best_decoy)
        if len(examples) < 12:
            examples.append(
                {
                    "row_id": row["row_id"],
                    "split": row["split"],
                    "scores": [
                        {"candidate_id": candidate_id, "score": round(score, 6), "selected": selected}
                        for candidate_id, score, selected in scored_sorted
                    ],
                }
            )
    return {
        "groups": len(rows),
        "accuracy": round(correct / len(rows), 6),
        "actual_vs_best_decoy_margin_mean": round(float(np.mean(margins)), 6),
        "actual_vs_best_decoy_margin_median": round(float(np.median(margins)), 6),
        "predicted_counts": dict(sorted(candidate_counts.items())),
        "examples": examples,
    }


def train(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(dataset_dir)
    if not rows:
        raise RuntimeError(f"No rows found in {dataset_dir}")
    embeddings = extract_embeddings(dataset_dir, rows, output_dir, args.model_name, args.batch_size, args.device)
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    test_rows = [row for row in rows if row["split"] == "test"]
    if len(train_rows) < 20:
        raise RuntimeError(f"Need at least 20 train pairs, got {len(train_rows)}")

    x_train_raw, y_train, current_train, _ = feature_matrix(train_rows, embeddings, dataset_dir)
    x_train, mean, std = _normalise_rows(x_train_raw)
    x_train_aug = np.concatenate([x_train, np.ones((len(x_train), 1))], axis=1)
    # Slow robot-view video is dominated by visual persistence. Predict the
    # action-conditioned residual instead of relearning the full future latent.
    weights = _ridge_multi(x_train_aug, y_train - current_train, args.alpha)

    def predict(split_rows: list[JSON]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_raw, y, current, _ = feature_matrix(split_rows, embeddings, dataset_dir)
        x_norm = (x_raw - mean) / std
        x_aug = np.concatenate([x_norm, np.ones((len(x_norm), 1))], axis=1)
        pred = current + x_aug @ weights
        return pred, y, current

    eval_report: JSON = {
        "schema_version": 1,
        "model_type": "dimos_go2_replay_latent_dynamics_head",
        "architecture": "frozen_dinov2_current_latent_plus_egomotion_to_future_latent_residual_ridge_head",
        "claim_boundary": "Fine-tunes/trains only a small latent dynamics head on DimOS Go2 replay-derived pairs. The DINOv2 backbone is frozen; this is not a trained V-JEPA or Go2 foundation world model.",
        "dataset_dir": str(dataset_dir),
        "backbone": args.model_name,
        "backbone_frozen": True,
        "alpha": args.alpha,
        "input_schema": {
            "current_latent": "frozen DINOv2 CLS embedding of current robot-view frame",
            "action": ["dx_body_m", "dy_body_m", "sin_dyaw", "cos_dyaw_minus_1", "distance_m", "horizon_s"],
            "target": "future-current residual in frozen DINOv2 CLS latent space",
        },
        "splits": {
            "train_pairs": len(train_rows),
            "validation_pairs": len(validation_rows),
            "test_pairs": len(test_rows),
        },
        "latent_prediction_metrics": {},
        "worldforge_candidate_scoring_eval": {},
    }

    for split, split_rows in (("train", train_rows), ("validation", validation_rows), ("test", test_rows)):
        if not split_rows:
            continue
        pred, y, current = predict(split_rows)
        eval_report["latent_prediction_metrics"][split] = metrics(y, pred, current)
        eval_report["worldforge_candidate_scoring_eval"][split] = evaluate_selection(
            split_rows, embeddings, dataset_dir, weights, mean, std
        )

    model = {
        "schema_version": 1,
        "model_type": eval_report["model_type"],
        "architecture": eval_report["architecture"],
        "claim_boundary": eval_report["claim_boundary"],
        "backbone": args.model_name,
        "backbone_frozen": True,
        "feature_names": [f"current_latent:{i}" for i in range(y_train.shape[1])]
        + ["dx_body_m", "dy_body_m", "sin_dyaw", "cos_dyaw_minus_1", "distance_m", "horizon_s", "bias"],
        "normalization": {
            "mean": [round(float(value), 10) for value in mean],
            "std": [round(float(value), 10) for value in std],
        },
        "weights_shape": list(weights.shape),
        "weights": [[round(float(value), 10) for value in row] for row in weights],
        "training_summary": eval_report,
    }
    (output_dir / "model.json").write_text(json.dumps(model, indent=2, sort_keys=True) + "\n")
    (output_dir / "eval_report.json").write_text(json.dumps(eval_report, indent=2, sort_keys=True) + "\n")

    sample_rows = (test_rows or validation_rows or train_rows)[:20]
    selection = evaluate_selection(sample_rows, embeddings, dataset_dir, weights, mean, std)
    (output_dir / "candidate_scores_sample.json").write_text(json.dumps(selection["examples"], indent=2, sort_keys=True) + "\n")
    (output_dir / "README.md").write_text(model_card(eval_report), encoding="utf-8")
    print(json.dumps(eval_report, indent=2, sort_keys=True))


def model_card(eval_report: JSON) -> str:
    test = eval_report["latent_prediction_metrics"].get("test", {})
    scoring = eval_report["worldforge_candidate_scoring_eval"].get("test", {})
    return f"""---
license: apache-2.0
library_name: transformers
tags:
- unitree-go2
- dimos
- world-model
- latent-dynamics
- dinov2
- worldforge
---

# Go2 DimOS Replay Latent Dynamics Head

This is an experimental WorldForge-style world-model head trained on the derived
`WorldForge Go2 DimOS Replay World Pairs` dataset.

## What Was Trained

Only a small ridge dynamics head was trained:

```text
frozen DINOv2 current-frame latent + egomotion/action delta
-> predicted residual future DINOv2 latent
-> current latent + residual = predicted future DINOv2 latent
```

The DINOv2 backbone remains frozen. This is not a trained V-JEPA model, not a
Go2 foundation model, and not a safety-certified controller.

## Evaluation

- Test future-latent cosine mean: `{test.get("future_cosine_mean", "n/a")}`
- Test no-motion cosine baseline: `{test.get("no_motion_cosine_mean", "n/a")}`
- Test cosine lift vs no-motion: `{test.get("cosine_lift_vs_no_motion", "n/a")}`
- Test candidate scoring accuracy: `{scoring.get("accuracy", "n/a")}`

Candidate scoring uses the WorldForge-style contract:

```text
score(candidate) =
  cosine(predicted_future_latent(current_image, candidate_delta), goal_future_latent)
```

For evaluation, the real future frame provides the goal latent and the real
egomotion delta is ranked against counterfactual deltas.

## Limitations

- Tiny replay-derived dataset.
- Egomotion labels come from pose deltas, not raw joystick commands.
- The model is intended as an inspectable research/demo artifact.
- It should not be used for direct robot control or safety decisions.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DimOS replay-derived latent dynamics head.")
    parser.add_argument("--dataset-dir", default="hf_dataset_dimos_replay")
    parser.add_argument("--output-dir", default="artifacts/dimos_replay_latent_dynamics")
    parser.add_argument("--model-name", default="facebook/dinov2-small")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
