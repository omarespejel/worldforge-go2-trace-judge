from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from textwrap import wrap
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_dimos_replay_latent_dynamics import (  # noqa: E402
    _cosine,
    _gelu,
    candidate_actions,
    extract_embeddings,
    load_rows,
)


JSON = dict[str, Any]
WIDTH = 1920
HEIGHT = 1080
WHITE = (250, 250, 247)
INK = (15, 18, 22)
MUTED = (83, 91, 101)
FAINT = (225, 228, 231)
GREEN = (0, 145, 97)
BLUE = (24, 95, 191)
RED = (190, 48, 48)


def font(size: int, *, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    if mono:
        candidates = ["/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Monaco.ttf"]
    elif bold:
        candidates = ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/Library/Fonts/Arial Bold.ttf"]
    else:
        candidates = ["/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf"]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


F_TITLE = font(64, bold=True)
F_H2 = font(36, bold=True)
F_BODY = font(27)
F_SMALL = font(21)
F_MONO = font(20, mono=True)


def read_json(path: Path) -> JSON:
    return json.loads(path.read_text())


def load_predictor(model: JSON):
    mean = np.asarray(model["normalization"]["mean"], dtype=float)
    std = np.asarray(model["normalization"]["std"], dtype=float)
    head = model.get("head", {})

    if head.get("head_type") == "ridge":
        weights = np.asarray(head["weights"], dtype=float)

        def predict_residual(raw_x: np.ndarray) -> np.ndarray:
            x_norm = (raw_x - mean) / std
            x_aug = np.concatenate([x_norm, np.ones((len(x_norm), 1))], axis=1)
            return x_aug @ weights

        return predict_residual

    if head.get("head_type") == "mlp":
        layers = head["layers"]
        w1 = np.asarray(layers["linear1_weight"], dtype=float)
        b1 = np.asarray(layers["linear1_bias"], dtype=float)
        w2 = np.asarray(layers["linear2_weight"], dtype=float)
        b2 = np.asarray(layers["linear2_bias"], dtype=float)

        def predict_residual(raw_x: np.ndarray) -> np.ndarray:
            x_norm = (raw_x - mean) / std
            hidden = _gelu(x_norm @ w1 + b1)
            return hidden @ w2 + b2

        return predict_residual

    raise RuntimeError(f"Unsupported model head: {head.get('head_type')!r}")


def score_row(row: JSON, embeddings: dict[str, list[float]], dataset_dir: Path, predict_residual) -> JSON:
    current = np.asarray(embeddings[str(dataset_dir / row["current_image"])], dtype=float)
    goal = np.asarray(embeddings[str(dataset_dir / row["future_image"])], dtype=float)
    scored: list[JSON] = []
    for candidate in candidate_actions(row):
        raw = np.asarray([current.tolist() + candidate["action_vector"]], dtype=float)
        predicted = current.reshape(1, -1) + predict_residual(raw)
        score = float(_cosine(predicted, goal.reshape(1, -1))[0])
        scored.append(
            {
                "candidate_id": candidate["candidate_id"],
                "score": round(score, 6),
                "selected_in_replay": bool(candidate["selected"]),
                "action_vector": [round(float(value), 6) for value in candidate["action_vector"]],
            }
        )
    scored = sorted(scored, key=lambda item: item["score"], reverse=True)
    actual = next(item for item in scored if item["candidate_id"] == "actual_egomotion")
    best_decoy = max(item["score"] for item in scored if item["candidate_id"] != "actual_egomotion")
    return {
        "row": row,
        "candidate_scores": scored,
        "selected": scored[0],
        "actual_margin": round(float(actual["score"] - best_decoy), 6),
        "selected_matches_replay": scored[0]["candidate_id"] == "actual_egomotion",
    }


def choose_demo(scores: list[JSON]) -> JSON:
    good = [item for item in scores if item["selected_matches_replay"] and item["actual_margin"] > 0]
    if good:
        return sorted(good, key=lambda item: item["actual_margin"], reverse=True)[0]
    return sorted(scores, key=lambda item: item["actual_margin"], reverse=True)[0]


def fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    src = Image.open(path).convert("RGB")
    scale = max(size[0] / src.width, size[1] / src.height)
    resized = src.resize((max(1, int(src.width * scale)), max(1, int(src.height * scale))), Image.Resampling.LANCZOS)
    left = max(0, (resized.width - size[0]) // 2)
    top = max(0, (resized.height - size[1]) // 2)
    return resized.crop((left, top, left + size[0], top + size[1]))


def draw_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    draw.rounded_rectangle(box, radius=12, fill=(255, 255, 255), outline=INK, width=2)


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, fnt, fill=INK) -> None:
    draw.text(xy, value, font=fnt, fill=fill)


def render_summary_image(result: JSON, dataset_dir: Path, model: JSON, output_dir: Path) -> Path:
    row = result["row"]
    out = Image.new("RGB", (WIDTH, HEIGHT), WHITE)
    draw = ImageDraw.Draw(out)
    for x in range(80, WIDTH, 160):
        draw.line((x, 0, x, HEIGHT), fill=(241, 242, 243), width=1)
    draw.rectangle((0, 0, WIDTH, 12), fill=INK)
    draw_text(draw, (78, 56), "Replay-MPC: a Go2 world model scores candidate futures", F_TITLE)
    draw_text(draw, (82, 132), "current frame + candidate action -> predicted future latent -> score", F_MONO, GREEN)

    current_box = (82, 218, 812, 668)
    future_box = (852, 218, 1582, 668)
    draw_panel(draw, current_box)
    draw_panel(draw, future_box)
    out.paste(fit_image(dataset_dir / row["current_image"], (730, 450)), (82, 218))
    out.paste(fit_image(dataset_dir / row["future_image"], (730, 450)), (852, 218))
    draw.rounded_rectangle(current_box, radius=12, outline=INK, width=2)
    draw.rounded_rectangle(future_box, radius=12, outline=INK, width=2)
    draw_text(draw, (92, 682), "current robot view", F_SMALL, MUTED)
    draw_text(draw, (862, 682), "actual future replay frame", F_SMALL, MUTED)

    right_box = (1618, 218, 1838, 668)
    draw_panel(draw, right_box)
    draw_text(draw, (1642, 250), "selected", F_MONO, MUTED)
    selected = result["selected"]["candidate_id"]
    selected_lines = wrap(selected.replace("_", " "), 13)
    y = 298
    for line in selected_lines:
        draw_text(draw, (1642, y), line, F_H2, GREEN if result["selected_matches_replay"] else RED)
        y += 42
    draw_text(draw, (1642, 450), "score", F_MONO, MUTED)
    draw_text(draw, (1642, 492), f"{result['selected']['score']:.4f}", F_H2, INK)
    draw_text(draw, (1642, 570), "margin", F_MONO, MUTED)
    draw_text(draw, (1642, 612), f"{result['actual_margin']:+.4f}", F_H2, GREEN if result["actual_margin"] > 0 else RED)

    scores = result["candidate_scores"][:6]
    max_score = max(item["score"] for item in scores)
    min_score = min(item["score"] for item in scores)
    span = max(1e-6, max_score - min_score)
    y = 765
    draw_text(draw, (82, 728), "candidate futures ranked by latent goal similarity", F_H2)
    for item in scores:
        label = item["candidate_id"].replace("_", " ")
        color = GREEN if item["candidate_id"] == result["selected"]["candidate_id"] else BLUE
        bar = int(720 * ((item["score"] - min_score) / span))
        draw_text(draw, (92, y), label, F_MONO, color)
        draw.rectangle((430, y + 4, 430 + bar, y + 26), fill=color)
        draw.rectangle((430, y + 4, 1150, y + 26), outline=FAINT, width=1)
        draw_text(draw, (1180, y - 2), f"{item['score']:.4f}", F_MONO, INK)
        if item["selected_in_replay"]:
            draw_text(draw, (1320, y - 2), "observed replay action", F_MONO, MUTED)
        y += 45

    footer = (
        f"source={row['source_dataset']} row={row['row_id']} | "
        f"backbone={model['backbone']} head={model['head']['head_type']} | "
        "WorldForge trace JSON written next to this image"
    )
    draw_text(draw, (82, 1025), footer, F_MONO, MUTED)
    output = output_dir / "predicted_vs_actual_future.jpg"
    out.save(output, quality=92)
    return output


def write_video(image_path: Path, output_dir: Path) -> Path:
    out = output_dir / "replay_mpc_demo.mp4"
    tmp_pattern = output_dir / "replay_mpc_frame_%04d.jpg"
    for index in range(1, 217):
        shutil.copyfile(image_path, Path(str(tmp_pattern) % index))
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            "24",
            "-i",
            str(tmp_pattern),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for path in output_dir.glob("replay_mpc_frame_*.jpg"):
        path.unlink()
    return out


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(dataset_dir)
    if args.split:
        rows = [row for row in rows if row.get("split") == args.split]
    if not rows:
        raise RuntimeError("No dataset rows found for replay-MPC demo.")

    model = read_json(model_dir / "model.json")
    embeddings = extract_embeddings(dataset_dir, rows, output_dir, model["backbone"], args.batch_size, args.device)
    predict_residual = load_predictor(model)
    scored = [score_row(row, embeddings, dataset_dir, predict_residual) for row in rows]
    result = choose_demo(scored)

    image_path = render_summary_image(result, dataset_dir, model, output_dir)
    video_path = write_video(image_path, output_dir)

    row = result["row"]
    score_info = {
        "schema_version": 1,
        "run_id": args.run_id,
        "demo_type": "replay_mpc_world_model_score_demo",
        "world_model": {
            "type": "small_action_conditioned_latent_world_model",
            "backbone": model["backbone"],
            "head": model["head"]["head_type"],
            "claim": "Predicts future visual embeddings from current Go2 replay frame and candidate egomotion.",
        },
        "observation": {
            "source_dataset": row["source_dataset"],
            "row_id": row["row_id"],
            "current_image": row["current_image"],
            "current_pose": row["current_pose"],
        },
        "goal": {
            "type": "held_out_replay_future_frame",
            "future_image": row["future_image"],
            "future_pose": row["future_pose"],
        },
        "score_contract": "score(candidate)=cosine(predicted_future_latent(current_image,candidate_delta), goal_future_latent)",
    }
    candidate_scores = {
        "schema_version": 1,
        "run_id": args.run_id,
        "scores": result["candidate_scores"],
    }
    selected_action = {
        "schema_version": 1,
        "run_id": args.run_id,
        "selected": result["selected"],
        "selected_matches_observed_replay_action": result["selected_matches_replay"],
        "actual_vs_best_decoy_margin": result["actual_margin"],
    }
    outcome = {
        "schema_version": 1,
        "run_id": args.run_id,
        "outcome_type": "offline_replay_eval",
        "actual_future_frame": row["future_image"],
        "selected_matches_observed_replay_action": result["selected_matches_replay"],
        "note": "Offline replay evaluation uses the actual future frame as the visual goal for auditing the world-model scorer.",
    }
    manifest = {
        "schema_version": 1,
        "run_id": args.run_id,
        "artifacts": {
            "summary_image": str(image_path),
            "video": str(video_path),
            "score_info": str(output_dir / "score_info.json"),
            "candidate_scores": str(output_dir / "candidate_scores.json"),
            "selected_action": str(output_dir / "selected_action.json"),
            "outcome_after_execution": str(output_dir / "outcome_after_execution.json"),
        },
    }
    for name, payload in (
        ("score_info.json", score_info),
        ("candidate_scores.json", candidate_scores),
        ("selected_action.json", selected_action),
        ("outcome_after_execution.json", outcome),
        ("run_manifest.json", manifest),
    ):
        (output_dir / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a no-robot replay-MPC demo from the latent world model.")
    parser.add_argument("--dataset-dir", default="hf_dataset_dimos_replay")
    parser.add_argument("--model-dir", default="hf_model_dimos_replay_latent")
    parser.add_argument("--output-dir", default="artifacts/replay_mpc_demo")
    parser.add_argument("--run-id", default="replay-mpc-demo")
    parser.add_argument("--split", default="test", choices=["", "train", "validation", "test"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
