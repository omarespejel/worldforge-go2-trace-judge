from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
from pathlib import Path
from textwrap import wrap
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from go2_find_colored_target import (
    SCORE_WEIGHTS,
    TARGET_COLORS,
    _build_candidates,
    _detect_scene,
    _observation_summary,
    _score,
)


JSON = dict[str, Any]
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = ROOT / "artifacts" / "live_ciro" / "direct_camera_unsafe_path.jpg"
DEFAULT_MODEL = ROOT / "artifacts" / "micro_world_scorer" / "model.json"
OUT_ROOT = ROOT / "artifacts" / "micro_world_demo"
FPS = 24
WIDTH = 1280
HEIGHT = 720

BG = (9, 12, 18)
PANEL = (18, 25, 36)
PANEL_2 = (25, 34, 48)
TEXT = (238, 242, 247)
MUTED = (168, 179, 196)
GREEN = (76, 218, 151)
BLUE = (82, 155, 255)
YELLOW = (250, 207, 82)
RED = (255, 94, 94)
LINE = (57, 71, 92)


def _font(size: int, *, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    if mono:
        candidates = [
            "/System/Library/Fonts/Monaco.ttf",
            "/System/Library/Fonts/Menlo.ttc",
            "/Library/Fonts/Menlo.ttc",
        ]
    elif bold:
        candidates = [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf",
        ]
    else:
        candidates = [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


F_TITLE = _font(42, bold=True)
F_H2 = _font(28, bold=True)
F_BODY = _font(22)
F_SMALL = _font(18)
F_MONO = _font(16, mono=True)


def _write_json(path: Path, data: JSON) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _load_json(path: Path) -> JSON:
    return json.loads(path.read_text())


def _num(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if value is True:
        return 1.0
    if value is False:
        return 0.0
    return float(value)


def _obs_features(observation: JSON) -> dict[str, float]:
    visual = observation.get("visual_summary", {})
    costmap = observation.get("costmap_summary", {})
    navigation = observation.get("navigation_state", {})
    bearing = _num(visual.get("target_bearing_degrees")) / 45.0
    confidence = _num(visual.get("target_confidence"))
    area = _num(visual.get("target_area_ratio"))
    return {
        "target_confidence": confidence,
        "target_bearing_norm": bearing,
        "target_abs_bearing_norm": abs(bearing),
        "target_area_ratio": area,
        "target_area_sqrt": math.sqrt(max(0.0, area)),
        "unsafe_color_risk": _num(visual.get("unsafe_color_risk")),
        "blocked_ahead": 1.0 if costmap.get("blocked_ahead") else 0.0,
        "localized": 1.0 if navigation.get("localized", True) else 0.0,
        "stuck_probability": _num(navigation.get("stuck_probability"), 0.05),
    }


def _model_vector(candidate_id: str, obs: dict[str, float], model: JSON) -> np.ndarray:
    candidate_ids = model["candidate_ids"]
    values: list[float] = [1.0]
    values.extend(1.0 if candidate_id == item else 0.0 for item in candidate_ids)
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
    for item in candidate_ids:
        active = 1.0 if candidate_id == item else 0.0
        for obs_name in (
            "target_bearing_norm",
            "target_abs_bearing_norm",
            "target_area_sqrt",
            "unsafe_color_risk",
            "blocked_ahead",
        ):
            values.append(active * float(obs[obs_name]))
    if len(values) != len(model["feature_names"]):
        raise RuntimeError(
            f"Feature length mismatch: produced {len(values)} values for {len(model['feature_names'])} model features"
        )
    return np.asarray(values, dtype=float)


def _predict_scores(model: JSON, observation: JSON, candidates: list[JSON]) -> list[float]:
    weights = np.asarray(model["weights"], dtype=float)
    obs = _obs_features(observation)
    scores = []
    for candidate in candidates:
        raw_score = float(_model_vector(str(candidate["id"]), obs, model) @ weights)
        scores.append(round(max(0.0, min(1.0, raw_score)), 4))
    return scores


def _reason(scene: JSON, candidate: JSON, selected_id: str, model_score: float) -> str:
    target = scene["target"]
    cid = str(candidate["id"])
    if cid == selected_id:
        return "selected by learned micro world scorer"
    if not target["found"]:
        return "lower score while target is not localized"
    if cid == "forward_small" and float(scene["unsafe_risk"]) > 0.35:
        return "rejected because unsafe marker raises path risk"
    if cid in ("turn_left", "turn_right"):
        direction = "left" if float(target["center_x"]) < 0 else "right"
        if cid.endswith(direction):
            return "useful turn, but not top model score"
        return "turns away from target bearing"
    if cid == "stop_capture":
        return "not close and centered enough for success"
    return f"model score {model_score:.3f}"


def _fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    src = Image.open(path).convert("RGB")
    sw, sh = src.size
    bw, bh = size
    scale = min(bw / sw, bh / sh)
    resized = src.resize((max(1, int(sw * scale)), max(1, int(sh * scale))), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (5, 7, 11))
    canvas.paste(resized, ((bw - resized.width) // 2, (bh - resized.height) // 2))
    return canvas


def _draw_detection_boxes(draw: ImageDraw.ImageDraw, scene: JSON, scale_x: float, scale_y: float, ox: int, oy: int) -> None:
    colors = {
        "red": RED,
        "green": GREEN,
        "yellow": YELLOW,
        "blue": BLUE,
    }
    for color, detection in scene["colors"].items():
        if not detection["found"] or detection["bbox"] is None:
            continue
        x1, y1, x2, y2 = detection["bbox"]
        box = (
            ox + int(x1 * scale_x),
            oy + int(y1 * scale_y),
            ox + int(x2 * scale_x),
            oy + int(y2 * scale_y),
        )
        draw.rectangle(box, outline=colors.get(color, TEXT), width=4)
        draw.text((box[0], max(0, box[1] - 24)), f"{color} {detection['confidence']:.2f}", font=F_SMALL, fill=colors.get(color, TEXT))


def _annotated_image(frame_path: Path, scene: JSON, scored_rows: list[JSON], out_path: Path) -> None:
    frame = Image.open(frame_path).convert("RGB")
    canvas = Image.new("RGB", (1280, 840), BG)
    image_box = (0, 92, 900, 748)
    fitted = _fit_image(frame_path, (image_box[2] - image_box[0], image_box[3] - image_box[1]))
    canvas.paste(fitted, (image_box[0], image_box[1]))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(image_box, outline=LINE, width=2)

    sx = fitted.width / frame.width
    sy = fitted.height / frame.height
    _draw_detection_boxes(draw, scene, sx, sy, image_box[0], image_box[1])

    draw.text((24, 22), "Go2 Cube Micro World Scorer", font=F_TITLE, fill=TEXT)
    draw.text((26, 64), "observation + goal + candidate action -> model score", font=F_SMALL, fill=MUTED)
    panel = (928, 92, 1256, 748)
    draw.rounded_rectangle(panel, radius=22, fill=PANEL, outline=LINE, width=2)
    draw.text((954, 122), "candidate ranking", font=F_H2, fill=TEXT)
    y = 176
    max_score = max(row["score"] for row in scored_rows) or 1.0
    for row in scored_rows:
        color = GREEN if row["selected"] else BLUE
        label = row["candidate_id"].replace("_", " ")
        draw.text((954, y), label, font=F_SMALL, fill=TEXT if row["selected"] else MUTED)
        draw.text((1230, y), f"{row['score']:.3f}", font=F_SMALL, fill=TEXT, anchor="ra")
        draw.rounded_rectangle((954, y + 28, 1230, y + 44), radius=8, fill=(8, 12, 18))
        draw.rounded_rectangle((954, y + 28, 954 + int(276 * row["score"] / max_score), y + 44), radius=8, fill=color)
        yy = y + 52
        for line in wrap(str(row["reason"]), 28)[:2]:
            draw.text((954, yy), line, font=F_SMALL, fill=color if row["selected"] else MUTED)
            yy += 22
        y += 126

    draw.text((24, 782), "Artifacts: score_info.json | candidate_scores.json | selected_action.json | outcome_after_execution.json", font=F_SMALL, fill=MUTED)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)


def _blank() -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)
    for y in range(HEIGHT):
        t = y / HEIGHT
        draw.line((0, y, WIDTH, y), fill=(int(BG[0] + 10 * t), int(BG[1] + 12 * t), int(BG[2] + 16 * t)))
    return img


def _draw_scores(draw: ImageDraw.ImageDraw, rows: list[JSON], x: int, y: int, w: int) -> None:
    max_score = max(row["score"] for row in rows) or 1.0
    for row in rows:
        color = GREEN if row["selected"] else BLUE
        label = row["candidate_id"].replace("_", " ")
        draw.text((x, y), label, font=F_BODY, fill=TEXT if row["selected"] else MUTED)
        draw.text((x + w, y), f"{row['score']:.3f}", font=F_BODY, fill=TEXT, anchor="ra")
        draw.rounded_rectangle((x, y + 34, x + w, y + 54), radius=10, fill=(7, 10, 15))
        draw.rounded_rectangle((x, y + 34, x + int(w * row["score"] / max_score), y + 54), radius=10, fill=color)
        y += 86


def _write_video(frame_path: Path, annotated_path: Path, rows: list[JSON], selected_id: str, out_video: Path) -> None:
    frame_dir = out_video.parent / "video_frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    original = _fit_image(frame_path, (590, 440))
    annotated = _fit_image(annotated_path, (760, 500))
    frame_index = 0

    def save(img: Image.Image, copies: int) -> None:
        nonlocal frame_index
        for _ in range(copies):
            img.save(frame_dir / f"frame_{frame_index:05d}.jpg", quality=91)
            frame_index += 1

    img = _blank()
    draw = ImageDraw.Draw(img)
    draw.text((70, 76), "1. Real robot-view observation", font=F_TITLE, fill=TEXT)
    img.paste(original, (70, 160))
    draw.text((710, 186), "Goal: find the red cube", font=F_H2, fill=GREEN)
    draw.text((710, 238), "Constraint: avoid green/yellow unsafe markers", font=F_BODY, fill=MUTED)
    draw.text((710, 314), "The scorer does not move the robot directly.", font=F_BODY, fill=TEXT)
    draw.text((710, 356), "It ranks candidate actions for the host runtime.", font=F_BODY, fill=TEXT)
    save(img, int(2.5 * FPS))

    img = _blank()
    draw = ImageDraw.Draw(img)
    draw.text((70, 76), "2. Candidate futures are scored", font=F_TITLE, fill=TEXT)
    _draw_scores(draw, rows, 86, 174, 480)
    draw.rounded_rectangle((655, 170, 1190, 500), radius=24, fill=PANEL, outline=LINE, width=2)
    draw.text((688, 206), f"selected: {selected_id}", font=F_H2, fill=GREEN)
    body = "The model is a small scorer head trained from real Go2 frames and label-safe counterfactual edits. It is not a Go2 foundation model."
    yy = 272
    for line in wrap(body, 42):
        draw.text((688, yy), line, font=F_BODY, fill=MUTED)
        yy += 34
    save(img, int(2.5 * FPS))

    img = _blank()
    draw = ImageDraw.Draw(img)
    draw.text((70, 66), "3. Evidence trail is written", font=F_TITLE, fill=TEXT)
    img.paste(annotated, (60, 148))
    artifacts = ["score_info.json", "candidate_scores.json", "selected_action.json", "outcome_after_execution.json"]
    y = 174
    for item in artifacts:
        draw.rounded_rectangle((860, y, 1204, y + 64), radius=18, fill=PANEL_2, outline=LINE, width=1)
        draw.text((884, y + 18), item, font=F_BODY, fill=TEXT)
        y += 84
    save(img, int(3.0 * FPS))

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(FPS),
        "-i",
        str(frame_dir / "frame_%05d.jpg"),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out_video),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run(args: argparse.Namespace) -> int:
    image_path = Path(args.image).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    if not image_path.is_file():
        raise RuntimeError(f"Missing image: {image_path}")
    if not model_path.is_file():
        raise RuntimeError(f"Missing model: {model_path}")

    model = _load_json(model_path)
    unsafe_colors = {
        item.strip().lower()
        for item in args.unsafe_colors.split(",")
        if item.strip().lower() in TARGET_COLORS
    }
    run_id = args.run_id or time.strftime("micro-world-%Y%m%d-%H%M%S")
    out_dir = Path(args.output_dir).expanduser().resolve() / run_id
    if out_dir.exists() and args.clean:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene = _detect_scene(image_path, args.target, unsafe_colors)
    observation = _observation_summary(scene, args.target)
    candidates = _build_candidates(scene)
    transparent_scores = [_score(candidate["features"]) for candidate in candidates]
    model_scores = _predict_scores(model, observation, candidates)
    best_index = max(range(len(model_scores)), key=model_scores.__getitem__)
    selected = candidates[best_index]
    selected_id = str(selected["id"])
    copied_frame = out_dir / "camera_frame.jpg"
    shutil.copy2(image_path, copied_frame)

    scored_rows = []
    for candidate, model_score, transparent_score in zip(candidates, model_scores, transparent_scores):
        scored_rows.append(
            {
                "candidate_id": candidate["id"],
                "score": model_score,
                "transparent_reference_score": transparent_score,
                "features": candidate["features"],
                "selected": candidate["id"] == selected_id,
                "reason": _reason(scene, candidate, selected_id, model_score),
            }
        )
    scored_rows.sort(key=lambda row: float(row["score"]), reverse=True)

    task = {
        "human_goal": f"find the {args.target} block and avoid unsafe colored markers",
        "goal_representation": {
            "type": "host_interpreted_visual_goal",
            "target_label": args.target,
            "unsafe_markers": sorted(unsafe_colors),
        },
    }
    score_info = {
        "schema_version": 1,
        "run_id": run_id,
        "step_index": 1,
        "embodiment": "unitree_go2",
        "host_runtime": "offline-go2-frame-replay",
        "input_source": "real_go2_camera_frame",
        "task": task,
        "observation_summary": observation,
    }
    score_info_artifact = {
        "provider": "go2-cube-micro-world-scorer",
        "capability": "score",
        "model_artifact": str(model_path),
        "claim_boundary": model.get("training_summary", {}).get("claim_boundary"),
        "score_info": score_info,
        "action_candidates": candidates,
    }
    candidate_scores = {
        "schema_version": 1,
        "run_id": run_id,
        "step_index": 1,
        "scores": scored_rows,
        "selected_candidate_id": selected_id,
        "lower_is_better": False,
        "scorer": {
            "model_type": model.get("model_type"),
            "provider": "go2-cube-micro-world-scorer",
            "transparent_reference_weights": SCORE_WEIGHTS,
        },
    }
    selected_action = {
        "schema_version": 1,
        "run_id": run_id,
        "step_index": 1,
        "selected_candidate_id": selected_id,
        "action": selected["action"],
        "params": selected["params"],
        "execute_with": "host_runtime:dimos-mcp",
        "worldforge_executes_robot": False,
        "demo_mode": "offline_replay_no_robot_motion",
    }
    outcome = {
        "schema_version": 1,
        "run_id": run_id,
        "step_index": 1,
        "executed": False,
        "execution_result": "offline demo: selected action not executed",
        "outcome_after_execution": {
            "target_confidence": scene["target"]["confidence"],
            "target_area_ratio": scene["target"]["area_ratio"],
            "unsafe_color_risk": scene["unsafe_risk"],
            "manual_intervention": False,
        },
    }
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "goal": task["human_goal"],
        "artifacts": {
            "camera_frame": "camera_frame.jpg",
            "annotated_image": "annotated_image.jpg",
            "micro_world_trace": "micro_world_trace.mp4",
            "score_info": "score_info.json",
            "candidate_scores": "candidate_scores.json",
            "selected_action": "selected_action.json",
            "outcome_after_execution": "outcome_after_execution.json",
        },
        "claim_boundary": "Offline replay of a small learned scorer head. Host runtime owns robot execution.",
    }

    _write_json(out_dir / "score_info.json", score_info_artifact)
    _write_json(out_dir / "observation_summary.json", observation)
    _write_json(out_dir / "candidate_scores.json", candidate_scores)
    _write_json(out_dir / "selected_action.json", selected_action)
    _write_json(out_dir / "outcome_after_execution.json", outcome)
    _write_json(out_dir / "run_manifest.json", manifest)
    _write_json(
        out_dir / "venue_input.json",
        {
            "schema_version": 1,
            "embodiment": "unitree_go2",
            "task": task,
            "observation_summary": observation,
            "candidates": candidates,
        },
    )

    annotated = out_dir / "annotated_image.jpg"
    _annotated_image(copied_frame, scene, scored_rows, annotated)
    _write_video(copied_frame, annotated, scored_rows, selected_id, out_dir / "micro_world_trace.mp4")
    print(json.dumps({"run_id": run_id, "output_dir": str(out_dir), "selected": selected_id}, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the trained Go2 cube micro world scorer on one real frame.")
    parser.add_argument("--image", default=str(DEFAULT_IMAGE))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--target", choices=TARGET_COLORS, default="red")
    parser.add_argument("--unsafe-colors", default="green,yellow")
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--run-id", default="latest")
    parser.add_argument("--clean", action="store_true", default=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
