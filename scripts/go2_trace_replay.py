from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from go2_find_colored_target import (
    _build_candidates,
    _detect_scene,
    _score,
    _write_json,
    _write_step_artifacts,
)


def _run(command: list[str]) -> None:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(command)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def _annotate_frame(
    *,
    image_path: Path,
    output_path: Path,
    target: str,
    unsafe_colors: set[str],
    scene: dict,
    candidates: list[dict],
    scores: list[float],
    selected: dict,
    frame_index: int,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    colors = {
        "red": (255, 64, 64),
        "green": (0, 255, 96),
        "blue": (64, 140, 255),
        "yellow": (255, 220, 0),
    }

    displayed_colors = {target, *unsafe_colors}
    for color, detection in scene["colors"].items():
        if color not in displayed_colors:
            continue
        if not detection["found"] or not detection["bbox"]:
            continue
        draw.rectangle(detection["bbox"], outline=colors[color], width=4)
        x0, y0, _, _ = detection["bbox"]
        draw.text(
            (x0, max(0, y0 - 22)),
            f"{color} conf={detection['confidence']}",
            fill=colors[color],
        )

    panel_h = 146
    draw.rectangle((0, 0, image.width, panel_h), fill=(0, 0, 0))
    draw.text(
        (14, 12),
        f"WorldForge Trace Replay: target={target} | score candidate futures | keep robot host-owned",
        fill=(255, 255, 255),
    )
    draw.text(
        (14, 42),
        "green/red/blue/yellow detections become observation_summary.visual_summary",
        fill=(230, 230, 230),
    )
    draw.text(
        (14, 68),
        f"frame={frame_index:02d} target_found={scene['target']['found']} "
        f"x={scene['target']['center_x']} conf={scene['target']['confidence']} "
        f"unsafe_risk={scene['unsafe_risk']}",
        fill=(255, 255, 255),
    )
    y = 96
    for candidate, score in zip(candidates, scores):
        chosen = candidate["id"] == selected["id"]
        suffix = " SELECTED" if chosen else ""
        draw.text(
            (14, y),
            f"{candidate['id']}: {score}{suffix}",
            fill=(255, 235, 64) if chosen else (220, 220, 220),
        )
        y += 18

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=92)


def run(args: argparse.Namespace) -> int:
    input_video = Path(args.input_video).expanduser()
    if not input_video.is_file():
        raise RuntimeError(f"Input video not found: {input_video}")

    output_dir = Path(args.output_dir).expanduser()
    frames_dir = output_dir / "frames"
    annotated_dir = output_dir / "annotated_frames"
    trace_dir = output_dir / "trace"
    shutil.rmtree(frames_dir, ignore_errors=True)
    shutil.rmtree(annotated_dir, ignore_errors=True)
    shutil.rmtree(trace_dir, ignore_errors=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)

    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(input_video),
            "-vf",
            f"fps={args.fps},scale={args.width}:-1",
            str(frames_dir / "frame_%04d.jpg"),
        ]
    )

    unsafe_colors = {
        color.strip()
        for color in args.unsafe_colors.split(",")
        if color.strip()
    }
    summary: list[dict] = []
    frames = sorted(frames_dir.glob("frame_*.jpg"))
    if not frames:
        raise RuntimeError("No frames were extracted from the input video.")

    for index, frame in enumerate(frames, 1):
        scene = _detect_scene(frame, args.target, unsafe_colors)
        candidates = _build_candidates(scene)
        scores = [_score(candidate["features"]) for candidate in candidates]
        best_index = max(range(len(scores)), key=scores.__getitem__)
        selected = candidates[best_index]
        step_dir = trace_dir / f"step_{index:02d}"
        _write_step_artifacts(
            step_dir=step_dir,
            run_id=args.run_id,
            step_index=index,
            frame_path=frame,
            target=args.target,
            scene=scene,
            candidates=candidates,
            selected=selected,
            scores=scores,
            executed=False,
            execution_result="replay-only: no robot command executed",
        )
        annotated_path = annotated_dir / frame.name
        _annotate_frame(
            image_path=frame,
            output_path=annotated_path,
            target=args.target,
            unsafe_colors=unsafe_colors,
            scene=scene,
            candidates=candidates,
            scores=scores,
            selected=selected,
            frame_index=index,
        )
        summary.append(
            {
                "frame_index": index,
                "selected_candidate_id": selected["id"],
                "score": scores[best_index],
                "target": scene["target"],
                "unsafe_risk": scene["unsafe_risk"],
                "step_dir": str(step_dir),
                "annotated_frame": str(annotated_path),
            }
        )

    replay_mp4 = output_dir / "worldforge_trace_replay.mp4"
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(args.fps),
            "-i",
            str(annotated_dir / "frame_%04d.jpg"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            "30",
            str(replay_mp4),
        ]
    )
    _write_json(
        output_dir / "summary.json",
        {
            "schema_version": 1,
            "run_id": args.run_id,
            "input_video": str(input_video),
            "target": args.target,
            "unsafe_colors": sorted(unsafe_colors),
            "frame_count": len(summary),
            "replay_video": str(replay_mp4),
            "steps": summary,
        },
    )
    print(f"replay_video={replay_mp4}")
    print(f"summary={output_dir / 'summary.json'}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a WorldForge-style replay from Go2 video.")
    parser.add_argument("--input-video", required=True)
    parser.add_argument("--output-dir", default="artifacts/replay_run")
    parser.add_argument("--run-id", default="go2-camera-replay")
    parser.add_argument("--target", choices=("red", "green", "blue", "yellow"), default="green")
    parser.add_argument("--unsafe-colors", default="red,yellow")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--width", type=int, default=960)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
