from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    from dimos.agents.mcp.mcp_adapter import McpAdapter
except ModuleNotFoundError:  # Offline replay mode can run without DimOS installed.
    McpAdapter = None  # type: ignore[assignment]


JSON = dict[str, Any]
SCORE_WEIGHTS = {
    "goal_alignment": 0.28,
    "information_gain": 0.27,
    "progress": 0.25,
    "clearance": 0.15,
    "not_stuck": 0.05,
    "execution_cost": -0.05,
}
TARGET_COLORS = ("red", "blue", "yellow", "green")
SAFE_CANDIDATE_LIMITS = {
    "forward": 0.14,
    "turn": 0.20,
    "duration": 0.55,
}


@dataclass(frozen=True)
class ColorDetection:
    color: str
    found: bool
    area_ratio: float = 0.0
    center_x: float = 0.0
    center_y: float = 0.0
    bbox: tuple[int, int, int, int] | None = None
    confidence: float = 0.0


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _score(features: JSON) -> float:
    return round(
        0.10
        + SCORE_WEIGHTS["goal_alignment"] * features["goal_alignment"]
        + SCORE_WEIGHTS["information_gain"] * features["information_gain"]
        + SCORE_WEIGHTS["progress"] * features["progress"]
        + SCORE_WEIGHTS["clearance"] * (1.0 - features["obstacle_risk"])
        + SCORE_WEIGHTS["not_stuck"] * (1.0 - features["stuck_risk"])
        + SCORE_WEIGHTS["execution_cost"] * features["execution_cost"],
        4,
    )


def _call_tool_text(adapter: Any, name: str, arguments: JSON | None = None) -> str:
    result = adapter.call_tool(name, arguments or {})
    content = result.get("content", [])
    if content and isinstance(content, list):
        first = content[0]
        if isinstance(first, dict) and first.get("type") == "text":
            return str(first.get("text", ""))
    return json.dumps(result, sort_keys=True)


def _extract_saved_path(response_text: str) -> Path:
    match = re.search(r"camera frame saved:\s+(.+?)\s+\(\d+\s+bytes\)", response_text)
    if match:
        return Path(match.group(1))
    fallback = re.search(r"(/[^()]+\.jpe?g)", response_text)
    if fallback:
        return Path(fallback.group(1).strip())
    raise RuntimeError(f"Could not parse camera path from response: {response_text}")


def _color_mask(rgb: np.ndarray, color: str) -> np.ndarray:
    # The Go2 camera tends to tint the whole frame green. HSV saturation keeps
    # the detector focused on real colored objects rather than the color cast.
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV"))
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    if color == "red":
        return ((h < 15) | (h > 235)) & (s > 70) & (v > 70)
    if color == "blue":
        return (h > 135) & (h < 180) & (s > 55) & (v > 55)
    if color == "yellow":
        return (h > 25) & (h < 50) & (s > 80) & (v > 80)
    if color == "green":
        return (h > 65) & (h < 105) & (s > 80) & (v > 55)
    raise ValueError(f"Unsupported color: {color}")


def _best_component(mask: np.ndarray, *, prefer_floor: bool = True) -> tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    best_y: list[int] = []
    best_x: list[int] = []
    best_score = 0.0

    for start_y, start_x in np.argwhere(mask):
        y0 = int(start_y)
        x0 = int(start_x)
        if seen[y0, x0]:
            continue

        stack = [(y0, x0)]
        seen[y0, x0] = True
        ys: list[int] = []
        xs: list[int] = []
        while stack:
            y, x = stack.pop()
            ys.append(y)
            xs.append(x)
            for ny in (y - 1, y, y + 1):
                if ny < 0 or ny >= height:
                    continue
                for nx in (x - 1, x, x + 1):
                    if nx < 0 or nx >= width or seen[ny, nx] or not mask[ny, nx]:
                        continue
                    seen[ny, nx] = True
                    stack.append((ny, nx))

        score = float(len(xs))
        if prefer_floor and xs:
            center_y = (float(np.mean(ys)) - height / 2.0) / (height / 2.0)
            # The hackathon target cubes sit on the floor. This discounts large
            # red clothing/skin/background blobs in the upper half of the image
            # without hard-cropping the frame.
            floor_weight = _clamp((center_y + 0.20) / 1.20, 0.05, 1.0)
            score *= 0.15 + floor_weight

        if score > best_score:
            best_score = score
            best_y = ys
            best_x = xs

    return np.asarray(best_y), np.asarray(best_x)


def _detect_color(image_path: Path, color: str) -> ColorDetection:
    image = Image.open(image_path).convert("RGB")
    rgb = np.asarray(image)
    height, width = rgb.shape[:2]
    mask = _color_mask(rgb, color)

    # Ignore tiny sensor/noise speckles by requiring a small but real colored blob.
    ys, xs = _best_component(mask)
    area_ratio = float(len(xs)) / float(width * height)
    if len(xs) < 80 or area_ratio < 0.00025:
        return ColorDetection(color=color, found=False)

    min_x, max_x = int(xs.min()), int(xs.max())
    min_y, max_y = int(ys.min()), int(ys.max())
    center_x_px = float(xs.mean())
    center_y_px = float(ys.mean())
    center_x = (center_x_px - width / 2.0) / (width / 2.0)
    center_y = (center_y_px - height / 2.0) / (height / 2.0)
    confidence = _clamp(area_ratio / 0.018)
    return ColorDetection(
        color=color,
        found=True,
        area_ratio=round(area_ratio, 5),
        center_x=round(center_x, 4),
        center_y=round(center_y, 4),
        bbox=(min_x, min_y, max_x, max_y),
        confidence=round(confidence, 4),
    )


def _detect_scene(image_path: Path, target: str, unsafe_colors: set[str]) -> JSON:
    detections = {color: _detect_color(image_path, color) for color in TARGET_COLORS}
    target_detection = detections[target]
    unsafe_hits = [detections[color] for color in unsafe_colors if detections[color].found]
    unsafe_risk = 0.0
    if unsafe_hits:
        for hit in unsafe_hits:
            centered = 1.0 - min(1.0, abs(hit.center_x))
            close = _clamp(hit.area_ratio / 0.018)
            unsafe_risk = max(unsafe_risk, 0.20 + 0.55 * centered + 0.25 * close)
    distractors = [
        detection
        for color, detection in detections.items()
        if color != target and detection.found and color not in unsafe_colors
    ]
    strongest_distractor = max(distractors, key=lambda d: d.area_ratio, default=None)
    return {
        "target": _detection_json(target_detection),
        "colors": {color: _detection_json(detection) for color, detection in detections.items()},
        "unsafe_risk": round(_clamp(unsafe_risk), 4),
        "strongest_distractor": (
            _detection_json(strongest_distractor) if strongest_distractor is not None else None
        ),
    }


def _detection_json(detection: ColorDetection | None) -> JSON | None:
    if detection is None:
        return None
    return {
        "color": detection.color,
        "found": detection.found,
        "area_ratio": detection.area_ratio,
        "center_x": detection.center_x,
        "center_y": detection.center_y,
        "bbox": detection.bbox,
        "confidence": detection.confidence,
    }


def _candidate_features(scene: JSON, candidate_id: str) -> JSON:
    target = scene["target"]
    target_found = bool(target["found"])
    x = float(target["center_x"]) if target_found else 0.0
    area = float(target["area_ratio"]) if target_found else 0.0
    centered = _clamp(1.0 - abs(x) * 1.8) if target_found else 0.0
    unsafe_risk = float(scene["unsafe_risk"])
    close_enough = target_found and abs(x) <= 0.18 and area >= 0.020

    if candidate_id == "turn_left":
        alignment = _clamp(0.20 + max(0.0, -x) * 1.20) if target_found else 0.62
        return {
            "goal_alignment": alignment,
            "information_gain": 0.72 if not target_found else _clamp(0.35 + abs(x)),
            "progress": 0.30 if target_found else 0.22,
            "obstacle_risk": 0.04,
            "stuck_risk": 0.03,
            "execution_cost": 0.12,
        }
    if candidate_id == "turn_right":
        alignment = _clamp(0.20 + max(0.0, x) * 1.20) if target_found else 0.62
        return {
            "goal_alignment": alignment,
            "information_gain": 0.72 if not target_found else _clamp(0.35 + abs(x)),
            "progress": 0.30 if target_found else 0.22,
            "obstacle_risk": 0.04,
            "stuck_risk": 0.03,
            "execution_cost": 0.12,
        }
    if candidate_id == "forward_small":
        return {
            "goal_alignment": centered if target_found else 0.08,
            "information_gain": 0.34 if target_found else 0.12,
            "progress": 0.82 if target_found and centered > 0.60 else 0.18,
            "obstacle_risk": _clamp(unsafe_risk + (0.28 if not target_found else 0.0)),
            "stuck_risk": 0.08 if unsafe_risk < 0.35 else 0.18,
            "execution_cost": 0.08,
        }
    if candidate_id == "stop_capture":
        return {
            "goal_alignment": 0.96 if close_enough else (0.35 if target_found else 0.10),
            "information_gain": 0.18,
            "progress": 0.92 if close_enough else 0.0,
            "obstacle_risk": 0.01,
            "stuck_risk": 0.01,
            "execution_cost": 0.20,
        }
    raise ValueError(f"Unknown candidate: {candidate_id}")


def _reason(scene: JSON, candidate_id: str, selected: bool) -> str:
    target = scene["target"]
    if not target["found"]:
        return "scan candidate while target is not visible"
    x = float(target["center_x"])
    if candidate_id == "turn_left":
        return "turns target toward image center" if x < -0.10 else "target is not left enough"
    if candidate_id == "turn_right":
        return "turns target toward image center" if x > 0.10 else "target is not right enough"
    if candidate_id == "forward_small":
        if scene["unsafe_risk"] > 0.35:
            return "forward progress penalized by unsafe color near path"
        return "safe forward progress when target is centered"
    if candidate_id == "stop_capture":
        return "success condition reached" if selected else "only useful once target is centered and close"
    return ""


def _build_candidates(scene: JSON) -> list[JSON]:
    definitions = [
        ("turn_left", {"turn": -SAFE_CANDIDATE_LIMITS["turn"], "duration": 0.45}),
        ("turn_right", {"turn": SAFE_CANDIDATE_LIMITS["turn"], "duration": 0.45}),
        (
            "forward_small",
            {"forward": SAFE_CANDIDATE_LIMITS["forward"], "duration": SAFE_CANDIDATE_LIMITS["duration"]},
        ),
        ("stop_capture", {"seconds": 0.5}),
    ]
    scored: list[tuple[JSON, float]] = []
    for candidate_id, params in definitions:
        features = _candidate_features(scene, candidate_id)
        scored.append(
            (
                {
                    "id": candidate_id,
                    "action": "wait" if candidate_id == "stop_capture" else "move_joystick",
                    "params": params,
                    "features": features,
                    "reason_hint": "",
                },
                _score(features),
            )
        )
    best_id = max(scored, key=lambda pair: pair[1])[0]["id"]
    candidates = []
    for candidate, _ in scored:
        candidate["reason_hint"] = _reason(scene, str(candidate["id"]), candidate["id"] == best_id)
        candidates.append(candidate)
    return candidates


def _observation_summary(scene: JSON, target: str) -> JSON:
    target_detection = scene["target"]
    return {
        "pose": {"x": 0.0, "y": 0.0, "yaw_degrees": 0.0},
        "costmap_summary": {
            "min_clearance_m": None,
            "unknown_area_ratio": None,
            "frontier_count": None,
            "blocked_ahead": scene["unsafe_risk"] > 0.45,
        },
        "visual_summary": {
            "target_label": target,
            "target_confidence": target_detection["confidence"],
            "target_bearing_degrees": round(float(target_detection["center_x"]) * 45.0, 2)
            if target_detection["found"]
            else None,
            "target_area_ratio": target_detection["area_ratio"],
            "unsafe_color_risk": scene["unsafe_risk"],
            "strongest_distractor": scene["strongest_distractor"],
        },
        "navigation_state": {
            "localized": True,
            "stuck_probability": 0.05,
            "active_goal": f"find_{target}_block",
        },
    }


def _write_json(path: Path, data: JSON) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _write_step_artifacts(
    *,
    step_dir: Path,
    run_id: str,
    step_index: int,
    frame_path: Path,
    target: str,
    scene: JSON,
    candidates: list[JSON],
    selected: JSON,
    scores: list[float],
    executed: bool,
    execution_result: str | None,
) -> None:
    step_dir.mkdir(parents=True, exist_ok=True)
    copied_frame = step_dir / "camera_frame.jpg"
    try:
        shutil.copy2(frame_path, copied_frame)
    except OSError:
        copied_frame = frame_path

    task = {
        "human_goal": f"find the {target} block and avoid unsafe colored markers",
        "goal_representation": {
            "type": "host_interpreted_visual_goal",
            "target_label": target,
            "unsafe_markers": "configured_by_host",
        },
    }
    observation = _observation_summary(scene, target)
    score_info = {
        "schema_version": 1,
        "run_id": run_id,
        "step_index": step_index,
        "embodiment": "unitree_go2",
        "host_runtime": "dimos-mcp-local-router",
        "input_source": "live_camera_color_detector",
        "task": task,
        "observation_summary": observation,
    }
    score_info_artifact = {
        "provider": "transparent-go2-color-target-score",
        "capability": "score",
        "score_info": score_info,
        "action_candidates": candidates,
    }
    candidate_scores = {
        "schema_version": 1,
        "run_id": run_id,
        "step_index": step_index,
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
        "weights": SCORE_WEIGHTS,
    }
    selected_action = {
        "schema_version": 1,
        "run_id": run_id,
        "step_index": step_index,
        "selected_candidate_id": selected["id"],
        "action": selected["action"],
        "params": selected["params"],
        "execute_with": "host_runtime:dimos-mcp",
        "worldforge_executes_robot": False,
    }
    outcome = {
        "schema_version": 1,
        "run_id": run_id,
        "step_index": step_index,
        "executed": executed,
        "execution_result": execution_result,
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
        "step_index": step_index,
        "goal": task["human_goal"],
        "artifacts": {
            "camera_frame": str(copied_frame),
            "score_info": "score_info.json",
            "observation_summary": "observation_summary.json",
            "candidate_scores": "candidate_scores.json",
            "selected_action": "selected_action.json",
            "outcome_after_execution": "outcome_after_execution.json",
        },
        "safety_boundary": {
            "worldforge_executes_robot": False,
            "host_executes_selected_action": executed,
            "motion_limits": SAFE_CANDIDATE_LIMITS,
        },
    }
    report = "\n".join(
        [
            f"# Step {step_index}: {task['human_goal']}",
            "",
            f"Frame: `{copied_frame}`",
            f"Target found: `{scene['target']['found']}` confidence `{scene['target']['confidence']}`",
            f"Unsafe color risk: `{scene['unsafe_risk']}`",
            "",
            "## Candidates",
            *[
                f"- `{item['candidate_id']}` score `{item['score']}`: {item['reason']}"
                for item in candidate_scores["scores"]
            ],
            "",
            f"Selected: `{selected['id']}`",
            f"Executed: `{executed}`",
            f"Execution result: `{execution_result}`",
            "",
        ]
    )

    _write_json(step_dir / "score_info.json", score_info_artifact)
    _write_json(step_dir / "observation_summary.json", observation)
    _write_json(step_dir / "venue_input.json", {
        "schema_version": 1,
        "embodiment": "unitree_go2",
        "host_runtime": "dimos-mcp-local-router",
        "task": task,
        "observation_summary": observation,
        "candidates": candidates,
    })
    _write_json(step_dir / "candidate_scores.json", candidate_scores)
    _write_json(step_dir / "selected_action.json", selected_action)
    _write_json(step_dir / "outcome_after_execution.json", outcome)
    _write_json(step_dir / "run_manifest.json", manifest)
    (step_dir / "report.md").write_text(report)


def run_demo(args: argparse.Namespace) -> int:
    adapter: Any | None = None
    dry_run_frame = Path(args.dry_run_frame).expanduser() if args.dry_run_frame else None
    if dry_run_frame is not None and not dry_run_frame.is_file():
        raise RuntimeError(f"Dry-run frame does not exist: {dry_run_frame}")
    if dry_run_frame is None or args.execute:
        if McpAdapter is None:
            raise RuntimeError("DimOS MCP adapter is required for live robot mode.")
        adapter = McpAdapter(args.mcp_url)
        if not adapter.wait_for_ready(timeout=8):
            raise RuntimeError(f"MCP server is not ready at {args.mcp_url}")

    run_id = args.run_id or time.strftime("color-target-%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir).expanduser() / run_id
    unsafe_colors = {
        color.strip().lower()
        for color in args.unsafe_colors.split(",")
        if color.strip() and color.strip().lower() in TARGET_COLORS
    }

    print(f"run_id={run_id}")
    print(f"output_dir={output_dir}")
    print(f"target={args.target} unsafe_colors={sorted(unsafe_colors)} execute={args.execute}")

    if args.balance_first and args.execute and adapter is not None:
        print(_call_tool_text(adapter, "execute_sport_command", {"command_name": "BalanceStand"}))

    summary_steps: list[JSON] = []
    success = False
    for step_index in range(1, args.max_steps + 1):
        if dry_run_frame is not None:
            frame_path = dry_run_frame
        else:
            assert adapter is not None
            filename = f"{run_id}_step_{step_index:02d}.jpg"
            capture_text = _call_tool_text(adapter, "capture_camera_frame", {"filename": filename})
            frame_path = _extract_saved_path(capture_text)
        scene = _detect_scene(frame_path, args.target, unsafe_colors)
        candidates = _build_candidates(scene)
        scores = [_score(candidate["features"]) for candidate in candidates]
        best_index = max(range(len(scores)), key=scores.__getitem__)
        selected = candidates[best_index]

        target = scene["target"]
        print(
            f"step={step_index} target_found={target['found']} "
            f"x={target['center_x']} area={target['area_ratio']} "
            f"unsafe={scene['unsafe_risk']} selected={selected['id']} score={scores[best_index]}"
        )

        execution_result = None
        if selected["id"] == "stop_capture":
            success = bool(target["found"] and abs(float(target["center_x"])) <= 0.18 and float(target["area_ratio"]) >= args.success_area)
            execution_result = "success: target centered and close" if success else "stop/capture selected"
        elif args.execute:
            assert adapter is not None
            execution_result = _call_tool_text(adapter, "move_joystick", selected["params"])
        else:
            execution_result = "dry-run: selected action not executed"

        step_dir = output_dir / f"step_{step_index:02d}"
        _write_step_artifacts(
            step_dir=step_dir,
            run_id=run_id,
            step_index=step_index,
            frame_path=frame_path,
            target=args.target,
            scene=scene,
            candidates=candidates,
            selected=selected,
            scores=scores,
            executed=bool(args.execute and selected["id"] != "stop_capture"),
            execution_result=execution_result,
        )
        summary_steps.append(
            {
                "step_index": step_index,
                "selected_candidate_id": selected["id"],
                "score": scores[best_index],
                "target": target,
                "unsafe_color_risk": scene["unsafe_risk"],
                "step_dir": str(step_dir),
                "execution_result": execution_result,
            }
        )
        if success:
            print("SUCCESS target centered and close")
            if args.celebrate and args.execute and adapter is not None:
                print(_call_tool_text(adapter, "execute_sport_command", {"command_name": "Hello"}))
            break

    _write_json(
        output_dir / "summary.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "target": args.target,
            "unsafe_colors": sorted(unsafe_colors),
            "execute": args.execute,
            "success": success,
            "steps": summary_steps,
        },
    )
    print(f"summary={output_dir / 'summary.json'}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous colored-target Go2 demo.")
    parser.add_argument("--target", choices=TARGET_COLORS, default="red")
    parser.add_argument("--unsafe-colors", default="green")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--success-area", type=float, default=0.020)
    parser.add_argument("--output-dir", default=".worldforge/color-target-demo")
    parser.add_argument("--run-id")
    parser.add_argument("--mcp-url", default="http://localhost:9990/mcp")
    parser.add_argument(
        "--dry-run-frame",
        help="Use one local image instead of the live robot camera. This never moves the robot unless --execute is also passed.",
    )
    parser.add_argument("--execute", action="store_true", help="Actually move the robot.")
    parser.add_argument("--balance-first", action="store_true")
    parser.add_argument("--celebrate", action="store_true")
    args = parser.parse_args()
    args.max_steps = max(1, min(12, args.max_steps))
    return args


if __name__ == "__main__":
    raise SystemExit(run_demo(parse_args()))
