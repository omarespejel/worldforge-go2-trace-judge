from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from textwrap import wrap
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from go2_find_colored_target import (
    TARGET_COLORS,
    _best_component,
    _build_candidates,
    _color_mask,
    _detect_color,
    _detect_scene,
    _score,
    _write_step_artifacts,
)


ROOT = Path(__file__).resolve().parents[1]
JSON = dict[str, Any]
COLORS = {
    "red": (226, 48, 58),
    "green": (26, 158, 80),
    "yellow": (230, 190, 52),
    "blue": (64, 125, 220),
}


@dataclass(frozen=True)
class CubeAsset:
    color: str
    source: str
    image: Image.Image


def _write_json(path: Path, data: JSON) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _floor_plate_paths() -> list[Path]:
    return [
        ROOT / "artifacts/live_ciro/direct_camera_no_red.jpg",
        ROOT / "artifacts/live_ciro/direct_camera_final_preflight.jpg",
        ROOT / "artifacts/live_ciro/direct_camera_red_block_front.jpg",
        ROOT / "artifacts/live_ciro/direct_camera_red_block_left.jpg",
        ROOT / "artifacts/live_ciro/direct_camera_red_block_right.jpg",
    ]


def _asset_sources() -> dict[str, list[Path]]:
    return {
        "red": [
            ROOT / "artifacts/live_ciro/direct_camera_red_block_front.jpg",
            ROOT / "artifacts/live_ciro/direct_camera_red_block_left.jpg",
            ROOT / "artifacts/live_ciro/direct_camera_red_block_right.jpg",
            ROOT / "artifacts/live_ciro/direct_camera_unsafe_path.jpg",
            ROOT / "artifacts/live_ciro/direct_camera_final_preflight.jpg",
        ],
        "green": [
            ROOT / "artifacts/live_ciro/direct_camera_unsafe_path.jpg",
            ROOT / "artifacts/live_ciro/direct_camera_final_preflight.jpg",
        ],
        "yellow": [
            ROOT / "artifacts/live_ciro/direct_camera_final_preflight.jpg",
            ROOT / "artifacts/live_ciro/direct_camera_unsafe_path.jpg",
        ],
    }


def _clean_plate(path: Path, *, target_color: str = "red") -> Image.Image:
    image = Image.open(path).convert("RGB")
    detection = _detect_color(path, target_color)
    if not detection.found or detection.bbox is None:
        return image

    # Only cover obvious floor-level target cubes. Upper-frame red/yellow venue
    # artifacts are left untouched because replacing them often looks worse.
    x1, y1, x2, y2 = detection.bbox
    if (y1 + y2) / 2 < image.height * 0.55:
        return image

    arr = np.asarray(image).copy()
    pad = 18
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(image.width - 1, x2 + pad)
    y2 = min(image.height - 1, y2 + pad)
    source_y1 = max(0, y1 - (y2 - y1 + 1) - 12)
    source_y2 = source_y1 + (y2 - y1 + 1)
    if source_y2 > arr.shape[0]:
        source_y2 = arr.shape[0]
        source_y1 = max(0, source_y2 - (y2 - y1 + 1))
    patch = arr[source_y1:source_y2, x1 : x2 + 1]
    if patch.shape[:2] == arr[y1 : y2 + 1, x1 : x2 + 1].shape[:2]:
        arr[y1 : y2 + 1, x1 : x2 + 1] = patch
    return Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=0.15))


def _rgba_asset(path: Path, color: str) -> CubeAsset | None:
    detection = _detect_color(path, color)
    if not detection.found or detection.bbox is None:
        return None
    image = Image.open(path).convert("RGB")
    rgb = np.asarray(image)
    raw_mask = _color_mask(rgb, color)
    ys, xs = _best_component(raw_mask)
    if len(xs) < 60:
        return None

    x1, y1, x2, y2 = detection.bbox
    pad = 12
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(image.width - 1, x2 + pad)
    y2 = min(image.height - 1, y2 + pad)

    patch = image.crop((x1, y1, x2 + 1, y2 + 1)).convert("RGBA")
    mask = Image.fromarray((raw_mask.astype("uint8") * 255)).crop((x1, y1, x2 + 1, y2 + 1))
    alpha = mask.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(radius=1.2))
    patch.putalpha(alpha)
    if patch.width < 10 or patch.height < 10:
        return None
    return CubeAsset(color=color, source=str(path.relative_to(ROOT)), image=patch)


def _fallback_recolor_asset(base: CubeAsset, color: str) -> CubeAsset:
    target = np.asarray(COLORS[color], dtype=np.float32)
    patch = base.image.convert("RGBA")
    arr = np.asarray(patch).copy()
    alpha = arr[:, :, 3]
    gray = arr[:, :, :3].mean(axis=2, keepdims=True)
    shaded = np.clip(target.reshape(1, 1, 3) * (0.65 + gray / 510.0), 0, 255).astype(np.uint8)
    arr[:, :, :3] = np.where(alpha[:, :, None] > 0, shaded, arr[:, :, :3])
    return CubeAsset(color=color, source=f"recolored:{base.source}", image=Image.fromarray(arr, "RGBA"))


def _load_assets() -> dict[str, list[CubeAsset]]:
    assets: dict[str, list[CubeAsset]] = {color: [] for color in TARGET_COLORS}
    for color, sources in _asset_sources().items():
        for source in sources:
            if not source.exists():
                continue
            asset = _rgba_asset(source, color)
            if asset is not None:
                assets[color].append(asset)

    # We only have a few real cubes. If yellow is too weak, tint a real cube but
    # keep the geometry and alpha from a true Go2-camera cube crop.
    if not assets["yellow"] and assets["red"]:
        assets["yellow"].append(_fallback_recolor_asset(assets["red"][0], "yellow"))
    if not assets["green"] and assets["red"]:
        assets["green"].append(_fallback_recolor_asset(assets["red"][0], "green"))
    return assets


def _place_from_norm(width: int, height: int, x_norm: float, y_norm: float) -> tuple[int, int, int]:
    cx = int(width * (0.5 + 0.46 * x_norm))
    cy = int(height * y_norm)
    scale_t = max(0.0, min(1.0, (y_norm - 0.54) / 0.36))
    size = int(15 + 68 * scale_t)
    return cx, cy, max(12, size)


def _shadow(size: tuple[int, int], strength: int) -> Image.Image:
    width, height = size
    shadow = Image.new("RGBA", (width + 24, max(10, height // 3)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(shadow)
    draw.ellipse((4, 2, width + 20, max(8, height // 3)), fill=(0, 0, 0, strength))
    return shadow.filter(ImageFilter.GaussianBlur(radius=5))


def _paste_asset(
    base: Image.Image,
    asset: CubeAsset,
    *,
    x_norm: float,
    y_norm: float,
    rng: random.Random,
    mask_canvas: Image.Image,
) -> JSON:
    width, height = base.size
    cx, cy, size = _place_from_norm(width, height, x_norm, y_norm)
    aspect = asset.image.width / max(1, asset.image.height)
    jitter = rng.uniform(0.92, 1.08)
    new_h = max(10, int(size * jitter))
    new_w = max(10, int(new_h * aspect))
    patch = asset.image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    patch = patch.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.0, 0.35)))

    x = max(2, min(width - new_w - 2, cx - new_w // 2))
    y = max(int(height * 0.42), min(height - new_h - 2, cy - new_h // 2))
    shadow = _shadow((new_w, new_h), strength=int(38 + 22 * (y / height)))
    base.alpha_composite(shadow, (max(0, x - 10), min(height - shadow.height - 1, y + new_h - shadow.height // 3)))

    # Weak reflection on polished floor, clipped and blurred so it reads as venue
    # reflection instead of a second cube.
    reflection_alpha = patch.getchannel("A").transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    reflection = patch.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    reflection_alpha = reflection_alpha.point(lambda value: int(value * 0.18))
    reflection.putalpha(reflection_alpha.filter(ImageFilter.GaussianBlur(radius=2.0)))
    base.alpha_composite(reflection, (x, min(height - new_h - 1, y + new_h - 2)))
    base.alpha_composite(patch, (x, y))

    mask_patch = patch.getchannel("A").point(lambda value: 255 if value > 24 else 0)
    mask_canvas.paste(mask_patch, (x, y), mask_patch)
    return {
        "color": asset.color,
        "source": asset.source,
        "x_norm": round(float(x_norm), 4),
        "y_norm": round(float(y_norm), 4),
        "bbox": [x, y, x + new_w, y + new_h],
        "area_ratio": round((new_w * new_h) / float(width * height), 6),
    }


def _config(index: int, rng: random.Random) -> JSON:
    modes = [
        ("left", -0.72, 0.78),
        ("center", 0.0, 0.80),
        ("right", 0.72, 0.78),
        ("far_left", -0.55, 0.63),
        ("far_center", 0.05, 0.61),
        ("far_right", 0.55, 0.63),
    ]
    mode, base_x, base_y = modes[index % len(modes)]
    target_x = max(-0.92, min(0.92, base_x + rng.uniform(-0.10, 0.10)))
    target_y = max(0.57, min(0.90, base_y + rng.uniform(-0.035, 0.035)))
    unsafe_blocks: list[JSON] = []
    if index % 4 in {1, 2}:
        unsafe_blocks.append(
            {
                "color": "green" if index % 2 else "yellow",
                "x_norm": rng.uniform(-0.22, 0.22),
                "y_norm": rng.uniform(0.78, 0.91),
                "role": "path_blocker",
            }
        )
    if index % 9 == 0:
        unsafe_blocks.append(
            {
                "color": "green",
                "x_norm": rng.choice([-0.75, 0.75]) + rng.uniform(-0.05, 0.05),
                "y_norm": rng.uniform(0.64, 0.78),
                "role": "side_marker",
            }
        )
    no_target = index % 17 == 16
    return {
        "index": index,
        "mode": "no_target" if no_target else mode,
        "target_color": "red",
        "target": None if no_target else {"color": "red", "x_norm": target_x, "y_norm": target_y},
        "unsafe_blocks": unsafe_blocks,
    }


def _annotate(image: Image.Image, metadata: JSON, scores: JSON) -> Image.Image:
    out = image.convert("RGB")
    draw = ImageDraw.Draw(out)
    f_small = _font(18)
    f_bold = _font(22, bold=True)
    selected = scores["selected_candidate_id"]
    draw.rounded_rectangle((18, 18, 430, 148), radius=10, fill=(0, 0, 0), outline=(45, 55, 70), width=2)
    draw.text((34, 32), f"selected={selected}", fill=(88, 236, 154), font=f_bold)
    y = 66
    for row in sorted(scores["scores"], key=lambda item: item["score"], reverse=True):
        draw.text((34, y), f"{row['candidate_id']:<14} {row['score']:.3f}", fill=(235, 240, 247), font=f_small)
        y += 22
    for obj in metadata["objects"]:
        x1, y1, x2, y2 = obj["bbox"]
        color = {"red": (255, 60, 70), "green": (55, 220, 110), "yellow": (255, 220, 55)}.get(obj["color"], (80, 150, 255))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
        draw.text((x1, max(0, y1 - 24)), obj["color"], fill=color, font=f_small)
    return out


def _contact_sheet(image_paths: list[Path], output: Path, *, cols: int = 4) -> None:
    if not image_paths:
        return
    cell_w, cell_h = 360, 260
    rows = math.ceil(len(image_paths) / cols)
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (244, 246, 248))
    draw = ImageDraw.Draw(sheet)
    font = _font(12)
    for idx, path in enumerate(image_paths):
        x = (idx % cols) * cell_w
        y = (idx // cols) * cell_h
        draw.rectangle((x, y, x + cell_w - 1, y + cell_h - 1), outline=(210, 215, 220), fill=(255, 255, 255))
        image = Image.open(path).convert("RGB")
        image.thumbnail((cell_w - 16, cell_h - 54))
        sheet.paste(image, (x + (cell_w - image.width) // 2, y + 8))
        label = str(path.relative_to(ROOT))
        draw.text((x + 10, y + cell_h - 40), "\n".join(wrap(label, 46)[:2]), fill=(0, 0, 0), font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=90)


def build(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and args.clean:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    traces_dir = output_dir / "traces"
    annotated_dir = output_dir / "annotated"

    rng = random.Random(args.seed)
    assets = _load_assets()
    plates = [path for path in _floor_plate_paths() if path.exists()]
    if not plates:
        raise RuntimeError("No approved floor plates found.")
    if not assets["red"]:
        raise RuntimeError("No red cube asset found.")

    rows: list[JSON] = []
    attempts = 0
    max_attempts = args.count * 8
    while len(rows) < args.count and attempts < max_attempts:
        attempts += 1
        index = len(rows)
        cfg = _config(index, rng)
        plate_path = rng.choice(plates)
        base = _clean_plate(plate_path).convert("RGBA")
        mask_canvas = Image.new("L", base.size, 0)
        objects: list[JSON] = []

        if cfg["target"] is not None:
            target_asset = rng.choice(assets["red"])
            objects.append(
                _paste_asset(
                    base,
                    target_asset,
                    x_norm=float(cfg["target"]["x_norm"]),
                    y_norm=float(cfg["target"]["y_norm"]),
                    rng=rng,
                    mask_canvas=mask_canvas,
                )
            )
        for unsafe in cfg["unsafe_blocks"]:
            color = str(unsafe["color"])
            if not assets.get(color):
                continue
            objects.append(
                _paste_asset(
                    base,
                    rng.choice(assets[color]),
                    x_norm=float(unsafe["x_norm"]),
                    y_norm=float(unsafe["y_norm"]),
                    rng=rng,
                    mask_canvas=mask_canvas,
                )
            )

        image = base.convert("RGB")
        image_path = images_dir / f"real_photo_edit_{index:04d}.jpg"
        mask_path = masks_dir / f"real_photo_edit_{index:04d}_mask.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(image_path, quality=92)
        mask_canvas.save(mask_path)

        unsafe_set = {str(block["color"]) for block in cfg["unsafe_blocks"]}
        scene = _detect_scene(image_path, "red", unsafe_set)
        if cfg["target"] is not None and not scene["target"]["found"]:
            image_path.unlink(missing_ok=True)
            mask_path.unlink(missing_ok=True)
            continue
        if cfg["target"] is not None:
            detected_x = float(scene["target"]["center_x"])
            expected_x = float(cfg["target"]["x_norm"])
            if abs(detected_x - expected_x) > 0.35:
                image_path.unlink(missing_ok=True)
                mask_path.unlink(missing_ok=True)
                continue

        candidates = _build_candidates(scene)
        scores = [_score(candidate["features"]) for candidate in candidates]
        best_index = max(range(len(scores)), key=scores.__getitem__)
        selected = candidates[best_index]
        run_id = f"real-photo-edit-{index:04d}"
        step_dir = traces_dir / run_id
        _write_step_artifacts(
            step_dir=step_dir,
            run_id=run_id,
            step_index=index,
            frame_path=image_path,
            target="red",
            scene=scene,
            candidates=candidates,
            selected=selected,
            scores=scores,
            executed=False,
            execution_result="offline real-photo-edit counterfactual",
        )

        metadata = {
            "schema_version": 1,
            "row_id": run_id,
            "source_domain": "real_photo_edit",
            "base_plate": str(plate_path.relative_to(ROOT)),
            "image": str(image_path.relative_to(ROOT)),
            "mask": str(mask_path.relative_to(ROOT)),
            "trace_dir": str(step_dir.relative_to(ROOT)),
            "objects": objects,
            "augmentation_config": cfg,
            "detected_scene": scene,
            "selected_candidate_id": selected["id"],
            "split": "test" if index % 10 == 0 else ("validation" if index % 10 in {1, 2} else "train"),
            "quality": {
                "auto_review": "pass",
                "label_source": "transparent_scorer_from_detected_scene",
                "claim_boundary": "label-preserving counterfactual image, not measured execution outcome",
            },
        }
        _write_json(step_dir / "augmentation_metadata.json", metadata)
        scores_payload = json.loads((step_dir / "candidate_scores.json").read_text())
        annotated = _annotate(image, metadata, scores_payload)
        annotated_path = annotated_dir / f"real_photo_edit_{index:04d}_annotated.jpg"
        annotated_path.parent.mkdir(parents=True, exist_ok=True)
        annotated.save(annotated_path, quality=90)
        metadata["annotated_image"] = str(annotated_path.relative_to(ROOT))
        _write_json(step_dir / "augmentation_metadata.json", metadata)
        rows.append(metadata)

    if len(rows) < args.count:
        raise RuntimeError(f"Generated only {len(rows)} rows after {attempts} attempts; target was {args.count}.")

    _write_json(
        output_dir / "dataset_manifest.json",
        {
            "schema_version": 1,
            "count": len(rows),
            "seed": args.seed,
            "attempts": attempts,
            "source_domains": ["real_photo_edit"],
            "asset_counts": {color: len(items) for color, items in assets.items()},
            "splits": {split: sum(1 for row in rows if row["split"] == split) for split in ("train", "validation", "test")},
            "rows": rows,
        },
    )
    with (output_dir / "metadata.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    sample_paths = [Path(row["annotated_image"]) for row in rows[: min(args.contact_count, len(rows))]]
    _contact_sheet([ROOT / path for path in sample_paths], output_dir / "contact_sheet.jpg")
    print(json.dumps({"count": len(rows), "attempts": attempts, "output_dir": str(output_dir)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build label-safe real-photo-edit Go2 cube counterfactuals.")
    parser.add_argument("--output-dir", default="artifacts/real_photo_edit_dataset")
    parser.add_argument("--count", type=int, default=480)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--contact-count", type=int, default=48)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
