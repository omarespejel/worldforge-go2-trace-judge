from __future__ import annotations

import argparse
import bisect
import hashlib
import io
import json
import math
import shutil
import sqlite3
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
LFS_ENDPOINT = "https://github.com/dimensionalOS/dimos.git/info/lfs/objects/batch"
SOURCE_REPO = "https://github.com/dimensionalOS/dimos"

DIMOS_LFS_OBJECTS: dict[str, dict[str, Any]] = {
    "go2_short": {
        "path": "data/.lfs/go2_short.db.tar.gz",
        "oid": "8a19846a0adf5755815fd039492c0255e0bc282e9df75a06648d7585cae8d2d2",
        "size": 83_971_952,
        "db_name": "go2_short.db",
        "description": "Small public DimOS Unitree Go2 replay with color_image, odom, and lidar streams.",
    },
    "go2_china_office": {
        "path": "data/.lfs/go2_china_office.db.tar.gz",
        "oid": "834539871fd325b15f3079a3490b278c54e78d0d40bfa1342dbdc983f6a3ee02",
        "size": 136_080_653,
        "db_name": "go2_china_office.db",
        "description": "Public DimOS Unitree Go2 office replay with robot-view image, odom, and lidar streams.",
    },
    "markers_go2": {
        "path": "data/.lfs/markers_go2.db.tar.gz",
        "oid": "5a43529f8dbc2aedcccca6ae89747235826123c2bc066e0dc8b87c2042219dae",
        "size": 99_270_761,
        "db_name": "markers_go2.db",
        "description": "Public DimOS Unitree Go2 marker replay with robot-view image, odom, and lidar streams.",
    },
    "go2_bigoffice": {
        "path": "data/.lfs/go2_bigoffice.db.tar.gz",
        "oid": "e66f5472e72f370446d8dcd802f70f3c3c07e4e083c5d6a394873877dec4c88d",
        "size": 196_309_743,
        "db_name": "go2_bigoffice.db",
        "description": "Public DimOS Unitree Go2 big-office replay with robot-view image, odom, and lidar streams.",
    }
}


JSON = dict[str, Any]


@dataclass(frozen=True)
class Frame:
    index: int
    ts: float
    pose_x: float
    pose_y: float
    pose_z: float
    qx: float
    qy: float
    qz: float
    qw: float
    jpeg: bytes

    @property
    def yaw(self) -> float:
        return yaw_from_quat(self.qx, self.qy, self.qz, self.qw)


def yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_angle(value: float) -> float:
    while value > math.pi:
        value -= 2.0 * math.pi
    while value < -math.pi:
        value += 2.0 * math.pi
    return value


def relative_delta(current: Frame, future: Frame) -> JSON:
    dx_world = future.pose_x - current.pose_x
    dy_world = future.pose_y - current.pose_y
    yaw = current.yaw
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    dx_body = cos_yaw * dx_world + sin_yaw * dy_world
    dy_body = -sin_yaw * dx_world + cos_yaw * dy_world
    dyaw = wrap_angle(future.yaw - current.yaw)
    return {
        "dx_world_m": round(dx_world, 6),
        "dy_world_m": round(dy_world, 6),
        "dx_body_m": round(dx_body, 6),
        "dy_body_m": round(dy_body, 6),
        "dyaw_rad": round(dyaw, 6),
        "distance_m": round(math.hypot(dx_world, dy_world), 6),
    }


def pose(frame: Frame) -> JSON:
    return {
        "x_m": round(frame.pose_x, 6),
        "y_m": round(frame.pose_y, 6),
        "z_m": round(frame.pose_z, 6),
        "yaw_rad": round(frame.yaw, 6),
        "quat_xyzw": [
            round(frame.qx, 6),
            round(frame.qy, 6),
            round(frame.qz, 6),
            round(frame.qw, 6),
        ],
    }


def safe_extract(tar: tarfile.TarFile, dest: Path) -> None:
    dest = dest.resolve()
    for member in tar.getmembers():
        target = (dest / member.name).resolve()
        if not str(target).startswith(str(dest)):
            raise RuntimeError(f"Refusing unsafe archive path: {member.name}")
    tar.extractall(dest)


def download_lfs_object(dataset: str, cache_dir: Path) -> Path:
    meta = DIMOS_LFS_OBJECTS[dataset]
    out = cache_dir / Path(meta["path"]).name
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size == meta["size"]:
        digest = hashlib.sha256(out.read_bytes()).hexdigest()
        if digest == meta["oid"]:
            return out

    payload = {
        "operation": "download",
        "transfers": ["basic"],
        "objects": [{"oid": meta["oid"], "size": meta["size"]}],
    }
    request = urllib.request.Request(
        LFS_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Accept": "application/vnd.git-lfs+json",
            "Content-Type": "application/vnd.git-lfs+json",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        body = json.loads(response.read())
    href = body["objects"][0]["actions"]["download"]["href"]
    with urllib.request.urlopen(href, timeout=60) as response, out.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)

    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    if digest != meta["oid"]:
        raise RuntimeError(f"SHA256 mismatch for {out}: {digest}")
    return out


def resolve_db(dataset: str, cache_dir: Path) -> Path:
    meta = DIMOS_LFS_OBJECTS[dataset]
    db_path = cache_dir / "extracted" / meta["db_name"]
    if db_path.exists():
        return db_path
    archive = download_lfs_object(dataset, cache_dir)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        safe_extract(tar, db_path.parent)
    if not db_path.exists():
        raise RuntimeError(f"Expected extracted DB at {db_path}")
    return db_path


def read_frames(db_path: Path) -> list[Frame]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        select
          color_image.id,
          color_image.ts,
          color_image.pose_x,
          color_image.pose_y,
          color_image.pose_z,
          color_image.pose_qx,
          color_image.pose_qy,
          color_image.pose_qz,
          color_image.pose_qw,
          color_image_blob.data
        from color_image
        join color_image_blob on color_image.id = color_image_blob.id
        order by color_image.ts asc
        """
    ).fetchall()
    conn.close()
    frames: list[Frame] = []
    for index, row in enumerate(rows):
        pose_values = [row[key] for key in ("pose_x", "pose_y", "pose_z", "pose_qx", "pose_qy", "pose_qz", "pose_qw")]
        if any(value is None for value in pose_values):
            continue
        frames.append(
            Frame(
                index=index,
                ts=float(row["ts"]),
                pose_x=float(row["pose_x"]),
                pose_y=float(row["pose_y"]),
                pose_z=float(row["pose_z"]),
                qx=float(row["pose_qx"]),
                qy=float(row["pose_qy"]),
                qz=float(row["pose_qz"]),
                qw=float(row["pose_qw"]),
                jpeg=bytes(row["data"]),
            )
        )
    return frames


def load_image(frame: Frame, max_width: int) -> Image.Image:
    # DimOS memory2 `jpeg` blobs wrap a sensor_msgs/Image LCM envelope around
    # the actual JPEG payload. Keep the extractor dependency-light by locating
    # the JPEG SOI marker directly instead of importing the full DimOS stack.
    payload = frame.jpeg
    marker = payload.find(b"\xff\xd8\xff")
    if marker >= 0:
        payload = payload[marker:]
    img = Image.open(io.BytesIO(payload)).convert("RGB")
    if img.width > max_width:
        scale = max_width / img.width
        img = img.resize((max_width, max(1, int(img.height * scale))), Image.Resampling.LANCZOS)
    return img


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def pair_preview(current: Image.Image, future: Image.Image, row: JSON) -> Image.Image:
    w = max(current.width, future.width)
    h = max(current.height, future.height)
    pad = 18
    label_h = 90
    out = Image.new("RGB", (w * 2 + pad * 3, h + label_h + pad * 2), (16, 20, 28))
    draw = ImageDraw.Draw(out)
    out.paste(current, (pad + (w - current.width) // 2, pad + label_h))
    out.paste(future, (pad * 2 + w + (w - future.width) // 2, pad + label_h))
    title = font(24, bold=True)
    small = font(18)
    draw.text((pad, pad), "current robot view", fill=(238, 242, 247), font=title)
    draw.text((pad * 2 + w, pad), "future robot view", fill=(238, 242, 247), font=title)
    delta = row["egomotion_delta"]
    subtitle = (
        f"dt={row['horizon_s']:.2f}s  "
        f"dx_body={delta['dx_body_m']:.3f}m  "
        f"dy_body={delta['dy_body_m']:.3f}m  "
        f"dyaw={delta['dyaw_rad']:.3f}rad"
    )
    draw.text((pad, pad + 40), subtitle, fill=(164, 176, 194), font=small)
    return out


def split_for_index(index: int, total: int) -> str:
    frac = index / max(1, total)
    if frac < 0.7:
        return "train"
    if frac < 0.85:
        return "validation"
    return "test"


def selected_datasets(args: argparse.Namespace) -> list[str]:
    if args.datasets:
        names = [item.strip() for item in args.datasets.split(",") if item.strip()]
    else:
        names = [args.dataset]
    unknown = [name for name in names if name not in DIMOS_LFS_OBJECTS]
    if unknown:
        raise RuntimeError(f"Unknown DimOS replay dataset(s): {unknown}")
    return names


def write_jsonl(path: Path, rows: list[JSON]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_imagefolder_views(output_dir: Path, rows: list[JSON]) -> None:
    for split in ("train", "validation", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        split_dir = output_dir / "imagefolder" / split
        split_dir.mkdir(parents=True, exist_ok=True)
        metadata_rows: list[JSON] = []
        for row in split_rows:
            source_preview = output_dir / row["file_name"]
            file_name = Path(row["file_name"]).name
            shutil.copyfile(source_preview, split_dir / file_name)
            metadata = dict(row)
            metadata["file_name"] = file_name
            metadata["pair_preview_source"] = row["file_name"]
            metadata_rows.append(metadata)
        write_jsonl(split_dir / "metadata.jsonl", metadata_rows)


def build(args: argparse.Namespace) -> None:
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = selected_datasets(args)
    rows: list[JSON] = []
    saved_frames: dict[tuple[str, int], str] = {}
    source_frame_counts: dict[str, int] = {}
    source_pair_counts: dict[str, int] = {}
    skipped_sources: dict[str, str] = {}
    frame_dir = output_dir / "images" / "frames"
    preview_dir = output_dir / "images" / "pair_previews"
    frame_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    def save_frame(dataset: str, frame: Frame) -> str:
        key = (dataset, frame.index)
        if key in saved_frames:
            return saved_frames[key]
        rel = f"images/frames/{dataset}_frame_{frame.index:06d}.jpg"
        load_image(frame, args.max_image_width).save(output_dir / rel, quality=90)
        saved_frames[key] = rel
        return rel

    for dataset in datasets:
        db_path = resolve_db(dataset, cache_dir)
        frames = read_frames(db_path)
        if len(frames) < 5:
            skipped_sources[dataset] = f"not enough image rows with usable pose fields in {db_path}: {len(frames)}"
            source_frame_counts[dataset] = len(frames)
            source_pair_counts[dataset] = 0
            continue
        source_frame_counts[dataset] = len(frames)
        timestamps = [frame.ts for frame in frames]
        pair_candidates: list[tuple[Frame, Frame]] = []
        for current in frames[:: args.sample_stride]:
            future_index = bisect.bisect_left(timestamps, current.ts + args.horizon_s)
            if future_index >= len(frames):
                continue
            future = frames[future_index]
            if abs((future.ts - current.ts) - args.horizon_s) > args.max_horizon_error_s:
                continue
            pair_candidates.append((current, future))
            if len(pair_candidates) >= args.max_pairs_per_source:
                break
        source_pair_counts[dataset] = len(pair_candidates)

        for pair_index, (current, future) in enumerate(pair_candidates):
            split = split_for_index(pair_index, len(pair_candidates))
            row_id = f"{dataset}_world_pair_{pair_index:06d}"
            current_rel = save_frame(dataset, current)
            future_rel = save_frame(dataset, future)
            horizon = future.ts - current.ts
            row: JSON = {
                "row_id": row_id,
                "split": split,
                "source_dataset": dataset,
                "source_repo": SOURCE_REPO,
                "source_lfs_path": DIMOS_LFS_OBJECTS[dataset]["path"],
                "current_frame_index": current.index,
                "future_frame_index": future.index,
                "current_image": current_rel,
                "future_image": future_rel,
                "current_ts": round(current.ts, 6),
                "future_ts": round(future.ts, 6),
                "horizon_s": round(horizon, 6),
                "current_pose": pose(current),
                "future_pose": pose(future),
                "egomotion_delta": relative_delta(current, future),
                "world_model_task": "Predict future visual latent from current robot-view image and candidate egomotion/action delta.",
                "score_contract": "score(candidate)=cosine(predicted_future_latent(current_image,candidate_delta), goal_future_latent)",
                "claim_boundary": "Derived from public DimOS replay data. Useful for exploratory latent dynamics/scoring, not safety-certified robot control.",
            }
            preview_rel = f"images/pair_previews/{row_id}.jpg"
            pair_preview(
                Image.open(output_dir / current_rel).convert("RGB"),
                Image.open(output_dir / future_rel).convert("RGB"),
                row,
            ).save(output_dir / preview_rel, quality=90)
            row["file_name"] = preview_rel
            rows.append(row)

    if not rows:
        raise RuntimeError("No world-model pairs produced; adjust horizon or stride")

    for split in ("train", "validation", "test"):
        write_jsonl(output_dir / "data" / f"{split}.jsonl", [row for row in rows if row["split"] == split])
    write_jsonl(output_dir / "metadata.jsonl", rows)
    write_imagefolder_views(output_dir, rows)

    split_counts = {split: sum(1 for row in rows if row["split"] == split) for split in ("train", "validation", "test")}
    summary = {
        "schema_version": 1,
        "dataset_name": "worldforge-go2-dimos-replay-world-pairs",
        "source_dataset": ",".join(datasets),
        "source_datasets": datasets,
        "source_repo": SOURCE_REPO,
        "source_lfs_paths": {name: DIMOS_LFS_OBJECTS[name]["path"] for name in datasets},
        "source_lfs_oids": {name: DIMOS_LFS_OBJECTS[name]["oid"] for name in datasets},
        "source_license": "Apache-2.0 per dimensionalOS/dimos LICENSE",
        "frame_count_in_source": sum(source_frame_counts.values()),
        "source_frame_counts": source_frame_counts,
        "pair_count": len(rows),
        "pair_counts_by_source": source_pair_counts,
        "skipped_sources": skipped_sources,
        "unique_exported_frames": len(saved_frames),
        "split_counts": split_counts,
        "horizon_s_requested": args.horizon_s,
        "sample_stride": args.sample_stride,
        "max_pairs_per_source": args.max_pairs_per_source,
        "max_image_width": args.max_image_width,
        "task": "egomotion-conditioned future visual latent prediction and candidate scoring",
        "limitations": [
            "Small public replay subset, not broad Go2 coverage.",
            "Actions are derived from pose deltas between frames, not joystick command logs.",
            "No safety claims; intended for research demos and WorldForge score-interface experiments.",
        ],
    }
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output_dir / "provenance.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output_dir / "README.md").write_text(dataset_card(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def dataset_card(summary: JSON) -> str:
    sources = "\n".join(
        f"- `{name}`: `{summary['source_lfs_paths'][name]}` / `{summary['source_lfs_oids'][name]}`"
        for name in summary["source_datasets"]
    )
    return f"""---
license: apache-2.0
task_categories:
- robotics
- image-to-image
tags:
- unitree-go2
- dimos
- world-model
- jepa-style
- worldforge
- egomotion
pretty_name: WorldForge Go2 DimOS Replay World Pairs
size_categories:
- n<1K
---

# WorldForge Go2 DimOS Replay World Pairs

This dataset is a compact, derived world-model dataset built from public
[`dimensionalOS/dimos`]({SOURCE_REPO}) Unitree Go2 replay assets.

It is designed for the WorldForge score contract:

```text
current robot-view image + candidate egomotion/action delta
-> predicted future visual latent
-> score against a goal/future latent
```

## Contents

- Source replay frames: `{summary["frame_count_in_source"]}`
- Exported frame pairs: `{summary["pair_count"]}`
- Unique exported frames: `{summary["unique_exported_frames"]}`
- Splits: `{json.dumps(summary["split_counts"], sort_keys=True)}`
- Source pair counts: `{json.dumps(summary["pair_counts_by_source"], sort_keys=True)}`

Source replay assets:

{sources}

Each row includes:

- `current_image`
- `future_image`
- `file_name` side-by-side preview for the Hugging Face image viewer
- timestamps and poses
- `egomotion_delta`
- the explicit world-model scoring contract

The repository also includes `imagefolder/train`, `imagefolder/validation`, and
`imagefolder/test` directories. Each split has pair-preview JPEGs plus a
`metadata.jsonl` file with the same labels, so it can be loaded with the standard
Hugging Face `imagefolder` builder.

## Provenance

The source material comes from `dimensionalOS/dimos`, whose checked-in `LICENSE`
file is Apache License 2.0. This repository currently reports license metadata
as `Other` on GitHub, so users should verify the source license text directly.

## Intended Use

This dataset is intended for:

- small latent-dynamics demos,
- action-conditioned future prediction experiments,
- WorldForge score-provider prototyping,
- educational robotics evidence-trace examples.

## Limitations

- This is not a broad robot foundation dataset.
- It is a small replay-derived dataset.
- The action labels are derived from pose deltas between frames, not raw joystick
  commands.
- It is not suitable for safety validation or direct robot control.
- Indoor replay imagery may contain real-world office context.

## Citation / Attribution

If you use this dataset, attribute both:

- DimensionalOS / DimOS as the source of the public replay data.
- WorldForge Go2 Trace Judge as the derived dataset/scoring package.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a DimOS replay-derived Go2 world-model pair dataset.")
    parser.add_argument("--dataset", choices=sorted(DIMOS_LFS_OBJECTS), default="go2_short")
    parser.add_argument("--datasets", default="", help="Comma-separated replay names. Overrides --dataset when set.")
    parser.add_argument("--cache-dir", default="artifacts/dimos_replay_lfs_cache")
    parser.add_argument("--output-dir", default="hf_dataset_dimos_replay")
    parser.add_argument("--horizon-s", type=float, default=3.0)
    parser.add_argument("--max-horizon-error-s", type=float, default=0.25)
    parser.add_argument("--sample-stride", type=int, default=4)
    parser.add_argument("--max-pairs-per-source", type=int, default=180)
    parser.add_argument("--max-image-width", type=int, default=480)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
