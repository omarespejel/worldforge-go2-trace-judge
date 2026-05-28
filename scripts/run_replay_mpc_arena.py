from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_replay_mpc_demo import load_predictor, score_row  # noqa: E402
from train_dimos_replay_latent_dynamics import extract_embeddings, load_rows  # noqa: E402


JSON = dict[str, Any]
WIDTH = 1920
HEIGHT = 1080
FPS = 24
WHITE = (250, 250, 247)
INK = (15, 18, 22)
MUTED = (83, 91, 101)
FAINT = (224, 226, 229)
GREEN = (0, 145, 97)
BLUE = (24, 95, 191)
RED = (190, 48, 48)


def font(size: int, *, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    if mono:
        candidates = ["/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Monaco.ttf"]
    elif bold:
        candidates = [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
        ]
    else:
        candidates = ["/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf"]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


F_TITLE = font(66, bold=True)
F_H1 = font(52, bold=True)
F_H2 = font(36, bold=True)
F_BODY = font(27)
F_SMALL = font(21)
F_TINY = font(17)
F_MONO = font(22, mono=True)
F_MONO_SMALL = font(18, mono=True)


def read_json(path: Path) -> JSON:
    return json.loads(path.read_text())


def canvas() -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), WHITE)
    draw = ImageDraw.Draw(img)
    for x in range(80, WIDTH, 160):
        draw.line((x, 0, x, HEIGHT), fill=(241, 242, 243), width=1)
    for y in range(80, HEIGHT, 160):
        draw.line((0, y, WIDTH, y), fill=(241, 242, 243), width=1)
    draw.rectangle((0, 0, WIDTH, 12), fill=INK)
    return img


def panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill=(255, 255, 255)) -> None:
    draw.rounded_rectangle(box, radius=12, fill=fill, outline=INK, width=2)


def text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    value: str,
    fnt: ImageFont.FreeTypeFont,
    *,
    fill=INK,
    anchor: str | None = None,
) -> None:
    draw.text(xy, value, font=fnt, fill=fill, anchor=anchor)


def fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    src = Image.open(path).convert("RGB")
    scale = max(size[0] / src.width, size[1] / src.height)
    resized = src.resize(
        (max(1, int(src.width * scale)), max(1, int(src.height * scale))),
        Image.Resampling.LANCZOS,
    )
    left = max(0, (resized.width - size[0]) // 2)
    top = max(0, (resized.height - size[1]) // 2)
    return resized.crop((left, top, left + size[0], top + size[1]))


def image_box(img: Image.Image, path: Path, box: tuple[int, int, int, int]) -> None:
    draw = ImageDraw.Draw(img)
    panel(draw, box)
    content = fit_image(path, (box[2] - box[0], box[3] - box[1]))
    img.paste(content, (box[0], box[1]))
    draw.rounded_rectangle(box, radius=12, outline=INK, width=2)


def choose_arena_examples(scored: list[JSON], count: int) -> list[JSON]:
    positives = [
        item
        for item in scored
        if item["selected_matches_replay"] and float(item["actual_margin"]) > 0.001
    ]
    positives = sorted(positives, key=lambda item: float(item["actual_margin"]), reverse=True)

    selected: list[JSON] = []
    seen_sources: set[str] = set()
    for item in positives:
        source = item["row"]["source_dataset"]
        if source not in seen_sources:
            selected.append(item)
            seen_sources.add(source)
        if len(selected) >= count:
            return selected

    for item in positives:
        if item not in selected:
            selected.append(item)
        if len(selected) >= count:
            return selected

    fallback = sorted(scored, key=lambda item: float(item["actual_margin"]), reverse=True)
    for item in fallback:
        if item not in selected:
            selected.append(item)
        if len(selected) >= count:
            break
    return selected


def trace_payload(result: JSON, model: JSON, run_id: str, index: int) -> JSON:
    row = result["row"]
    return {
        "schema_version": 1,
        "run_id": f"{run_id}-{index:03d}",
        "score_info": {
            "demo_type": "replay_mpc_arena_world_model_score",
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
        },
        "candidate_scores": {
            "scores": result["candidate_scores"],
        },
        "selected_action": {
            "selected": result["selected"],
            "selected_matches_observed_replay_action": result["selected_matches_replay"],
            "actual_vs_best_decoy_margin": result["actual_margin"],
        },
        "outcome_after_execution": {
            "outcome_type": "offline_replay_eval",
            "actual_future_frame": row["future_image"],
            "selected_matches_observed_replay_action": result["selected_matches_replay"],
        },
    }


def render_title(summary: JSON) -> Image.Image:
    img = canvas()
    draw = ImageDraw.Draw(img)
    text(draw, (84, 84), "Replay-MPC Arena", F_TITLE)
    text(
        draw,
        (88, 176),
        "A Go2 latent world model scores candidate futures across held-out DimOS replay scenes.",
        F_BODY,
        fill=MUTED,
    )
    metric_items = [
        (str(summary["dataset_pairs"]), "DimOS Go2 frame pairs"),
        (str(summary["arena_decisions"]), "arena decisions rendered"),
        (f"{summary['selected_match_rate'] * 100:.0f}%", "selected observed replay action"),
    ]
    for i, (value, label) in enumerate(metric_items):
        x = 118 + i * 560
        panel(draw, (x, 360, x + 450, 580))
        text(draw, (x + 34, 408), value, F_H1, fill=GREEN)
        text(draw, (x + 34, 492), label, F_SMALL, fill=MUTED)
    panel(draw, (160, 706, 1760, 842))
    text(draw, (202, 754), "current view + candidate action -> predicted future latent -> score", F_MONO, fill=INK)
    text(draw, (202, 798), "every decision writes score_info, candidate_scores, selected_action, outcome", F_MONO_SMALL, fill=MUTED)
    text(draw, (92, 1008), "github.com/omarespejel/worldforge-go2-trace-judge", F_MONO_SMALL, fill=BLUE)
    return img


def render_decision_card(result: JSON, dataset_dir: Path, model: JSON, index: int) -> Image.Image:
    row = result["row"]
    img = canvas()
    draw = ImageDraw.Draw(img)
    text(draw, (78, 54), f"Replay-MPC decision {index:02d}", F_H1)
    text(
        draw,
        (82, 124),
        f"{row['source_dataset']} / {row['row_id']}",
        F_MONO_SMALL,
        fill=MUTED,
    )
    image_box(img, dataset_dir / row["current_image"], (82, 212, 780, 640))
    image_box(img, dataset_dir / row["future_image"], (820, 212, 1518, 640))
    text(draw, (92, 660), "current robot view", F_SMALL, fill=MUTED)
    text(draw, (830, 660), "actual future replay frame", F_SMALL, fill=MUTED)

    panel(draw, (1558, 212, 1838, 640))
    selected = result["selected"]
    status_color = GREEN if result["selected_matches_replay"] else RED
    text(draw, (1586, 250), "selected", F_MONO_SMALL, fill=MUTED)
    y = 296
    for part in selected["candidate_id"].replace("_", " ").split():
        text(draw, (1586, y), part, F_H2, fill=status_color)
        y += 40
    text(draw, (1586, 452), "score", F_MONO_SMALL, fill=MUTED)
    text(draw, (1586, 492), f"{selected['score']:.4f}", F_H2)
    text(draw, (1586, 568), "margin", F_MONO_SMALL, fill=MUTED)
    text(draw, (1586, 604), f"{result['actual_margin']:+.4f}", F_H2, fill=status_color)

    text(draw, (84, 742), "candidate futures", F_H2)
    scores = result["candidate_scores"][:6]
    max_score = max(item["score"] for item in scores)
    min_score = min(item["score"] for item in scores)
    span = max(1e-6, max_score - min_score)
    y = 802
    for item in scores:
        is_selected = item["candidate_id"] == selected["candidate_id"]
        color = GREEN if is_selected else BLUE
        label = item["candidate_id"].replace("_", " ")
        text(draw, (94, y), label, F_MONO, fill=color if is_selected else INK)
        bar_width = int(700 * ((item["score"] - min_score) / span))
        draw.rectangle((476, y + 4, 1176, y + 28), outline=FAINT, width=1)
        draw.rectangle((476, y + 4, 476 + bar_width, y + 28), fill=color)
        text(draw, (1210, y), f"{item['score']:.4f}", F_MONO, fill=INK)
        if item["selected_in_replay"]:
            text(draw, (1370, y), "observed replay action", F_MONO_SMALL, fill=MUTED)
        y += 43

    text(
        draw,
        (84, 1044),
        f"backbone={model['backbone']}  head={model['head']['head_type']}  trace=decision_{index:03d}",
        F_TINY,
        fill=MUTED,
    )
    return img


def render_contact_sheet(cards: list[Path], output_dir: Path) -> Path:
    thumbs = [Image.open(path).convert("RGB").resize((600, 338), Image.Resampling.LANCZOS) for path in cards[:6]]
    sheet = Image.new("RGB", (1920, 1080), WHITE)
    draw = ImageDraw.Draw(sheet)
    draw.rectangle((0, 0, 1920, 12), fill=INK)
    text(draw, (78, 54), "Replay-MPC Arena", F_H1)
    text(draw, (82, 122), "Multiple held-out Go2 replay scenes, same scoring contract.", F_BODY, fill=MUTED)
    positions = [(78, 206), (660, 206), (1242, 206), (78, 610), (660, 610), (1242, 610)]
    for thumb, pos in zip(thumbs, positions):
        sheet.paste(thumb, pos)
        draw.rounded_rectangle((pos[0], pos[1], pos[0] + 600, pos[1] + 338), radius=10, outline=INK, width=2)
    output = output_dir / "arena_contact_sheet.jpg"
    sheet.save(output, quality=92)
    return output


def write_video(frame_paths: list[Path], output_dir: Path) -> Path:
    frame_dir = output_dir / "video_frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True)
    frame_index = 0
    for path in frame_paths:
        hold = 36 if path.name.startswith("decision_") else 48
        for _ in range(hold):
            shutil.copyfile(path, frame_dir / f"frame_{frame_index:05d}.jpg")
            frame_index += 1
    output = output_dir / "replay_mpc_arena.mp4"
    subprocess.run(
        [
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
            "19",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a multi-scene replay-MPC arena video.")
    parser.add_argument("--dataset-dir", default="hf_dataset_dimos_replay")
    parser.add_argument("--model-dir", default="hf_model_dimos_replay_latent")
    parser.add_argument("--output-dir", default="artifacts/replay_mpc_arena")
    parser.add_argument("--run-id", default="replay-mpc-arena")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--examples", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cards_dir = output_dir / "cards"
    traces_dir = output_dir / "decision_traces"
    cards_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)

    rows = [row for row in load_rows(dataset_dir) if row.get("split") == args.split]
    if not rows:
        raise RuntimeError(f"No rows found for split={args.split!r}")
    model = read_json(model_dir / "model.json")
    embeddings = extract_embeddings(dataset_dir, rows, output_dir, model["backbone"], args.batch_size, args.device)
    predictor = load_predictor(model)
    scored = [score_row(row, embeddings, dataset_dir, predictor) for row in rows]
    examples = choose_arena_examples(scored, args.examples)

    match_count = sum(1 for item in examples if item["selected_matches_replay"])
    summary = {
        "schema_version": 1,
        "run_id": args.run_id,
        "dataset_pairs": len(load_rows(dataset_dir)),
        "scored_split": args.split,
        "scored_rows": len(scored),
        "arena_decisions": len(examples),
        "selected_matches_observed_replay_action": match_count,
        "selected_match_rate": round(match_count / max(1, len(examples)), 6),
        "sources": sorted({item["row"]["source_dataset"] for item in examples}),
        "artifacts": {
            "video": str(output_dir / "replay_mpc_arena.mp4"),
            "contact_sheet": str(output_dir / "arena_contact_sheet.jpg"),
            "decision_traces": str(traces_dir),
        },
    }

    frame_paths: list[Path] = []
    title_path = cards_dir / "title.jpg"
    render_title(summary).save(title_path, quality=92)
    frame_paths.append(title_path)
    decision_cards: list[Path] = []
    for index, result in enumerate(examples, start=1):
        card = render_decision_card(result, dataset_dir, model, index)
        card_path = cards_dir / f"decision_{index:03d}.jpg"
        card.save(card_path, quality=92)
        decision_cards.append(card_path)
        frame_paths.append(card_path)
        trace_dir = traces_dir / f"decision_{index:03d}"
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace = trace_payload(result, model, args.run_id, index)
        for key, payload in trace.items():
            if key in {"schema_version", "run_id"}:
                continue
            (trace_dir / f"{key}.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        (trace_dir / "trace_manifest.json").write_text(
            json.dumps(trace, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    contact_sheet = render_contact_sheet(decision_cards, output_dir)
    video = write_video(frame_paths, output_dir)
    summary["artifacts"]["video"] = str(video)
    summary["artifacts"]["contact_sheet"] = str(contact_sheet)
    (output_dir / "arena_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
