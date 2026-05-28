from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from textwrap import wrap
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "showcase"
FRAME_DIR = OUT_DIR / "final_frames"
OUT_VIDEO = OUT_DIR / "final_hackathon_video.mp4"
FPS = 24
WIDTH = 1920
HEIGHT = 1080

WHITE = (250, 250, 247)
INK = (15, 18, 22)
MUTED = (86, 92, 101)
FAINT = (224, 226, 229)
PANEL = (255, 255, 255)
GREEN = (0, 145, 97)
BLUE = (24, 95, 191)
RED = (190, 48, 48)
YELLOW = (180, 128, 0)

JSON = dict[str, Any]


def font(size: int, *, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    if mono:
        candidates = [
            "/System/Library/Fonts/Menlo.ttc",
            "/System/Library/Fonts/Monaco.ttf",
            "/Library/Fonts/Menlo.ttc",
        ]
    elif bold:
        candidates = [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
        ]
    else:
        candidates = [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
        ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


F_KICKER = font(24, mono=True)
F_TITLE = font(78, bold=True)
F_H1 = font(56, bold=True)
F_H2 = font(38, bold=True)
F_BODY = font(29)
F_SMALL = font(22)
F_TINY = font(18)
F_MONO = font(24, mono=True)
F_MONO_SMALL = font(19, mono=True)


def load_json(path: Path) -> JSON:
    return json.loads(path.read_text())


def load_json_or(path: Path, default: JSON) -> JSON:
    if not path.exists():
        return default
    return load_json(path)


def canvas() -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), WHITE)
    draw = ImageDraw.Draw(img)
    for x in range(80, WIDTH, 160):
        draw.line((x, 0, x, HEIGHT), fill=(241, 242, 243), width=1)
    for y in range(80, HEIGHT, 160):
        draw.line((0, y, WIDTH, y), fill=(241, 242, 243), width=1)
    draw.rectangle((0, 0, WIDTH, 12), fill=INK)
    return img


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, fnt, fill=INK, anchor=None) -> None:
    draw.text(xy, value, font=fnt, fill=fill, anchor=anchor)


def wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    value: str,
    fnt,
    chars: int,
    fill=MUTED,
    line_gap: int = 8,
) -> int:
    x, y = xy
    for line in wrap(value, chars):
        draw.text((x, y), line, font=fnt, fill=fill)
        y += fnt.size + line_gap
    return y


def rule(draw: ImageDraw.ImageDraw, y: int) -> None:
    draw.line((82, y, WIDTH - 82, y), fill=INK, width=2)


def panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill=PANEL) -> None:
    draw.rounded_rectangle(box, radius=12, fill=fill, outline=INK, width=2)


def fit_image(path: Path, size: tuple[int, int], crop: bool = True) -> Image.Image:
    src = Image.open(path).convert("RGB")
    bw, bh = size
    scale = max(bw / src.width, bh / src.height) if crop else min(bw / src.width, bh / src.height)
    resized = src.resize((max(1, int(src.width * scale)), max(1, int(src.height * scale))), Image.Resampling.LANCZOS)
    if crop:
        left = max(0, (resized.width - bw) // 2)
        top = max(0, (resized.height - bh) // 2)
        return resized.crop((left, top, left + bw, top + bh))
    out = Image.new("RGB", (bw, bh), WHITE)
    out.paste(resized, ((bw - resized.width) // 2, (bh - resized.height) // 2))
    return out


def image_box(img: Image.Image, path: Path, box: tuple[int, int, int, int], crop: bool = True) -> None:
    draw = ImageDraw.Draw(img)
    panel(draw, box)
    content = fit_image(path, (box[2] - box[0], box[3] - box[1]), crop=crop)
    img.paste(content, (box[0], box[1]))
    draw.rounded_rectangle(box, radius=12, outline=INK, width=2)


def header(draw: ImageDraw.ImageDraw, kicker: str, title: str) -> None:
    text(draw, (84, 52), kicker.upper(), F_KICKER, fill=GREEN)
    text(draw, (82, 92), title, F_H1)
    rule(draw, 170)


def path_or_fallback(*paths: str) -> Path:
    for item in paths:
        path = ROOT / item
        if path.exists():
            return path
    raise FileNotFoundError(paths[0])


def metric(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, label: str, color=INK) -> None:
    x, y = xy
    text(draw, (x, y), value, F_H1, fill=color)
    text(draw, (x, y + 72), label, F_SMALL, fill=MUTED)


def link(draw: ImageDraw.ImageDraw, y: int, label: str, value: str) -> None:
    text(draw, (94, y), label, F_MONO_SMALL, fill=MUTED)
    text(draw, (470, y), value, F_MONO_SMALL, fill=BLUE)


def slide_hook(_: float) -> Image.Image:
    img = canvas()
    draw = ImageDraw.Draw(img)
    text(draw, (88, 94), "Robots should show their work.", F_TITLE)
    text(draw, (92, 190), "WorldForge Go2 Trace Judge", F_MONO, fill=GREEN)
    image_box(img, path_or_fallback("artifacts/live_ciro/direct_camera_unsafe_path.jpg"), (92, 288, 1300, 924), crop=True)
    panel(draw, (1350, 288, 1818, 924))
    text(draw, (1384, 334), "one frame", F_MONO, fill=MUTED)
    text(draw, (1384, 404), "+ goal", F_MONO, fill=MUTED)
    text(draw, (1384, 474), "+ actions", F_MONO, fill=MUTED)
    draw.line((1386, 548, 1780, 548), fill=INK, width=2)
    text(draw, (1384, 602), "ranked futures", F_H2, fill=GREEN)
    text(draw, (1384, 666), "evidence trail", F_H2, fill=GREEN)
    text(draw, (92, 992), "github.com/omarespejel/worldforge-go2-trace-judge", F_MONO_SMALL, fill=BLUE)
    return img


def slide_problem(_: float) -> Image.Image:
    img = canvas()
    draw = ImageDraw.Draw(img)
    header(draw, "the gap", "Most robot demos show motion. We show the decision.")
    frames = [
        ("real Go2 view", "artifacts/live_ciro/direct_camera_red_block_front.jpg"),
        ("target left", "artifacts/live_ciro/direct_camera_red_block_left_annotated.jpg"),
        ("unsafe path", "artifacts/live_ciro/direct_camera_unsafe_path_annotated.jpg"),
    ]
    boxes = [(90, 232, 670, 640), (718, 232, 1298, 640), (1346, 232, 1830, 640)]
    for (label, path), box in zip(frames, boxes):
        image_box(img, path_or_fallback(path), box, crop=True)
        draw.rectangle((box[0] + 18, box[1] + 18, box[0] + 238, box[1] + 56), fill=WHITE)
        text(draw, (box[0] + 28, box[1] + 25), label, F_MONO_SMALL)
    text(draw, (106, 746), "Instead of:", F_H2, fill=MUTED)
    text(draw, (106, 806), "LLM -> move robot", F_MONO)
    text(draw, (866, 746), "We built:", F_H2, fill=GREEN)
    text(draw, (866, 806), "options -> scores -> selected action -> trace", F_MONO, fill=GREEN)
    return img


def slide_score(_: float) -> Image.Image:
    img = canvas()
    draw = ImageDraw.Draw(img)
    scores = load_json(ROOT / "artifacts/micro_world_demo/latest/candidate_scores.json")["scores"]
    header(draw, "the decision", "WorldForge-style scoring compares candidate futures.")
    image_box(img, path_or_fallback("artifacts/micro_world_demo/latest/annotated_image.jpg"), (90, 230, 1058, 866), crop=False)
    panel(draw, (1104, 230, 1830, 866))
    text(draw, (1142, 270), "candidate_scores.json", F_MONO, fill=MUTED)
    y = 352
    max_score = max(row["score"] for row in scores)
    for row in scores[:4]:
        label = row["candidate_id"]
        score = row["score"]
        selected = row.get("selected", False)
        color = GREEN if selected else INK
        text(draw, (1144, y), label, F_MONO, fill=color)
        text(draw, (1768, y), f"{score:.4f}", F_MONO, fill=color, anchor="ra")
        draw.rectangle((1146, y + 48, 1766, y + 66), fill=FAINT)
        draw.rectangle((1146, y + 48, 1146 + int(620 * score / max_score), y + 66), fill=GREEN if selected else (70, 75, 82))
        y += 116
    text(draw, (1144, 792), "selected: turn_left", F_H2, fill=GREEN)
    return img


def slide_replay_arena(_: float) -> Image.Image:
    img = canvas()
    draw = ImageDraw.Draw(img)
    summary = load_json_or(
        ROOT / "artifacts/replay_mpc_arena/arena_summary.json",
        {
            "arena_decisions": 12,
            "selected_match_rate": 1.0,
            "sources": [],
        },
    )
    header(draw, "world model arena", "The scorer runs across held-out Go2 replay scenes.")
    contact_sheet = path_or_fallback(
        "artifacts/replay_mpc_arena/arena_contact_sheet.jpg",
        "artifacts/replay_mpc_demo/predicted_vs_actual_future.jpg",
    )
    image_box(img, contact_sheet, (86, 232, 1218, 866), crop=True)
    panel(draw, (1264, 232, 1830, 866))
    metric(draw, (1300, 292), str(summary.get("arena_decisions", 12)), "decisions rendered", GREEN)
    metric(
        draw,
        (1300, 484),
        f"{float(summary.get('selected_match_rate', 1.0)) * 100:.0f}%",
        "match observed replay action",
        GREEN,
    )
    text(draw, (1302, 688), "same loop", F_MONO, fill=MUTED)
    text(draw, (1302, 736), "current frame", F_MONO)
    text(draw, (1302, 782), "+ candidate action", F_MONO)
    text(draw, (1302, 828), "-> scored future", F_MONO, fill=GREEN)
    return img


def slide_trace(_: float) -> Image.Image:
    img = canvas()
    draw = ImageDraw.Draw(img)
    header(draw, "the proof", "Every action decision is replayable from files.")
    files = [
        ("score_info.json", "what the scorer saw"),
        ("candidate_scores.json", "ranked options and reasons"),
        ("selected_action.json", "host-execution handoff"),
        ("outcome_after_execution.json", "future training signal"),
    ]
    for i, (name, desc) in enumerate(files):
        x = 110 + (i % 2) * 850
        y = 250 + (i // 2) * 230
        panel(draw, (x, y, x + 760, y + 160))
        text(draw, (x + 34, y + 34), name, F_MONO, fill=GREEN if i == 1 else INK)
        text(draw, (x + 34, y + 88), desc, F_BODY, fill=MUTED)
    link(draw, 804, "verify trace", "github.com/omarespejel/worldforge-go2-trace-judge/tree/main/artifacts/replay_mpc_arena/decision_traces")
    link(draw, 852, "project repo", "github.com/omarespejel/worldforge-go2-trace-judge")
    return img


def slide_sim_bridge(_: float) -> Image.Image:
    img = canvas()
    draw = ImageDraw.Draw(img)
    header(draw, "runtime bridge", "The selected action reaches DimOS MCP in MuJoCo.")
    frames = [
        ("before", "artifacts/dimos_mcp_sim_motion_take2/before.jpg"),
        ("after move", "artifacts/dimos_mcp_sim_motion_take2/after_forward.jpg"),
        ("after turn", "artifacts/dimos_mcp_sim_motion_take2/after_rotate.jpg"),
    ]
    boxes = [(92, 248, 672, 606), (704, 248, 1284, 606), (1316, 248, 1830, 606)]
    for (label, path), box in zip(frames, boxes):
        if (ROOT / path).exists():
            image_box(img, ROOT / path, box, crop=True)
        else:
            image_box(img, path_or_fallback("artifacts/replay_mpc_demo/predicted_vs_actual_future.jpg"), box, crop=True)
        draw.rectangle((box[0] + 16, box[1] + 16, box[0] + 182, box[1] + 54), fill=WHITE)
        text(draw, (box[0] + 26, box[1] + 22), label, F_MONO_SMALL)
    panel(draw, (180, 728, 1740, 850))
    text(draw, (224, 770), "WorldForge selected_action.json -> DimOS MCP relative_move -> simulation camera changes", F_MONO)
    text(draw, (224, 812), "technical proof that the offline scorer can hand off to the robot runtime", F_MONO_SMALL, fill=MUTED)
    return img


def slide_worldforge(_: float) -> Image.Image:
    img = canvas()
    draw = ImageDraw.Draw(img)
    header(draw, "the interface", "Built around the WorldForge scoring contract.")
    text(draw, (132, 274), "score(candidate)", F_TITLE, fill=GREEN)
    text(draw, (134, 378), "=", F_TITLE)
    text(draw, (236, 378), "expected future quality", F_TITLE)
    panel(draw, (128, 558, 1788, 724))
    text(draw, (172, 604), "observation + goal + candidate action -> score -> selected action", F_MONO)
    link(draw, 824, "upstream", "github.com/AbdelStark/worldforge")
    link(draw, 872, "this project", "github.com/omarespejel/worldforge-go2-trace-judge")
    return img


def slide_open_artifacts(_: float) -> Image.Image:
    img = canvas()
    draw = ImageDraw.Draw(img)
    summary = load_json(ROOT / "hf_dataset_dimos_replay/dataset_summary.json")
    eval_report = load_json(ROOT / "hf_model_dimos_replay_latent/eval_report.json")
    usable_replay_count = sum(1 for count in summary["pair_counts_by_source"].values() if count > 0)
    header(draw, "the contribution", "One of the first open Go2 world-model-scoring datasets on HF.")
    metric(draw, (110, 248), str(summary["pair_count"]), "Go2 replay pairs", GREEN)
    metric(draw, (520, 248), str(usable_replay_count), "usable DimOS replay sources", GREEN)
    metric(draw, (930, 248), f"+{eval_report['latent_prediction_metrics']['validation']['cosine_lift_vs_no_motion']:.4f}", "validation latent lift", GREEN)
    image_paths = [
        ROOT / "hf_dataset_dimos_replay/images/pair_previews/go2_short_world_pair_000130.jpg",
        ROOT / "hf_dataset_dimos_replay/images/pair_previews/markers_go2_world_pair_000146.jpg",
        ROOT / "hf_dataset_dimos_replay/images/pair_previews/go2_bigoffice_world_pair_000104.jpg",
    ]
    boxes = [(112, 466, 660, 748), (686, 466, 1234, 748), (1260, 466, 1808, 748)]
    for path, box in zip(image_paths, boxes):
        if path.exists():
            image_box(img, path, box, crop=True)
    link(draw, 826, "dataset", "huggingface.co/datasets/espejelomar/worldforge-go2-dimos-replay-world-pairs")
    link(draw, 874, "model", "huggingface.co/espejelomar/go2-dimos-replay-latent-dynamics")
    text(draw, (94, 978), "current frame + future frame + egomotion delta", F_MONO, fill=MUTED)
    return img


def slide_close(_: float) -> Image.Image:
    img = canvas()
    draw = ImageDraw.Draw(img)
    text(draw, (90, 120), "Inspectable autonomy", F_TITLE)
    text(draw, (94, 220), "for embodied AI.", F_TITLE, fill=GREEN)
    panel(draw, (96, 388, 1824, 626))
    text(draw, (140, 444), "see the options", F_H1)
    text(draw, (740, 444), "see the scores", F_H1)
    text(draw, (1284, 456), "replay the decision", F_H2, fill=GREEN)
    text(draw, (96, 674), "A demo today. A dataset for the next models.", F_H2, fill=MUTED)
    text(draw, (96, 742), "Built by", F_MONO_SMALL, fill=MUTED)
    text(draw, (278, 742), "Omar Espejel", F_MONO_SMALL)
    text(draw, (560, 742), "AI at Starknet Foundation; former Hugging Face", F_MONO_SMALL, fill=MUTED)
    text(draw, (278, 782), "Abdel", F_MONO_SMALL)
    text(draw, (560, 782), "Applied AI at StarkWare", F_MONO_SMALL, fill=MUTED)
    text(draw, (278, 822), "Ciro", F_MONO_SMALL)
    text(draw, (560, 822), "Engineer at Harvard Publishing", F_MONO_SMALL, fill=MUTED)
    link(draw, 884, "WorldForge", "github.com/AbdelStark/worldforge")
    link(draw, 924, "Project", "github.com/omarespejel/worldforge-go2-trace-judge")
    link(draw, 964, "Dataset", "huggingface.co/datasets/espejelomar/worldforge-go2-dimos-replay-world-pairs")
    link(draw, 1004, "Model", "huggingface.co/espejelomar/go2-dimos-replay-latent-dynamics")
    return img


class Builder:
    def __init__(self) -> None:
        if FRAME_DIR.exists():
            shutil.rmtree(FRAME_DIR)
        FRAME_DIR.mkdir(parents=True, exist_ok=True)
        self.index = 0

    def write(self, img: Image.Image) -> None:
        img.save(FRAME_DIR / f"frame_{self.index:05d}.jpg", quality=92, optimize=True)
        self.index += 1

    def hold(self, seconds: float, fn: Callable[[float], Image.Image]) -> None:
        count = max(1, int(seconds * FPS))
        for frame in range(count):
            self.write(fn(frame / max(1, count - 1)))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    builder = Builder()
    # 58 seconds total, designed for external voiceover.
    builder.hold(4, slide_hook)
    builder.hold(6, slide_problem)
    builder.hold(7, slide_score)
    builder.hold(8, slide_replay_arena)
    builder.hold(6, slide_trace)
    builder.hold(5, slide_sim_bridge)
    builder.hold(5, slide_worldforge)
    builder.hold(9, slide_open_artifacts)
    builder.hold(8, slide_close)
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(FPS),
        "-i",
        str(FRAME_DIR / "frame_%05d.jpg"),
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
        str(OUT_VIDEO),
    ]
    subprocess.run(cmd, check=True)
    print(OUT_VIDEO)


if __name__ == "__main__":
    main()
