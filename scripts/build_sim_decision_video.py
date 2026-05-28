from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from textwrap import wrap
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "showcase"
FRAME_DIR = OUT_DIR / "sim_decision_frames"
OUT_VIDEO = OUT_DIR / "sim_decision_trace_video.mp4"

WIDTH = 1920
HEIGHT = 1080
FPS = 24

WHITE = (250, 250, 247)
INK = (15, 18, 22)
MUTED = (86, 92, 101)
FAINT = (224, 226, 229)
PANEL = (255, 255, 255)
GREEN = (0, 145, 97)
BLUE = (24, 95, 191)
RED = (190, 48, 48)

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
F_TITLE = font(76, bold=True)
F_H1 = font(54, bold=True)
F_H2 = font(36, bold=True)
F_BODY = font(29)
F_SMALL = font(22)
F_TINY = font(18)
F_MONO = font(24, mono=True)
F_MONO_SMALL = font(19, mono=True)


def load_json(path: Path) -> JSON:
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


def text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    value: str,
    fnt: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int] = INK,
    anchor: str | None = None,
) -> None:
    draw.text(xy, value, font=fnt, fill=fill, anchor=anchor)


def wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    value: str,
    fnt: ImageFont.FreeTypeFont,
    chars: int,
    fill: tuple[int, int, int] = MUTED,
    line_gap: int = 8,
) -> int:
    x, y = xy
    for line in wrap(value, chars):
        draw.text((x, y), line, font=fnt, fill=fill)
        y += fnt.size + line_gap
    return y


def panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: tuple[int, int, int] = PANEL) -> None:
    draw.rounded_rectangle(box, radius=12, fill=fill, outline=INK, width=2)


def fit_image(path: Path, size: tuple[int, int], *, crop: bool = True) -> Image.Image:
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


def image_box(img: Image.Image, path: Path, box: tuple[int, int, int, int], *, crop: bool = True) -> None:
    draw = ImageDraw.Draw(img)
    panel(draw, box)
    content = fit_image(path, (box[2] - box[0], box[3] - box[1]), crop=crop)
    img.paste(content, (box[0], box[1]))
    draw.rounded_rectangle(box, radius=12, outline=INK, width=2)


def header(draw: ImageDraw.ImageDraw, kicker: str, title: str) -> None:
    text(draw, (84, 52), kicker.upper(), F_KICKER, fill=GREEN)
    text(draw, (82, 92), title, F_H1)
    draw.line((82, 170, WIDTH - 82, 170), fill=INK, width=2)


def data_paths() -> tuple[Path, Path, Path, Path, Path]:
    trace = ROOT / "artifacts/replay_mpc_arena/decision_traces/decision_001"
    score_info = load_json(trace / "score_info.json")
    current = ROOT / "hf_dataset_dimos_replay" / score_info["observation"]["current_image"]
    future = ROOT / "hf_dataset_dimos_replay" / score_info["goal"]["future_image"]
    return (
        current,
        future,
        ROOT / "artifacts/dimos_mcp_sim_motion_take2/before.jpg",
        ROOT / "artifacts/dimos_mcp_sim_motion_take2/after_forward.jpg",
        ROOT / "artifacts/dimos_mcp_sim_motion_take2/after_rotate.jpg",
    )


def slide_title(_: float) -> Image.Image:
    img = canvas()
    draw = ImageDraw.Draw(img)
    text(draw, (88, 124), "Decision first. Motion second.", F_TITLE)
    text(draw, (92, 220), "WorldForge-style trace -> DimOS MCP simulation", F_MONO, fill=GREEN)
    panel(draw, (92, 336, 1828, 806))
    text(draw, (140, 398), "1", F_TITLE, fill=GREEN)
    text(draw, (250, 420), "score candidate futures", F_H2)
    text(draw, (140, 548), "2", F_TITLE, fill=GREEN)
    text(draw, (250, 570), "select the best action", F_H2)
    text(draw, (140, 698), "3", F_TITLE, fill=GREEN)
    text(draw, (250, 720), "handoff to DimOS simulation", F_H2)
    text(draw, (94, 940), "github.com/omarespejel/worldforge-go2-trace-judge", F_MONO_SMALL, fill=BLUE)
    return img


def slide_decision(_: float) -> Image.Image:
    img = canvas()
    draw = ImageDraw.Draw(img)
    current, future, *_ = data_paths()
    scores = load_json(ROOT / "artifacts/replay_mpc_arena/decision_traces/decision_001/candidate_scores.json")["scores"]
    selected = load_json(ROOT / "artifacts/replay_mpc_arena/decision_traces/decision_001/selected_action.json")

    header(draw, "replay-mpc", "The model scores six possible futures.")
    image_box(img, current, (92, 238, 706, 626), crop=True)
    image_box(img, future, (92, 666, 706, 924), crop=True)
    text(draw, (112, 202), "current Go2 replay frame", F_MONO_SMALL, fill=MUTED)
    text(draw, (112, 632), "held-out future target", F_MONO_SMALL, fill=MUTED)

    panel(draw, (760, 238, 1828, 924))
    text(draw, (798, 280), "candidate_scores.json", F_MONO, fill=MUTED)
    max_score = max(row["score"] for row in scores)
    min_score = min(row["score"] for row in scores)
    span = max(1e-6, max_score - min_score)
    y = 358
    for row in scores:
        is_selected = row["candidate_id"] == selected["selected"]["candidate_id"]
        color = GREEN if is_selected else INK
        label = "selected action" if is_selected else row["candidate_id"]
        text(draw, (800, y), label, F_MONO, fill=color)
        text(draw, (1778, y), f"{row['score']:.4f}", F_MONO, fill=color, anchor="ra")
        draw.rectangle((802, y + 44, 1778, y + 66), fill=FAINT)
        width = int(260 + 716 * ((row["score"] - min_score) / span))
        draw.rectangle((802, y + 44, 802 + width, y + 66), fill=color)
        y += 84

    text(draw, (802, 860), f"margin over best decoy: {selected['actual_vs_best_decoy_margin']:+.4f}", F_MONO_SMALL, fill=GREEN)
    return img


def slide_trace(_: float) -> Image.Image:
    img = canvas()
    draw = ImageDraw.Draw(img)
    header(draw, "evidence", "The decision is written as replayable JSON.")
    selected = load_json(ROOT / "artifacts/replay_mpc_arena/decision_traces/decision_001/selected_action.json")
    score_info = load_json(ROOT / "artifacts/replay_mpc_arena/decision_traces/decision_001/score_info.json")
    snippet = {
        "score_contract": score_info["score_contract"],
        "world_model": score_info["world_model"]["type"],
        "selected": selected["selected"]["candidate_id"],
        "score": selected["selected"]["score"],
        "trace": "score_info + candidate_scores + selected_action + outcome",
    }
    panel(draw, (104, 236, 1816, 856))
    y = 286
    for line in json.dumps(snippet, indent=2).splitlines():
        text(draw, (146, y), line, F_MONO_SMALL, fill=GREEN if "selected" in line else INK)
        y += 34
    wrapped(
        draw,
        (110, 910),
        "This is the part judges can verify: every decision has the observation, the candidate scores, the selected action, and the outcome artifact.",
        F_BODY,
        106,
        fill=MUTED,
    )
    return img


def slide_sim(_: float) -> Image.Image:
    img = canvas()
    draw = ImageDraw.Draw(img)
    _, _, before, after_forward, after_rotate = data_paths()
    header(draw, "dimos mcp", "Then the decision is handed to simulation.")
    boxes = [(92, 258, 630, 628), (692, 258, 1230, 628), (1292, 258, 1830, 628)]
    labels = ["before", "relative_move", "rotate 90 deg"]
    for path, box, label in zip([before, after_forward, after_rotate], boxes, labels):
        image_box(img, path, box, crop=True)
        draw.rectangle((box[0] + 18, box[1] + 18, box[0] + 260, box[1] + 58), fill=WHITE)
        text(draw, (box[0] + 30, box[1] + 26), label, F_MONO_SMALL)

    text(draw, (646, 420), "->", F_H1, fill=GREEN)
    text(draw, (1246, 420), "->", F_H1, fill=GREEN)
    panel(draw, (148, 724, 1772, 894))
    text(draw, (184, 766), "DimOS MCP command", F_MONO, fill=MUTED)
    text(draw, (184, 820), "WorldForge selected action -> DimOS MCP relative_move(...) -> observe", F_MONO, fill=GREEN)
    text(draw, (184, 952), "WorldForge-style score trace stays outside the robot runtime. DimOS owns execution.", F_BODY, fill=MUTED)
    return img


def slide_close(_: float) -> Image.Image:
    img = canvas()
    draw = ImageDraw.Draw(img)
    text(draw, (88, 118), "The point: inspectable robot autonomy.", F_TITLE)
    panel(draw, (92, 280, 1828, 766))
    bullets = [
        "candidate futures are compared before motion",
        "the selected action is explainable and replayable",
        "DimOS receives a normal MCP command for simulation/execution",
    ]
    y = 356
    for bullet in bullets:
        text(draw, (142, y), "+", F_H2, fill=GREEN)
        text(draw, (204, y + 4), bullet, F_H2)
        y += 118
    text(draw, (94, 900), "Replay-MPC Arena + DimOS MCP simulation proof", F_MONO, fill=GREEN)
    text(draw, (94, 950), "github.com/omarespejel/worldforge-go2-trace-judge", F_MONO_SMALL, fill=BLUE)
    return img


def write_frames() -> None:
    if FRAME_DIR.exists():
        shutil.rmtree(FRAME_DIR)
    FRAME_DIR.mkdir(parents=True)
    timeline = [
        (slide_title, 3.0),
        (slide_decision, 6.0),
        (slide_trace, 5.0),
        (slide_sim, 6.0),
        (slide_close, 3.0),
    ]
    index = 0
    for maker, seconds in timeline:
        frame_count = int(seconds * FPS)
        for frame in range(frame_count):
            image = maker(frame / FPS)
            image.save(FRAME_DIR / f"frame_{index:05d}.jpg", quality=94)
            index += 1


def encode_video() -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        str(FPS),
        "-i",
        str(FRAME_DIR / "frame_%05d.jpg"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(OUT_VIDEO),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_frames()
    encode_video()
    print(OUT_VIDEO)


if __name__ == "__main__":
    main()
