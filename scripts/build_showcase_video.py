from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "showcase"
FRAME_DIR = OUT_DIR / "frames"
OUT_VIDEO = OUT_DIR / "worldforge_go2_trace_judge_showcase.mp4"
FPS = 24
WIDTH = 1920
HEIGHT = 1080

BG = (11, 15, 22)
PANEL = (24, 31, 43)
PANEL_2 = (30, 39, 54)
TEXT = (238, 242, 247)
MUTED = (165, 177, 195)
GREEN = (76, 218, 151)
BLUE = (82, 155, 255)
YELLOW = (250, 207, 82)
RED = (255, 94, 94)
LINE = (58, 72, 95)


def font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    candidates = []
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
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


F_TITLE = font(84, bold=True)
F_H1 = font(54, bold=True)
F_H2 = font(40, bold=True)
F_BODY = font(30)
F_SMALL = font(24)
F_TINY = font(20)
F_MONO = font(22, mono=True)


def ease(x: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * max(0.0, min(1.0, x)))


def new_canvas() -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)
    # Subtle horizon bands.
    for y in range(HEIGHT):
        t = y / HEIGHT
        r = int(BG[0] + 12 * t)
        g = int(BG[1] + 14 * t)
        b = int(BG[2] + 18 * t)
        draw.line([(0, y), (WIDTH, y)], fill=(r, g, b))
    return img


def rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill=PANEL, outline=LINE, radius=24) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=2)


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], content: str, fnt, fill=TEXT, anchor=None) -> None:
    draw.text(xy, content, font=fnt, fill=fill, anchor=anchor)


def wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], content: str, fnt, width_chars: int, fill=MUTED, line_gap=8) -> int:
    x, y = xy
    for line in wrap(content, width_chars):
        draw.text((x, y), line, font=fnt, fill=fill)
        y += fnt.size + line_gap
    return y


def fit_image(path: Path, box: tuple[int, int, int, int], crop: bool = False) -> Image.Image:
    src = Image.open(path).convert("RGB")
    bw, bh = box[2] - box[0], box[3] - box[1]
    sw, sh = src.size
    scale = max(bw / sw, bh / sh) if crop else min(bw / sw, bh / sh)
    nw, nh = max(1, int(sw * scale)), max(1, int(sh * scale))
    src = src.resize((nw, nh), Image.Resampling.LANCZOS)
    if crop:
        left = (nw - bw) // 2
        top = (nh - bh) // 2
        return src.crop((left, top, left + bw, top + bh))
    canvas = Image.new("RGB", (bw, bh), (8, 11, 17))
    canvas.paste(src, ((bw - nw) // 2, (bh - nh) // 2))
    return canvas


def paste_panel_image(img: Image.Image, path: Path, box: tuple[int, int, int, int], crop: bool = False) -> None:
    draw = ImageDraw.Draw(img)
    rounded(draw, box, fill=(8, 11, 17), radius=20)
    fitted = fit_image(path, box, crop=crop)
    img.paste(fitted, (box[0], box[1]))
    draw.rounded_rectangle(box, radius=20, outline=LINE, width=2)


def load_json(path: Path):
    return json.loads(path.read_text())


def score_rows(step_dir: Path) -> list[dict]:
    data = load_json(step_dir / "candidate_scores.json")
    return sorted(data["scores"], key=lambda row: row["score"], reverse=True)


def draw_scores(draw: ImageDraw.ImageDraw, rows: list[dict], box: tuple[int, int, int, int], title: str) -> None:
    rounded(draw, box, fill=PANEL_2, radius=22)
    x1, y1, x2, y2 = box
    text(draw, (x1 + 30, y1 + 24), title, F_H2)
    y = y1 + 96
    max_score = max(row["score"] for row in rows) or 1.0
    selected = rows[0]["candidate_id"]
    for row in rows:
        cid = row["candidate_id"]
        score = float(row["score"])
        color = GREEN if cid == selected else BLUE
        label = cid.replace("_", " ")
        text(draw, (x1 + 34, y), label, F_SMALL, fill=TEXT if cid == selected else MUTED)
        text(draw, (x2 - 34, y), f"{score:.3f}", F_SMALL, fill=TEXT, anchor="ra")
        bar_x = x1 + 34
        bar_y = y + 38
        bar_w = x2 - x1 - 68
        draw.rounded_rectangle((bar_x, bar_y, bar_x + bar_w, bar_y + 18), radius=9, fill=(12, 17, 25))
        draw.rounded_rectangle((bar_x, bar_y, bar_x + int(bar_w * score / max_score), bar_y + 18), radius=9, fill=color)
        if cid == selected:
            text(draw, (bar_x, bar_y + 30), "selected", F_TINY, fill=GREEN)
        y += 96


def draw_header(draw: ImageDraw.ImageDraw, title: str, subtitle: str | None = None) -> None:
    text(draw, (84, 58), title, F_H1)
    if subtitle:
        text(draw, (86, 122), subtitle, F_SMALL, fill=MUTED)
    draw.line((84, 164, WIDTH - 84, 164), fill=LINE, width=2)


def add_badge(draw: ImageDraw.ImageDraw, xy: tuple[int, int], label: str, fill=BLUE) -> None:
    x, y = xy
    w = int(draw.textlength(label, font=F_SMALL)) + 34
    draw.rounded_rectangle((x, y, x + w, y + 44), radius=22, fill=(fill[0] // 4, fill[1] // 4, fill[2] // 4), outline=fill, width=2)
    text(draw, (x + 17, y + 9), label, F_SMALL, fill=TEXT)


class VideoBuilder:
    def __init__(self) -> None:
        if FRAME_DIR.exists():
            shutil.rmtree(FRAME_DIR)
        FRAME_DIR.mkdir(parents=True, exist_ok=True)
        self.index = 0

    def write(self, img: Image.Image) -> None:
        path = FRAME_DIR / f"frame_{self.index:05d}.jpg"
        img.save(path, quality=91, optimize=True)
        self.index += 1

    def hold(self, seconds: float, render) -> None:
        frames = max(1, int(seconds * FPS))
        for i in range(frames):
            t = i / max(1, frames - 1)
            self.write(render(t))


def title_slide(t: float) -> Image.Image:
    img = new_canvas()
    draw = ImageDraw.Draw(img)
    y = int(250 - 20 * (1 - ease(t)))
    text(draw, (96, y), "WorldForge", F_H1, fill=BLUE)
    text(draw, (96, y + 72), "Go2 Trace Judge", F_TITLE)
    wrapped(
        draw,
        (102, y + 190),
        "Inspectable robot decisions: what the Go2 saw, which actions it considered, why one won, and what evidence was saved.",
        F_BODY,
        72,
        fill=MUTED,
        line_gap=10,
    )
    add_badge(draw, (104, y + 340), "real Go2 camera")
    add_badge(draw, (330, y + 340), "candidate scores", GREEN)
    add_badge(draw, (590, y + 340), "replayable evidence", YELLOW)
    text(draw, (96, HEIGHT - 104), "Transparent scorer now. Learned world model later.", F_H2, fill=TEXT)
    return img


def live_material_slide(t: float) -> Image.Image:
    img = new_canvas()
    draw = ImageDraw.Draw(img)
    draw_header(draw, "Real Venue Material", "Captured from the Unitree Go2 camera, then converted into trace artifacts.")
    paths = [
        ("red target left", ROOT / "artifacts/live_ciro/direct_camera_red_block_left_annotated.jpg"),
        ("red target right", ROOT / "artifacts/live_ciro/direct_camera_red_block_right_floor_aware_annotated.jpg"),
        ("unsafe markers", ROOT / "artifacts/live_ciro/direct_camera_unsafe_path_annotated.jpg"),
        ("raw POV frame", ROOT / "data/go2_camera_photo.jpg"),
    ]
    boxes = [(84, 220, 912, 530), (1008, 220, 1836, 530), (84, 620, 912, 930), (1008, 620, 1836, 930)]
    for idx, ((label, path), box) in enumerate(zip(paths, boxes)):
        alpha_t = ease(min(1, max(0, t * 2.2 - idx * 0.25)))
        paste_panel_image(img, path, box, crop=True)
        draw.rounded_rectangle((box[0] + 16, box[1] + 16, box[0] + 310, box[1] + 62), radius=18, fill=(0, 0, 0))
        text(draw, (box[0] + 34, box[1] + 27), label, F_SMALL, fill=(int(TEXT[0]*alpha_t), int(TEXT[1]*alpha_t), int(TEXT[2]*alpha_t)))
    return img


def replay_slide(t: float, summary: dict) -> Image.Image:
    steps = summary["steps"]
    idx = min(len(steps) - 1, int(t * (len(steps) - 1)))
    step = steps[idx]
    step_dir = ROOT / step["step_dir"]
    img = new_canvas()
    draw = ImageDraw.Draw(img)
    draw_header(draw, "Robot POV Replay", "Each frame becomes observation, candidates, scores, and selected action.")
    paste_panel_image(img, ROOT / step["annotated_frame"], (72, 204, 1290, 916), crop=False)
    rows = score_rows(step_dir)
    draw_scores(draw, rows, (1330, 204, 1848, 700), "Candidate Scores")
    selected = step["selected_candidate_id"].replace("_", " ")
    target = step["target"]
    rounded(draw, (1330, 730, 1848, 916), fill=PANEL_2, radius=22)
    text(draw, (1360, 760), f"step {step['frame_index']:02d} / {len(steps)}", F_SMALL, fill=MUTED)
    text(draw, (1360, 804), f"selected: {selected}", F_H2, fill=GREEN)
    text(draw, (1360, 864), f"target x={target['center_x']:.2f} conf={target['confidence']:.2f}", F_SMALL, fill=TEXT)
    # Progress line.
    px1, py, px2 = 84, 980, 1836
    draw.line((px1, py, px2, py), fill=LINE, width=8)
    draw.line((px1, py, px1 + int((px2 - px1) * idx / max(1, len(steps) - 1)), py), fill=GREEN, width=8)
    return img


def score_explain_slide(t: float, summary: dict) -> Image.Image:
    examples = [
        ("Target left -> turn left", 1),
        ("Target centered -> move forward", 38),
        ("Target right -> turn right", 40),
    ]
    idx = min(2, int(t * 3))
    label, step_no = examples[idx]
    step_dir = ROOT / "artifacts/replay_run/trace" / f"step_{step_no:02d}"
    frame = ROOT / "artifacts/replay_run/annotated_frames" / f"frame_{step_no:04d}.jpg"
    img = new_canvas()
    draw = ImageDraw.Draw(img)
    draw_header(draw, "The Important Part: It Compares Options", "Not just action output; the trace records rejected alternatives.")
    paste_panel_image(img, frame, (84, 226, 1010, 820), crop=False)
    rows = score_rows(step_dir)
    draw_scores(draw, rows, (1050, 226, 1836, 820), label)
    y = 874
    text(draw, (104, y), "Score = goal alignment + information gain + progress - risk - execution cost", F_BODY, fill=TEXT)
    text(draw, (104, y + 54), "This is the world-model boundary: observation + goal + candidate action -> score.", F_SMALL, fill=MUTED)
    return img


def unsafe_slide(t: float) -> Image.Image:
    img = new_canvas()
    draw = ImageDraw.Draw(img)
    draw_header(draw, "Unsafe Markers Change The Decision", "Green/yellow blocks were treated as calibrated unsafe colors in the live scene.")
    paste_panel_image(img, ROOT / "artifacts/live_ciro/direct_camera_unsafe_path_annotated.jpg", (78, 216, 1160, 900), crop=False)
    rows = score_rows(ROOT / "artifacts/live_ciro_detection/direct-camera-unsafe-path/step_01")
    draw_scores(draw, rows, (1210, 216, 1848, 720), "Live Frame Scores")
    rounded(draw, (1210, 752, 1848, 900), fill=PANEL_2, radius=22)
    text(draw, (1244, 784), "Forward was penalized", F_H2, fill=YELLOW)
    wrapped(draw, (1246, 844), "The scorer saw target progress, but lowered the forward action because the unsafe marker occupied the path.", F_SMALL, 42, fill=MUTED)
    return img


def evidence_slide(t: float) -> Image.Image:
    img = new_canvas()
    draw = ImageDraw.Draw(img)
    draw_header(draw, "Evidence Trail", "Every decision writes the same artifact shape.")
    cards = [
        ("score_info.json", "observation + goal + action candidates"),
        ("candidate_scores.json", "ranked costs and reasons"),
        ("selected_action.json", "the one action to execute"),
        ("outcome_after_execution.json", "what happened next"),
    ]
    for i, (name, desc) in enumerate(cards):
        x = 120 + (i % 2) * 860
        y = 250 + (i // 2) * 300
        rounded(draw, (x, y, x + 760, y + 220), fill=PANEL_2, radius=24)
        text(draw, (x + 34, y + 30), name, F_H2, fill=GREEN if i == 0 else TEXT)
        wrapped(draw, (x + 36, y + 94), desc, F_BODY, 38, fill=MUTED)
        text(draw, (x + 36, y + 160), "replayable | inspectable | trainable", F_SMALL, fill=BLUE)
    snippet = '{ "candidate_id": "turn_left", "score": 0.798, "reason": "turns target toward image center" }'
    text(draw, (126, 920), snippet, F_MONO, fill=(205, 226, 255))
    return img


def dataset_slide(t: float) -> Image.Image:
    summary = load_json(ROOT / "dataset/go2_trace_dataset_summary.json")
    audit = load_json(ROOT / "artifacts/dataset_audit/audit_summary.json")
    img = new_canvas()
    draw = ImageDraw.Draw(img)
    draw_header(draw, "From Demo To Training Data", "The repo exports the candidate/outcome shape needed for a future learned scorer.")
    stats = [
        ("frames", str(summary["step_count"])),
        ("candidate rows", str(summary["candidate_row_count"])),
        ("usable steps", str(audit["usable_step_count"])),
        ("selected actions", ", ".join(f"{k}:{v}" for k, v in summary["selected_candidate_counts"].items())),
    ]
    for i, (k, v) in enumerate(stats):
        x = 120 + (i % 2) * 860
        y = 250 + (i // 2) * 250
        rounded(draw, (x, y, x + 760, y + 180), fill=PANEL_2, radius=24)
        text(draw, (x + 34, y + 30), v, F_TITLE if i < 3 else F_H2, fill=GREEN if i < 3 else BLUE)
        text(draw, (x + 36, y + 124), k, F_BODY, fill=MUTED)
    wrapped(draw, (124, 830), "Claim boundary: this is not a trained Go2 world model. It is the evidence/data contract that makes one possible.", F_H2, 76, fill=TEXT)
    return img


def live_control_slide(t: float) -> Image.Image:
    img = new_canvas()
    draw = ImageDraw.Draw(img)
    draw_header(draw, "Host-Owned Robot Control", "WorldForge judges decisions; DimOS/Unitree host owns physical execution and safety.")
    paste_panel_image(img, ROOT / "artifacts/final_rise_move/before_after.jpg", (90, 250, 1030, 792), crop=False)
    rounded(draw, (1090, 250, 1830, 792), fill=PANEL_2, radius=24)
    lines = [
        "Stand / recover / balance accepted",
        "Bounded Move commands accepted",
        "StopMove and Sit returned status 0",
        "Closed-loop walking still needs hardening",
    ]
    y = 304
    for i, line in enumerate(lines):
        color = GREEN if i < 3 else YELLOW
        draw.ellipse((1128, y + 8, 1154, y + 34), fill=color)
        text(draw, (1176, y), line, F_BODY, fill=TEXT)
        y += 96
    wrapped(draw, (1128, 690), "This framing is honest and production-grade: autonomy evidence without pretending away runtime limits.", F_SMALL, 48, fill=MUTED)
    return img


def closing_slide(t: float) -> Image.Image:
    img = new_canvas()
    draw = ImageDraw.Draw(img)
    text(draw, (96, 210), "What We Shipped", F_TITLE)
    items = [
        "real Go2 camera replay",
        "annotated target tracking",
        "candidate scoring traces",
        "human review pack",
        "trace dataset + tiny ranker smoke test",
        "runbook for the next live robot window",
    ]
    y = 360
    for item in items:
        draw.rounded_rectangle((108, y + 6, 134, y + 32), radius=13, fill=GREEN)
        text(draw, (160, y), item, F_BODY)
        y += 74
    text(draw, (96, 890), "github.com/omarespejel/worldforge-go2-trace-judge", F_H2, fill=BLUE)
    return img


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = load_json(ROOT / "artifacts/replay_run/summary.json")
    builder = VideoBuilder()
    builder.hold(4.0, title_slide)
    builder.hold(8.0, live_material_slide)
    builder.hold(18.0, lambda t: replay_slide(t, summary))
    builder.hold(15.0, lambda t: score_explain_slide(t, summary))
    builder.hold(10.0, unsafe_slide)
    builder.hold(10.0, evidence_slide)
    builder.hold(8.0, dataset_slide)
    builder.hold(8.0, live_control_slide)
    builder.hold(6.0, closing_slide)

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
        "20",
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
