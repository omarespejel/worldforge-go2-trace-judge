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
VIDEO_FRAMES = OUT_DIR / "robot_video_frames"
OUT_VIDEO = OUT_DIR / "final_hackathon_video.mp4"
SOURCE_ROBOT_VIDEO = Path("/Users/espejelomar/Downloads/output.mp4")
DIMOS_MEDIA_PREVIEW = ROOT / "artifacts" / "third_party" / "dimos_media" / "preview"
FPS = 24
WIDTH = 1920
HEIGHT = 1080

BG = (10, 14, 21)
PANEL = (22, 30, 43)
PANEL_2 = (29, 38, 54)
TEXT = (238, 242, 247)
MUTED = (164, 176, 194)
GREEN = (76, 218, 151)
BLUE = (82, 155, 255)
YELLOW = (250, 207, 82)
RED = (255, 94, 94)
LINE = (58, 72, 95)


JSON = dict[str, Any]


def font(size: int, *, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
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
    for item in candidates:
        path = Path(item)
        if path.exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


F_TITLE = font(86, bold=True)
F_H1 = font(58, bold=True)
F_H2 = font(40, bold=True)
F_BODY = font(30)
F_SMALL = font(23)
F_TINY = font(19)
F_MONO = font(22, mono=True)


def load_json(path: Path) -> JSON:
    return json.loads(path.read_text())


def canvas() -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)
    for y in range(HEIGHT):
        t = y / HEIGHT
        draw.line((0, y, WIDTH, y), fill=(int(BG[0] + 11 * t), int(BG[1] + 13 * t), int(BG[2] + 18 * t)))
    return img


def rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill=PANEL, outline=LINE, radius=20) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=2)


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, fnt, fill=TEXT, anchor=None) -> None:
    draw.text(xy, value, font=fnt, fill=fill, anchor=anchor)


def wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, fnt, chars: int, fill=MUTED, line_gap: int = 8) -> int:
    x, y = xy
    for line in wrap(value, chars):
        draw.text((x, y), line, font=fnt, fill=fill)
        y += fnt.size + line_gap
    return y


def fit(path: Path, box: tuple[int, int, int, int], *, crop: bool = False) -> Image.Image:
    src = Image.open(path).convert("RGB")
    bw, bh = box[2] - box[0], box[3] - box[1]
    sw, sh = src.size
    scale = max(bw / sw, bh / sh) if crop else min(bw / sw, bh / sh)
    resized = src.resize((max(1, int(sw * scale)), max(1, int(sh * scale))), Image.Resampling.LANCZOS)
    if crop:
        left = max(0, (resized.width - bw) // 2)
        top = max(0, (resized.height - bh) // 2)
        return resized.crop((left, top, left + bw, top + bh))
    out = Image.new("RGB", (bw, bh), (6, 9, 14))
    out.paste(resized, ((bw - resized.width) // 2, (bh - resized.height) // 2))
    return out


def image_panel(img: Image.Image, path: Path, box: tuple[int, int, int, int], *, crop: bool = False) -> None:
    draw = ImageDraw.Draw(img)
    rounded(draw, box, fill=(6, 9, 14), radius=22)
    img.paste(fit(path, box, crop=crop), (box[0], box[1]))
    draw.rounded_rectangle(box, radius=22, outline=LINE, width=2)


def paste_image_panel(img: Image.Image, content: Image.Image, box: tuple[int, int, int, int]) -> None:
    draw = ImageDraw.Draw(img)
    rounded(draw, box, fill=(6, 9, 14), radius=22)
    bw, bh = box[2] - box[0], box[3] - box[1]
    scale = min(bw / content.width, bh / content.height)
    resized = content.resize((max(1, int(content.width * scale)), max(1, int(content.height * scale))), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (bw, bh), (6, 9, 14))
    panel.paste(resized, ((bw - resized.width) // 2, (bh - resized.height) // 2))
    img.paste(panel, (box[0], box[1]))
    draw.rounded_rectangle(box, radius=22, outline=LINE, width=2)


def header(draw: ImageDraw.ImageDraw, title: str, subtitle: str) -> None:
    text(draw, (82, 56), title, F_H1)
    text(draw, (84, 124), subtitle, F_SMALL, fill=MUTED)
    draw.line((84, 166, WIDTH - 84, 166), fill=LINE, width=2)


def badge(draw: ImageDraw.ImageDraw, xy: tuple[int, int], label: str, color=BLUE) -> None:
    x, y = xy
    width = int(draw.textlength(label, font=F_SMALL)) + 34
    draw.rounded_rectangle(
        (x, y, x + width, y + 44),
        radius=22,
        fill=(max(8, color[0] // 4), max(8, color[1] // 4), max(8, color[2] // 4)),
        outline=color,
        width=2,
    )
    text(draw, (x + 17, y + 9), label, F_SMALL)


def extract_robot_video_frames() -> list[Path]:
    if VIDEO_FRAMES.exists():
        shutil.rmtree(VIDEO_FRAMES)
    VIDEO_FRAMES.mkdir(parents=True, exist_ok=True)
    if not SOURCE_ROBOT_VIDEO.is_file():
        return []
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(SOURCE_ROBOT_VIDEO),
        "-vf",
        "fps=1/4,scale=960:-1",
        "-frames:v",
        "6",
        str(VIDEO_FRAMES / "robot_%02d.jpg"),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return sorted(VIDEO_FRAMES.glob("robot_*.jpg"))


def dataset_montage() -> Image.Image:
    paths = sorted((ROOT / "hf_dataset" / "images" / "real_photo_edit" / "train").glob("*.jpg"))[:12]
    if len(paths) < 12:
        paths = sorted((ROOT / "artifacts" / "real_photo_edit_dataset" / "images").glob("*.jpg"))[:12]
    cell_w, cell_h = 350, 210
    gap = 14
    label_h = 32
    out = Image.new("RGB", (4 * cell_w + 3 * gap, 3 * (cell_h + label_h) + 2 * gap), (245, 247, 250))
    draw = ImageDraw.Draw(out)
    for idx, path in enumerate(paths[:12]):
        row = idx // 4
        col = idx % 4
        x = col * (cell_w + gap)
        y = row * (cell_h + label_h + gap)
        thumb = fit(path, (0, 0, cell_w, cell_h), crop=True)
        out.paste(thumb, (x, y))
        draw.text((x + 8, y + cell_h + 6), path.stem.replace("real_photo_edit_", "edit "), font=F_TINY, fill=(35, 43, 55))
    return out


class Builder:
    def __init__(self) -> None:
        if FRAME_DIR.exists():
            shutil.rmtree(FRAME_DIR)
        FRAME_DIR.mkdir(parents=True, exist_ok=True)
        self.index = 0

    def write(self, img: Image.Image) -> None:
        img.save(FRAME_DIR / f"frame_{self.index:05d}.jpg", quality=91, optimize=True)
        self.index += 1

    def hold(self, seconds: float, fn: Callable[[float], Image.Image]) -> None:
        count = max(1, int(seconds * FPS))
        for idx in range(count):
            t = idx / max(1, count - 1)
            self.write(fn(t))


def title_slide(_: float) -> Image.Image:
    img = canvas()
    draw = ImageDraw.Draw(img)
    text(draw, (94, 230), "WorldForge", F_H1, fill=BLUE)
    text(draw, (94, 304), "Go2 Micro World Scorer", F_TITLE)
    wrapped(
        draw,
        (100, 430),
        "A real Unitree Go2 saw the venue. We turned those robot-view frames into counterfactual decisions, trained a small scorer, and saved every decision as evidence.",
        F_BODY,
        78,
        fill=MUTED,
        line_gap=10,
    )
    badge(draw, (104, 614), "real Go2 POV", GREEN)
    badge(draw, (326, 614), "label-safe edits", YELLOW)
    badge(draw, (580, 614), "candidate scoring", BLUE)
    text(draw, (96, 890), "Not a black-box robot trick: observation + goal + candidate action -> score.", F_H2)
    return img


def robot_proof_slide(paths: list[Path]) -> Image.Image:
    img = canvas()
    draw = ImageDraw.Draw(img)
    header(draw, "Real Robot Material", "The source is the Go2 at the venue, not a synthetic room.")
    if paths:
        boxes = [(82, 230, 670, 670), (712, 230, 1300, 670), (1342, 230, 1830, 670)]
        for path, box in zip(paths[:3], boxes):
            image_panel(img, path, box, crop=True)
    else:
        fallback = ROOT / "artifacts" / "live_ciro" / "direct_camera_red_block_front.jpg"
        image_panel(img, fallback, (150, 228, 1770, 740), crop=False)
    rounded(draw, (160, 754, 1760, 922), fill=PANEL_2, radius=24)
    wrapped(
        draw,
        (202, 792),
        "The live run was messy, which is normal for hackathon robotics. The useful contribution is turning each robot-view frame into an inspectable decision trace instead of just showing motion.",
        F_BODY,
        92,
        fill=TEXT,
    )
    return img


def dimos_context_slide(_: float) -> Image.Image:
    img = canvas()
    draw = ImageDraw.Draw(img)
    header(draw, "DimOS Base Layer", "Public DimOS media shows the robot OS substrate; our work is the decision trace layer above it.")
    examples = [
        ("navigation + mapping", DIMOS_MEDIA_PREVIEW / "navigation.jpg"),
        ("agentic control + MCP", DIMOS_MEDIA_PREVIEW / "agentic_control.jpg"),
        ("spatial memory", DIMOS_MEDIA_PREVIEW / "spatial_memory.jpg"),
    ]
    boxes = [(82, 238, 626, 628), (688, 238, 1232, 628), (1294, 238, 1838, 628)]
    for (label, path), box in zip(examples, boxes):
        if path.is_file():
            image_panel(img, path, box, crop=True)
        else:
            rounded(draw, box, fill=(6, 9, 14), radius=22)
        draw.rounded_rectangle((box[0] + 22, box[1] + 22, box[0] + 372, box[1] + 72), radius=18, fill=(0, 0, 0))
        text(draw, (box[0] + 42, box[1] + 35), label, F_SMALL)

    rounded(draw, (164, 738, 1756, 922), fill=PANEL_2, radius=24)
    wrapped(
        draw,
        (206, 778),
        "DimOS gives us robot IO, skills, replay, mapping, and MCP. WorldForge-style scoring asks a narrower question before execution: given this observation, goal, and candidate actions, which action has the best expected outcome?",
        F_BODY,
        92,
        fill=TEXT,
    )
    text(draw, (206, 888), "DimOS media from dimensionalOS/dimos; used as platform context, not as our own robot footage.", F_TINY, fill=MUTED)
    return img


def observation_slide(_: float) -> Image.Image:
    img = canvas()
    draw = ImageDraw.Draw(img)
    header(draw, "Curated Go2 POV Frames", "Only signal-bearing frames are kept for the dataset and demo.")
    examples = [
        ("red target left", ROOT / "artifacts" / "live_ciro" / "direct_camera_red_block_left_annotated.jpg"),
        ("red target centered", ROOT / "artifacts" / "live_ciro" / "direct_camera_red_block_front.jpg"),
        ("unsafe marker", ROOT / "artifacts" / "live_ciro" / "direct_camera_unsafe_path_annotated.jpg"),
        ("no red visible", ROOT / "artifacts" / "live_ciro" / "direct_camera_no_red_annotated.jpg"),
    ]
    boxes = [(82, 230, 910, 520), (1010, 230, 1838, 520), (82, 624, 910, 914), (1010, 624, 1838, 914)]
    for (label, path), box in zip(examples, boxes):
        image_panel(img, path, box, crop=True)
        draw.rounded_rectangle((box[0] + 20, box[1] + 20, box[0] + 340, box[1] + 68), radius=18, fill=(0, 0, 0))
        text(draw, (box[0] + 40, box[1] + 31), label, F_SMALL)
    return img


def dataset_slide(_: float) -> Image.Image:
    img = canvas()
    draw = ImageDraw.Draw(img)
    summary = load_json(ROOT / "hf_dataset" / "dataset_summary.json")
    header(draw, "Counterfactual Dataset", "Same real Go2 plates, real cube cutouts moved to new positions with labels preserved.")
    paste_image_panel(img, dataset_montage(), (74, 214, 1260, 900))
    rounded(draw, (1306, 214, 1846, 900), fill=PANEL_2, radius=24)
    text(draw, (1340, 250), "HF-ready rows", F_H2, fill=GREEN)
    stats = [
        ("train", summary["row_counts"].get("train", 0)),
        ("validation", summary["row_counts"].get("validation", 0)),
        ("test", summary["row_counts"].get("test", 0)),
        ("real_seed", summary["row_counts"].get("real_seed", 0)),
    ]
    y = 326
    for name, value in stats:
        text(draw, (1342, y), name, F_BODY, fill=MUTED)
        text(draw, (1800, y), str(value), F_BODY, fill=TEXT, anchor="ra")
        y += 58
    wrapped(
        draw,
        (1342, 620),
        "Each row carries image, mask/bbox, observation summary, candidate actions, candidate scores, selected action, and limitations.",
        F_SMALL,
        37,
        fill=TEXT,
    )
    return img


def metrics_slide(_: float) -> Image.Image:
    img = canvas()
    draw = ImageDraw.Draw(img)
    report = load_json(ROOT / "artifacts" / "micro_world_scorer" / "eval_report.json")
    header(draw, "Micro World Scorer", "A small learned score head, trained locally. Honest boundary: not a Go2 foundation model.")
    rounded(draw, (100, 240, 1820, 872), fill=PANEL_2, radius=28)
    columns = [
        (150, 292, "Input", BLUE, "cube geometry + unsafe risk + candidate action token"),
        (705, 292, "Output", GREEN, "predicted candidate score for WorldForge-style ranking"),
        (1260, 292, "Claim", YELLOW, "micro scorer over trace labels, not full robot autonomy"),
    ]
    for x, y, label, color, body in columns:
        text(draw, (x, y), label, F_H2, fill=color)
        wrapped(draw, (x, y + 62), body, F_BODY, 27, fill=TEXT)

    score_metrics = report["score_metrics"]["test"]
    selection = report["selection_metrics"]["test"]
    baselines = report["baselines"]["test"]
    rows = [
        ("test selection accuracy", f"{selection['accuracy'] * 100:.1f}%"),
        ("test MAE", f"{score_metrics['mae']:.4f}"),
        ("test R2", f"{score_metrics['r2']:.3f}"),
        ("random baseline", f"{baselines['random_expected_accuracy'] * 100:.0f}%"),
        ("always-forward baseline", f"{baselines['always_forward_small_accuracy'] * 100:.1f}%"),
    ]
    y = 554
    for label, value in rows:
        text(draw, (180, y), label, F_BODY, fill=MUTED)
        text(draw, (830, y), value, F_BODY, fill=TEXT, anchor="ra")
        y += 58
    wrapped(draw, (1020, 572), report["claim_boundary"], F_SMALL, 54, fill=MUTED)
    return img


def audit_slide(_: float) -> Image.Image:
    img = canvas()
    draw = ImageDraw.Draw(img)
    audit = load_json(ROOT / "artifacts" / "model_audit" / "honesty_report.json")
    jepa = load_json(ROOT / "artifacts" / "micro_jepa_scorer" / "eval_report.json")
    dino = load_json(ROOT / "artifacts" / "dinov2_scorer" / "eval_report.json")
    header(draw, "Model Audit", "We tried the cooler ML path, then kept the claim honest.")
    rows = [
        (
            "geometry micro scorer",
            "trace geometry + action",
            f"{audit['main_model']['test_selection_accuracy'] * 100:.1f}%",
            f"{audit['main_model']['test_score_metrics']['r2']:.3f}",
        ),
        (
            "micro JEPA-style scorer",
            "predict latent -> score",
            f"{jepa['selection_metrics_from_predicted_latents']['test']['accuracy'] * 100:.1f}%",
            f"{jepa['score_metrics_from_predicted_latents']['test']['r2']:.3f}",
        ),
        (
            "DINOv2 hybrid scorer",
            "frozen DINOv2 + trace features",
            f"{dino['selection_metrics']['test']['accuracy'] * 100:.1f}%",
            f"{dino['score_metrics']['test']['r2']:.3f}",
        ),
        (
            "shuffled-label control",
            "destroyed labels",
            f"{audit['shuffled_label_control']['test_selection_accuracy']['mean'] * 100:.1f}%",
            f"{audit['shuffled_label_control']['test_r2']['mean']:.3f}",
        ),
    ]
    rounded(draw, (106, 238, 1814, 672), fill=PANEL_2, radius=26)
    headers = [("model", 146), ("input", 640), ("accuracy", 1200), ("R2", 1580)]
    for label, x in headers:
        text(draw, (x, 278), label, F_SMALL, fill=MUTED)
    y = 340
    for name, input_text, accuracy, r2 in rows:
        color = RED if "shuffled" in name else GREEN if "JEPA" in name else TEXT
        text(draw, (146, y), name, F_BODY, fill=color)
        text(draw, (640, y), input_text, F_BODY, fill=MUTED)
        text(draw, (1200, y), accuracy, F_BODY, fill=color)
        text(draw, (1580, y), r2, F_BODY, fill=color)
        y += 78
    rounded(draw, (188, 742, 1732, 918), fill=PANEL, radius=24)
    wrapped(
        draw,
        (230, 784),
        "Result: the JEPA-style scorer is architecturally cleaner, but the strongest honest claim is still the trace interface. DINOv2 did not materially improve geometry-derived labels; the audit makes that visible instead of hiding it.",
        F_BODY,
        92,
        fill=TEXT,
    )
    return img


def demo_slide(_: float) -> Image.Image:
    img = canvas()
    draw = ImageDraw.Draw(img)
    header(draw, "One-Command Scorer Demo", "A real frame goes through the learned scorer and writes trace artifacts.")
    image_panel(img, ROOT / "artifacts" / "micro_world_demo" / "latest" / "annotated_image.jpg", (72, 220, 1268, 910), crop=False)
    scores = load_json(ROOT / "artifacts" / "micro_world_demo" / "latest" / "candidate_scores.json")
    rows = scores["scores"]
    rounded(draw, (1314, 220, 1846, 910), fill=PANEL_2, radius=24)
    text(draw, (1344, 256), "candidate_scores.json", F_H2, fill=TEXT)
    y = 330
    max_score = max(row["score"] for row in rows) or 1.0
    for row in rows:
        color = GREEN if row["selected"] else BLUE
        label = row["candidate_id"].replace("_", " ")
        text(draw, (1348, y), label, F_SMALL, fill=TEXT if row["selected"] else MUTED)
        text(draw, (1800, y), f"{row['score']:.3f}", F_SMALL, fill=TEXT, anchor="ra")
        draw.rounded_rectangle((1348, y + 32, 1800, y + 52), radius=10, fill=(8, 12, 18))
        draw.rounded_rectangle((1348, y + 32, 1348 + int(452 * row["score"] / max_score), y + 52), radius=10, fill=color)
        y += 92
    text(draw, (1348, 800), f"selected: {scores['selected_candidate_id']}", F_H2, fill=GREEN)
    return img


def evidence_slide(_: float) -> Image.Image:
    img = canvas()
    draw = ImageDraw.Draw(img)
    header(draw, "Evidence Trail", "The same artifact shape can later swap in a stronger world model.")
    cards = [
        ("score_info.json", "what the scorer saw: task, observation, candidate actions"),
        ("candidate_scores.json", "ranked actions, scores, reasons, selected candidate"),
        ("selected_action.json", "safe host-execution handoff; WorldForge does not directly drive the robot"),
        ("outcome_after_execution.json", "post-action outcome fields for future training labels"),
    ]
    for idx, (name, desc) in enumerate(cards):
        x = 116 + (idx % 2) * 860
        y = 244 + (idx // 2) * 278
        rounded(draw, (x, y, x + 790, y + 220), fill=PANEL_2, radius=24)
        text(draw, (x + 34, y + 32), name, F_H2, fill=GREEN if idx == 0 else TEXT)
        wrapped(draw, (x + 36, y + 98), desc, F_BODY, 40, fill=MUTED)
    snippet = '{ "observation + goal + candidate_action": "score", "selected": "turn_left" }'
    text(draw, (120, 918), snippet, F_MONO, fill=(207, 226, 255))
    return img


def package_slide(_: float) -> Image.Image:
    img = canvas()
    draw = ImageDraw.Draw(img)
    header(draw, "Submission Package", "GitHub-first, Hugging Face-ready, and explicit about limitations.")
    items = [
        ("Dataset", "real Go2 seed frames + label-safe real-photo-edit counterfactuals"),
        ("Model", "small micro world scorer with eval report and sample predictions"),
        ("Demo", "one command creates annotated image, JSON evidence, and trace MP4"),
        ("Video", "final 75-90s story for judges"),
    ]
    y = 250
    for label, body in items:
        rounded(draw, (124, y, 1796, y + 146), fill=PANEL_2, radius=24)
        text(draw, (166, y + 36), label, F_H2, fill=GREEN)
        text(draw, (430, y + 44), body, F_BODY, fill=TEXT)
        y += 176
    return img


def closing_slide(_: float) -> Image.Image:
    img = canvas()
    draw = ImageDraw.Draw(img)
    text(draw, (96, 248), "The Point", F_TITLE)
    wrapped(
        draw,
        (104, 382),
        "Robots should not just emit actions. They should expose the options they considered, the score behind each option, and the evidence needed to improve the scorer later.",
        F_H2,
        72,
        fill=TEXT,
        line_gap=12,
    )
    text(draw, (104, 730), "github.com/omarespejel/worldforge-go2-trace-judge", F_H2, fill=BLUE)
    text(draw, (104, 806), "WorldForge-style decision traces for embodied AI", F_BODY, fill=MUTED)
    return img


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    robot_frames = extract_robot_video_frames()
    builder = Builder()
    builder.hold(3.5, title_slide)
    builder.hold(7.0, lambda _t: robot_proof_slide(robot_frames))
    builder.hold(6.5, dimos_context_slide)
    builder.hold(8.0, observation_slide)
    builder.hold(11.0, dataset_slide)
    builder.hold(10.0, metrics_slide)
    builder.hold(10.0, audit_slide)
    builder.hold(12.0, demo_slide)
    builder.hold(10.0, evidence_slide)
    builder.hold(5.0, package_slide)
    builder.hold(5.0, closing_slide)
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
