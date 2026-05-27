from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from threading import Event
from typing import Any

from PIL import Image

from dimos.robot.unitree.connection import UnitreeWebRTCConnection as DimosWebRTCConnection
from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD
from unitree_webrtc_connect.webrtc_driver import (
    UnitreeWebRTCConnection as RawWebRTCConnection,
    WebRTCConnectionMethod,
)

from go2_find_colored_target import (
    _build_candidates,
    _detect_scene,
    _score,
    _write_json,
    _write_step_artifacts,
)


ROBOT_IP = "192.168.12.1"
RUN_ID = "live-avoidance-final"
OUT_DIR = Path(".worldforge/color-target-demo") / RUN_ID
ARTIFACT_DIR = Path("artifacts")
TARGET = "red"
UNSAFE = {"green"}
MAX_STEPS = 5
SUCCESS_AREA = 0.0020
SUCCESS_CENTER_X = 0.20


def log(message: str) -> None:
    print(message, flush=True)


def capture_frame(step_index: int) -> Path:
    out = ARTIFACT_DIR / f"{RUN_ID}_step_{step_index:02d}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    seen = Event()
    err: dict[str, str] = {}
    conn = DimosWebRTCConnection(ROBOT_IP)
    sub = None
    try:
        def on_frame(frame: Any) -> None:
            if seen.is_set():
                return
            try:
                arr = frame.to_ndarray(format="rgb24")
                Image.fromarray(arr).save(out, quality=92)
            except Exception as exc:  # noqa: BLE001
                err["message"] = repr(exc)
            finally:
                seen.set()

        sub = conn.raw_video_stream().subscribe(on_frame)
        deadline = time.monotonic() + 20
        while not seen.is_set() and time.monotonic() < deadline:
            time.sleep(0.1)
    finally:
        if sub is not None:
            sub.dispose()
        conn.stop()

    if err:
        raise RuntimeError(err["message"])
    if not out.exists():
        raise RuntimeError("camera capture timed out")
    return out


async def _with_raw_conn(fn):
    last_error: BaseException | None = None
    for attempt in range(1, 4):
        conn = RawWebRTCConnection(WebRTCConnectionMethod.LocalAP, ip=ROBOT_IP)
        try:
            await asyncio.wait_for(conn.connect(), timeout=25)
            return await fn(conn)
        except BaseException as exc:  # Unitree WebRTC can call sys.exit on SDP failures.
            last_error = exc
            log(f"WEBRTC_RETRY attempt={attempt} error={type(exc).__name__}: {exc}")
            await asyncio.sleep(2.0 * attempt)
        finally:
            try:
                await asyncio.wait_for(conn.disconnect(), timeout=5)
            except BaseException:
                pass
    assert last_error is not None
    raise last_error


async def _joystick(conn, lx: float = 0.0, ly: float = 0.0, rx: float = 0.0, ry: float = 0.0) -> None:
    conn.datachannel.pub_sub.publish_without_callback(
        RTC_TOPIC["WIRELESS_CONTROLLER"],
        data={"lx": lx, "ly": ly, "rx": rx, "ry": ry},
    )


def execute_selected(candidate: dict[str, Any]) -> str:
    async def run(conn):
        await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["BalanceStand"]}
        )
        await asyncio.sleep(0.25)

        params = candidate["params"]
        forward = max(-0.12, min(0.12, float(params.get("forward", 0.0))))
        left = max(-0.12, min(0.12, float(params.get("left", 0.0))))
        turn = max(-0.18, min(0.18, float(params.get("turn", 0.0))))
        duration = max(0.20, min(0.50, float(params.get("duration", 0.40))))

        start = time.monotonic()
        while time.monotonic() - start < duration:
            await _joystick(conn, lx=left, ly=forward, rx=turn)
            await asyncio.sleep(0.03)
        for _ in range(35):
            await _joystick(conn, 0, 0, 0, 0)
            await asyncio.sleep(0.03)
        return {
            "forward": forward,
            "left": left,
            "turn": turn,
            "duration": duration,
        }

    if candidate["id"] == "stop_capture":
        return "stop_capture: no movement"
    result = asyncio.run(_with_raw_conn(run))
    return f"executed {candidate['id']}: {json.dumps(result, sort_keys=True)}"


def sit_down() -> str:
    async def run(conn):
        response = await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["Sit"]}
        )
        return response

    response = asyncio.run(_with_raw_conn(run))
    return f"Sit response: {json.dumps(response, default=str)[:500]}"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"RUN_ID={RUN_ID}")
    log("GOAL=find red cube, avoid green cube, stop near target, then sit")
    log(f"OUT_DIR={OUT_DIR.resolve()}")

    summary: list[dict[str, Any]] = []
    success = False
    for step_index in range(1, MAX_STEPS + 1):
        log(f"STEP {step_index}: capture")
        frame = capture_frame(step_index)
        scene = _detect_scene(frame, TARGET, UNSAFE)
        candidates = _build_candidates(scene)
        scores = [_score(candidate["features"]) for candidate in candidates]
        best = max(range(len(scores)), key=scores.__getitem__)
        selected = candidates[best]
        target = scene["target"]
        close_enough = (
            bool(target["found"])
            and abs(float(target["center_x"])) <= SUCCESS_CENTER_X
            and float(target["area_ratio"]) >= SUCCESS_AREA
        )
        if close_enough:
            selected = next(candidate for candidate in candidates if candidate["id"] == "stop_capture")
            best = [candidate["id"] for candidate in candidates].index("stop_capture")

        log(
            "DECISION "
            f"target_found={target['found']} x={target['center_x']} area={target['area_ratio']} "
            f"unsafe={scene['unsafe_risk']} selected={selected['id']} "
            + " ".join(f"{candidate['id']}={score}" for candidate, score in zip(candidates, scores))
        )

        if selected["id"] == "stop_capture":
            execution_result = "success: red cube centered/close enough; stopping before sit"
            success = True
        else:
            time.sleep(2.0)
            execution_result = execute_selected(selected)
            log(f"EXECUTION {execution_result}")

        step_dir = OUT_DIR / f"step_{step_index:02d}"
        _write_step_artifacts(
            step_dir=step_dir,
            run_id=RUN_ID,
            step_index=step_index,
            frame_path=frame,
            target=TARGET,
            scene=scene,
            candidates=candidates,
            selected=selected,
            scores=scores,
            executed=selected["id"] != "stop_capture",
            execution_result=execution_result,
        )
        summary.append(
            {
                "step_index": step_index,
                "frame": str(frame.resolve()),
                "selected": selected["id"],
                "target": target,
                "unsafe": scene["unsafe_risk"],
                "execution_result": execution_result,
                "step_dir": str(step_dir.resolve()),
            }
        )
        if success:
            break
        time.sleep(0.4)

    log("FINAL: sit down")
    try:
        sit_result = sit_down()
    except Exception as exc:  # noqa: BLE001
        sit_result = f"sit failed: {type(exc).__name__}: {exc}"
    log(sit_result)
    _write_json(
        OUT_DIR / "summary.json",
        {
            "schema_version": 1,
            "run_id": RUN_ID,
            "success": success,
            "target": TARGET,
            "unsafe_colors": sorted(UNSAFE),
            "steps": summary,
            "final_action": sit_result,
        },
    )
    log(f"SUMMARY={OUT_DIR / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
