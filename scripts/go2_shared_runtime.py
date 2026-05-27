from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import threading
import time
from typing import Any

from PIL import Image as PILImage
from langchain_core.messages import AIMessage
from pydantic import Field

from dimos.agents.annotation import skill
from dimos.agents.mcp.mcp_server import McpServer
from dimos.core.coordination.blueprints import autoconnect
from dimos.core.coordination.module_coordinator import ModuleCoordinator
from dimos.core.core import rpc
from dimos.core.global_config import global_config
from dimos.core.module import Module, ModuleConfig
from dimos.core.transport import pLCMTransport
from dimos.robot.unitree.connection import UnitreeWebRTCConnection as DimosWebRTCConnection
from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD
from unitree_webrtc_connect.webrtc_driver import (
    UnitreeWebRTCConnection as RawWebRTCConnection,
    WebRTCConnectionMethod,
)


SAFE_SPORT_COMMANDS = {
    "BalanceStand",
    "StandUp",
    "StandDown",
    "RecoveryStand",
    "Sit",
    "RiseSit",
    "Hello",
    "Stretch",
}


class Go2DirectSkillsConfig(ModuleConfig):
    robot_ip: str = Field(default_factory=lambda m: m["g"].robot_ip or "192.168.12.1")
    artifact_dir: str = "artifacts"


class Go2DirectSkills(Module):
    """Small hackathon-safe Go2 skill surface.

    This intentionally avoids the full mapping/navigation stack so the robot can be
    shared by humancli and MCP on a Mac without CUDA or long-running mapper startup.
    """

    config: Go2DirectSkillsConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lock = threading.Lock()

    async def _with_raw_conn(self, fn):
        method = (
            WebRTCConnectionMethod.LocalAP
            if self.config.robot_ip == "192.168.12.1"
            else WebRTCConnectionMethod.LocalSTA
        )
        conn = RawWebRTCConnection(method, ip=self.config.robot_ip)
        await asyncio.wait_for(conn.connect(), timeout=25)
        result = "Command did not complete."
        try:
            return await fn(conn)
        finally:
            await asyncio.wait_for(conn.disconnect(), timeout=5)

    @skill
    def execute_sport_command(self, command_name: str) -> str:
        """Execute a safe built-in Unitree Go2 posture/greeting command."""
        if command_name not in SAFE_SPORT_COMMANDS:
            allowed = ", ".join(sorted(SAFE_SPORT_COMMANDS))
            return f"Rejected '{command_name}'. Allowed commands: {allowed}"

        async def run(conn):
            api_id = SPORT_CMD[command_name]
            response = await asyncio.wait_for(
                conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["SPORT_MOD"], {"api_id": api_id}
                ),
                timeout=8,
            )
            return response

        with self._lock:
            response = asyncio.run(self._with_raw_conn(run))
        return f"{command_name} response: {json.dumps(response, default=str)[:800]}"

    @skill
    def move_joystick(
        self,
        forward: float = 0.0,
        left: float = 0.0,
        turn: float = 0.0,
        duration: float = 0.5,
    ) -> str:
        """Move briefly using bounded joystick values, then send repeated stop packets.

        Args:
            forward: Forward/backward joystick value in [-0.5, 0.5].
            left: Left/right joystick value in [-0.5, 0.5].
            turn: In-place yaw joystick value in [-0.5, 0.5].
            duration: Seconds to command motion, capped at 5 seconds.
        """
        forward = max(-0.5, min(0.5, float(forward)))
        left = max(-0.5, min(0.5, float(left)))
        turn = max(-0.5, min(0.5, float(turn)))
        duration = max(0.05, min(5.0, float(duration)))

        async def joystick(conn, lx=0.0, ly=0.0, rx=0.0, ry=0.0):
            conn.datachannel.pub_sub.publish_without_callback(
                RTC_TOPIC["WIRELESS_CONTROLLER"],
                data={"lx": lx, "ly": ly, "rx": rx, "ry": ry},
            )

        async def run(conn):
            await conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["BalanceStand"]}
            )
            await asyncio.sleep(0.3)
            start = time.monotonic()
            while time.monotonic() - start < duration:
                await joystick(conn, lx=left, ly=forward, rx=turn, ry=0.0)
                await asyncio.sleep(0.03)
            for _ in range(30):
                await joystick(conn, 0.0, 0.0, 0.0, 0.0)
                await asyncio.sleep(0.035)

        with self._lock:
            asyncio.run(self._with_raw_conn(run))
        return (
            "move_joystick complete: "
            f"forward={forward}, left={left}, turn={turn}, duration={duration}"
        )

    @skill
    def capture_camera_frame(self, filename: str = "live_robot_frame.jpg") -> str:
        """Capture one live Go2 camera frame and save it under the artifact directory."""
        safe_name = Path(filename).name
        artifact_dir = Path(self.config.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        out = artifact_dir / safe_name

        seen = threading.Event()
        error: dict[str, str] = {}
        subscription = None

        with self._lock:
            conn = DimosWebRTCConnection(self.config.robot_ip)
            try:
                def on_frame(frame):
                    if seen.is_set():
                        return
                    try:
                        arr = frame.to_ndarray(format="rgb24")
                        PILImage.fromarray(arr).save(out, quality=92)
                    except Exception as exc:  # noqa: BLE001
                        error["message"] = repr(exc)
                    finally:
                        seen.set()

                subscription = conn.raw_video_stream().subscribe(on_frame)
                deadline = time.monotonic() + 15
                while not seen.is_set() and time.monotonic() < deadline:
                    time.sleep(0.1)
            finally:
                if subscription is not None:
                    subscription.dispose()
                conn.stop()

        if error:
            return f"camera capture failed: {error['message']}"
        if not out.exists():
            return "camera capture timed out before a frame arrived"
        return f"camera frame saved: {out.resolve()} ({out.stat().st_size} bytes)"


class HumanCommandRouterConfig(ModuleConfig):
    robot_ip: str = Field(default_factory=lambda m: m["g"].robot_ip or "192.168.12.1")
    artifact_dir: str = "artifacts"


class HumanCommandRouter(Module):
    """Small offline command router for DimOS humancli.

    The stock DimOS `McpClient` creates a LangChain/OpenAI agent during startup.
    That is useful when API credentials are present, but brittle on venue Wi-Fi.
    This router keeps the demo local: humancli publishes text on `/human_input`,
    and we answer on `/agent` while executing a cautious fixed command set.
    """

    config: HumanCommandRouterConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lock = threading.Lock()
        self._human_transport: pLCMTransport[str] | None = None
        self._agent_transport: pLCMTransport[AIMessage] | None = None
        self._agent_idle_transport: pLCMTransport[bool] | None = None
        self._unsubscribe = None

    async def _with_raw_conn(self, fn):
        method = (
            WebRTCConnectionMethod.LocalAP
            if self.config.robot_ip == "192.168.12.1"
            else WebRTCConnectionMethod.LocalSTA
        )
        conn = RawWebRTCConnection(method, ip=self.config.robot_ip)
        await asyncio.wait_for(conn.connect(), timeout=25)
        try:
            return await fn(conn)
        finally:
            await asyncio.wait_for(conn.disconnect(), timeout=5)

    def _publish(self, message: str) -> None:
        if self._agent_transport is not None:
            self._agent_transport.publish(AIMessage(content=message))

    def _set_idle(self, idle: bool) -> None:
        if self._agent_idle_transport is not None:
            self._agent_idle_transport.publish(idle)

    def _sport(self, command_name: str) -> str:
        if command_name not in SAFE_SPORT_COMMANDS:
            allowed = ", ".join(sorted(SAFE_SPORT_COMMANDS))
            return f"Rejected '{command_name}'. Allowed commands: {allowed}"

        async def run(conn):
            return await asyncio.wait_for(
                conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD[command_name]}
                ),
                timeout=8,
            )

        with self._lock:
            response = asyncio.run(self._with_raw_conn(run))
        return f"{command_name} complete: {json.dumps(response, default=str)[:500]}"

    def _move(self, forward: float = 0.0, left: float = 0.0, turn: float = 0.0) -> str:
        forward = max(-0.25, min(0.25, float(forward)))
        left = max(-0.25, min(0.25, float(left)))
        turn = max(-0.25, min(0.25, float(turn)))
        duration = 0.5

        async def joystick(conn, lx=0.0, ly=0.0, rx=0.0, ry=0.0):
            conn.datachannel.pub_sub.publish_without_callback(
                RTC_TOPIC["WIRELESS_CONTROLLER"],
                data={"lx": lx, "ly": ly, "rx": rx, "ry": ry},
            )

        async def run(conn):
            await conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["BalanceStand"]}
            )
            await asyncio.sleep(0.25)
            start = time.monotonic()
            while time.monotonic() - start < duration:
                await joystick(conn, lx=left, ly=forward, rx=turn, ry=0.0)
                await asyncio.sleep(0.03)
            for _ in range(30):
                await joystick(conn, 0.0, 0.0, 0.0, 0.0)
                await asyncio.sleep(0.035)

        with self._lock:
            asyncio.run(self._with_raw_conn(run))
        return f"brief move complete: forward={forward}, left={left}, turn={turn}"

    def _capture(self, filename: str = "humancli_frame.jpg") -> str:
        safe_name = Path(filename).name
        artifact_dir = Path(self.config.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        out = artifact_dir / safe_name

        seen = threading.Event()
        error: dict[str, str] = {}
        subscription = None

        with self._lock:
            conn = DimosWebRTCConnection(self.config.robot_ip)
            try:
                def on_frame(frame):
                    if seen.is_set():
                        return
                    try:
                        arr = frame.to_ndarray(format="rgb24")
                        PILImage.fromarray(arr).save(out, quality=92)
                    except Exception as exc:  # noqa: BLE001
                        error["message"] = repr(exc)
                    finally:
                        seen.set()

                subscription = conn.raw_video_stream().subscribe(on_frame)
                deadline = time.monotonic() + 15
                while not seen.is_set() and time.monotonic() < deadline:
                    time.sleep(0.1)
            finally:
                if subscription is not None:
                    subscription.dispose()
                conn.stop()

        if error:
            return f"camera capture failed: {error['message']}"
        if not out.exists():
            return "camera capture timed out before a frame arrived"
        return f"camera frame saved: {out.resolve()} ({out.stat().st_size} bytes)"

    def _handle_message(self, message: str) -> None:
        text = message.strip().lower()
        if not text:
            return

        self._set_idle(False)
        self._publish(f"Received: {message}")
        try:
            if text in {"help", "commands"} or "help" in text:
                result = (
                    "Available local commands: stand, balance, sit, rise sit, "
                    "hello, stretch, forward, back, sidestep left/right, "
                    "turn left/right, capture frame, stop."
                )
            elif "capture" in text or "camera" in text or "photo" in text or "frame" in text:
                result = self._capture()
            elif "rise" in text and "sit" in text:
                result = self._sport("RiseSit")
            elif "sit" in text:
                result = self._sport("Sit")
            elif "balance" in text:
                result = self._sport("BalanceStand")
            elif "recover" in text:
                result = self._sport("RecoveryStand")
            elif "stand down" in text or "lie" in text:
                result = self._sport("StandDown")
            elif "stand" in text:
                result = self._sport("StandUp")
            elif "hello" in text:
                result = self._sport("Hello")
            elif "stretch" in text:
                result = self._sport("Stretch")
            elif "stop" in text:
                result = self._move()
            elif "forward" in text:
                result = self._move(forward=0.12)
            elif "back" in text or "backward" in text:
                result = self._move(forward=-0.12)
            elif "sidestep left" in text or "step left" in text:
                result = self._move(left=0.12)
            elif "sidestep right" in text or "step right" in text:
                result = self._move(left=-0.12)
            elif "turn left" in text or "rotate left" in text:
                result = self._move(turn=-0.18)
            elif "turn right" in text or "rotate right" in text:
                result = self._move(turn=0.18)
            else:
                result = (
                    "I can run only the local safety-bounded command set. "
                    "Type 'help' for commands."
                )
        except BaseException as exc:  # noqa: BLE001
            result = f"Command failed: {type(exc).__name__}: {exc}"
        finally:
            self._publish(result)
            self._set_idle(True)

    @rpc
    def start(self) -> None:
        self._human_transport = pLCMTransport("/human_input")
        self._agent_transport = pLCMTransport("/agent")
        self._agent_idle_transport = pLCMTransport("/agent_idle")
        self._unsubscribe = self._human_transport.subscribe(self._handle_message)
        self._set_idle(True)
        self._publish(
            "Go2 local router is online. Type 'help' for safe commands."
        )

    @rpc
    def stop(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        for transport in (
            self._human_transport,
            self._agent_transport,
            self._agent_idle_transport,
        ):
            if transport is not None:
                transport.stop()


def build_blueprint():
    robot_ip = os.environ.get("ROBOT_IP", "192.168.12.1")
    global_config.update(robot_ip=robot_ip, viewer="none")
    return autoconnect(
        McpServer.blueprint(),
        Go2DirectSkills.blueprint(robot_ip=robot_ip),
        HumanCommandRouter.blueprint(robot_ip=robot_ip),
    )


if __name__ == "__main__":
    coordinator = ModuleCoordinator.build(build_blueprint())
    coordinator.start_rpyc_service()
    coordinator.loop()
