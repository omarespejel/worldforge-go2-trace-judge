#!/usr/bin/env python3
"""Bounded no-robot DimOS smoke checks.

The script is intentionally conservative. It runs CLI/list/config checks by
default. Replay and simulation checks are opt-in, foreground-only, and killed as
a process group at timeout so they do not leave orphaned workers.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIMOS_REPO = Path(
    os.environ.get(
        "DIMOS_REPO",
        "/Users/espejelomar/StarkNet/zk-ai/_pr_work/dimos-worldforge-go2-trace-judge",
    )
)
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "dimos_simulation_smoke"


def default_dimos_bin(dimos_repo: Path) -> Path:
    local = dimos_repo / ".venv" / "bin" / "dimos"
    if local.exists():
        return local
    return Path("dimos")


def run_bounded(
    cmd: list[str],
    cwd: Path,
    timeout_s: float,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    start = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
        env=env,
    )
    timed_out = False
    output = ""
    try:
        output, _ = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            output, _ = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            output, _ = proc.communicate(timeout=5)

    elapsed_s = time.monotonic() - start
    return {
        "cmd": cmd,
        "cwd": str(cwd),
        "returncode": proc.returncode,
        "timed_out": timed_out,
        "elapsed_s": round(elapsed_s, 3),
        "stdout": output,
        "stdout_tail": output[-4000:],
    }


def command_for(mode: str, dimos_bin: Path) -> list[str]:
    base = [str(dimos_bin)]
    if mode == "help":
        return [*base, "--help"]
    if mode == "list":
        return [*base, "list"]
    if mode == "show-config":
        return [*base, "show-config"]
    if mode == "replay-go2":
        return [
            *base,
            "--replay",
            "--viewer",
            "none",
            "--rerun-open",
            "none",
            "--memory-limit",
            "1GB",
            "run",
            "unitree-go2",
        ]
    if mode == "simulation-go2":
        return [
            *base,
            "--simulation",
            "mujoco",
            "--viewer",
            "none",
            "--rerun-open",
            "none",
            "--memory-limit",
            "1GB",
            "run",
            "unitree-go2",
        ]
    if mode == "agentic-simulation-go2":
        return [
            *base,
            "--simulation",
            "mujoco",
            "--viewer",
            "none",
            "--rerun-open",
            "none",
            "--memory-limit",
            "1GB",
            "run",
            "unitree-go2-agentic",
        ]
    raise ValueError(f"unknown mode: {mode}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimos-repo", type=Path, default=DEFAULT_DIMOS_REPO)
    parser.add_argument("--dimos-bin", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--mode",
        action="append",
        choices=[
            "help",
            "list",
            "show-config",
            "replay-go2",
            "simulation-go2",
            "agentic-simulation-go2",
        ],
        help="Mode to run. May be repeated. Defaults to help/list/show-config.",
    )
    parser.add_argument("--timeout-s", type=float, default=90.0)
    parser.add_argument(
        "--bypass-system-config",
        action="store_true",
        help=(
            "Set PYTEST_VERSION so DimOS skips host system configurators. "
            "Use only for structural no-sudo diagnostics, not real runs."
        ),
    )
    args = parser.parse_args()

    dimos_repo = args.dimos_repo.expanduser().resolve()
    dimos_bin = (args.dimos_bin or default_dimos_bin(dimos_repo)).expanduser()
    if not dimos_bin.is_absolute():
        dimos_bin = dimos_bin.resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    modes = args.mode or ["help", "list", "show-config"]
    results = []
    env = os.environ.copy()
    if args.bypass_system_config:
        env["PYTEST_VERSION"] = "dimos-smoke"
    for mode in modes:
        cmd = command_for(mode, dimos_bin)
        result = run_bounded(cmd, dimos_repo, args.timeout_s, env=env)
        result["mode"] = mode
        result["bypass_system_config"] = args.bypass_system_config
        results.append(result)
        status = "timeout" if result["timed_out"] else f"rc={result['returncode']}"
        print(f"{mode}: {status} in {result['elapsed_s']}s")

    summary = {
        "schema_version": 1,
        "dimos_repo": str(dimos_repo),
        "dimos_bin": str(dimos_bin),
        "timeout_s": args.timeout_s,
        "bypass_system_config": args.bypass_system_config,
        "results": results,
    }
    (output_dir / "smoke_report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    concise = [
        {
            "mode": r["mode"],
            "returncode": r["returncode"],
            "timed_out": r["timed_out"],
            "elapsed_s": r["elapsed_s"],
            "tail": r["stdout_tail"],
        }
        for r in results
    ]
    (output_dir / "smoke_summary.json").write_text(json.dumps(concise, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_dir / 'smoke_report.json'}")
    print(f"Wrote {output_dir / 'smoke_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
