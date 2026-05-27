# Submission Notes

## Project Name

WorldForge Go2 Trace Judge

## One-Liner

Inspectable autonomy for a Unitree Go2: every move is scored, selected, and replayed with evidence.

## Problem

Most robot demos show the robot moving but hide the decision process. That makes it hard to debug,
trust, or improve autonomy.

## Solution

Use DimOS/host runtime for robot control and WorldForge-style scoring artifacts for decision evidence:

```text
observation_summary
candidate actions
score_info
candidate_scores
selected_action
outcome_after_execution
```

## Demo

The current fallback demo uses real Go2 camera video in target-only mode. Unsafe color
markers should be enabled only after live calibration with the actual blocks:

```text
artifacts/replay_run/worldforge_trace_replay.mp4
artifacts/replay_run/report.html
```

The live demo uses the same scoring loop but executes bounded Go2 moves through MCP.

## Dataset Artifact

The package also exports a candidate-level dataset:

```text
dataset/go2_trace_candidates.jsonl
dataset/go2_trace_dataset_summary.json
```

Each row has:

```text
observation + goal + candidate action + score label + selected flag + outcome
```

This is the useful research artifact: it is the shape needed to replace the transparent
scorer with a learned scorer once more real robot outcomes are collected.

## Tiny Ranker Smoke Test

The included ranker is a sanity check, not a new foundation model:

```text
artifacts/ranker_smoke/model.json
```

It distills the transparent scorer from trace artifacts. Later, the same training entry
point can use measured outcome labels.

## Why It Matters

This separates:

- policy/candidate generation: what could the robot do?
- scoring/world model: which candidate leads to the best future?
- host execution: how does the robot safely perform the selected action?
- evidence: what happened and why?

## What Is Experimental

- The current scorer is transparent and heuristic.
- The live bridge is host-owned and not an upstream WorldForge dependency.
- The learned model path depends on collecting more candidate/outcome traces.

## Future Work

- Train a small learned ranker from collected traces.
- Add Rerun visualization of candidate paths and selected actions.
- Extend the same trace contract to SO-101 / LeRobot manipulation.
