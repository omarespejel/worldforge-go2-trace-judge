# 90-Second Judges Script

## Opening

This is WorldForge Go2 Trace Judge: inspectable autonomy for a Unitree Go2.

The important part is not only that the robot moves. The important part is that every
move has evidence.

## Demo Beat

The Go2 camera sees the scene. The replay task is to find the target color. In a
calibrated live block setup, the same loop can also avoid unsafe colored markers.

The host runtime proposes four bounded actions:

```text
turn_left
turn_right
forward_small
stop_capture
```

For each frame, the trace judge scores each possible next move. It logs:

```text
observation_summary
candidate actions
score_info
candidate_scores
selected_action
outcome_after_execution
```

Then the host runtime executes only the selected safe action. In replay mode, no robot
command is executed; the same evidence files are still produced from real Go2 camera
video.

## Core Claim

WorldForge is not the motor controller. DimOS or the host robot runtime owns the
hardware and safety boundary.

WorldForge is the decision evidence layer. It makes the autonomy inspectable,
replayable, and swappable.

Today the scorer is transparent. Tomorrow the same trace dataset can train a learned
world-model scorer.

## One-Sentence Close

We are turning robot behavior from a black box into a replayable decision trace.

## If Asked About Training

We did not train a Go2 foundation world model during the hackathon. That would be the
wrong claim.

What we built is the data contract a learned scorer needs:

```text
observation + goal + candidate action -> score + measured outcome
```

The included tiny ranker is a smoke test that proves the traces can be consumed by a
model. It distills the transparent scorer today; it can be replaced with outcome labels
after more live robot runs.
