# 90-Second Judges Script

## Opening

This is WorldForge Go2 Trace Judge: an inspectable decision layer for a Unitree Go2.

The point is not just that a robot moved. The point is that every possible next
move becomes visible, scored, and replayable.

## Demo Beat

We collected real Go2 robot-view frames at the venue. Then we generated label-safe
counterfactual scenes by moving real cube cutouts inside those real Go2 camera
frames.

For each scene, the robot has four bounded candidate actions:

```text
turn_left
turn_right
forward_small
stop_capture
```

The scorer receives:

```text
observation + goal + candidate action
```

and returns a candidate score. The demo writes:

```text
score_info.json
candidate_scores.json
selected_action.json
outcome_after_execution.json
```

So the robot decision is not a black box. You can inspect why `turn_left` won and
why `forward_small` was rejected when an unsafe marker was in the path.

## Core Claim

WorldForge is not the motor controller. The host runtime owns robot execution and
safety.

WorldForge-style traces provide the scoring/evidence boundary:

```text
what did the robot see?
what could it do?
which action leads to the better future?
what evidence was saved?
```

## Model Claim

We did not train a Go2 foundation world model during the hackathon.

What we trained is a small micro world scorer: a learned score head over real Go2
frames and counterfactual decision traces.

That is the honest useful step. It shows how robot decisions can move from hidden
policy output to inspectable candidate scoring.

## Close

We are turning robot behavior from a black box into a replayable decision trace:

```text
observation + goal + candidate action -> score -> selected action -> evidence
```
