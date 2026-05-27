# Trace Schema

This package uses a WorldForge-shaped trace contract without making WorldForge own
physical robot control.

## Directory Shape

Each decision step writes:

```text
step_XX/
  camera_frame.jpg
  observation_summary.json
  venue_input.json
  score_info.json
  candidate_scores.json
  selected_action.json
  outcome_after_execution.json
  run_manifest.json
  report.md
```

## Meaning

`observation_summary.json`

What the host runtime knows about the world. For this demo, it contains target color
confidence, target bearing, target area, unsafe color risk, and coarse navigation state.

`venue_input.json`

The scorer input:

```text
task + observation_summary + candidate actions
```

`score_info.json`

The same payload wrapped as a provider-style `score` request. This is the key
world-model boundary: it shows what a transparent scorer or future learned scorer saw.

`candidate_scores.json`

One row per candidate action, with score, features, and rejection/selection reason.
Higher is better in this package.

`selected_action.json`

The selected action and bounded host execution parameters. It explicitly says
`worldforge_executes_robot: false`.

`outcome_after_execution.json`

What happened after selection. In replay mode, this records that no robot command was
executed. In live mode, it records the host execution result and post-step signals.

## Training Dataset

Run:

```bash
make dataset
```

This flattens all `step_XX` artifacts into:

```text
dataset/go2_trace_candidates.jsonl
```

Each line is:

```text
observation + goal + candidate + transparent_score_label + selected + outcome
```

For the hackathon, labels come from the transparent scorer. Later, the same rows can be
relabelled from measured outcomes such as progress, target centering, unsafe risk
reduction, and manual intervention.
