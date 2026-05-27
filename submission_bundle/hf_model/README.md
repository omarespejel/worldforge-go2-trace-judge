---
license: mit
tags:
- robotics
- unitree-go2
- world-models
- decision-traces
- candidate-ranking
pipeline_tag: tabular-regression
---

# Go2 Cube Micro World Scorer

This is a small candidate-action scorer for the WorldForge Go2 Trace Judge
hackathon artifact.

It scores:

```text
observation summary + goal + candidate action -> candidate score
```

The model is intentionally small and inspectable: a pure-NumPy ridge-regression
head trained on geometry/risk features from curated real Go2 camera frames and
label-safe real-photo-edit counterfactual traces.

## Claim Boundary

This is **not** a Go2 foundation world model, not a trained V-JEPA model, and not
a validated autonomous navigation policy.

It is a micro world scorer: a learned score head showing that WorldForge-style
decision traces can train a candidate-ranking model.

## Training Data

Source package:

```text
hf_dataset/
```

Rows come from:

- curated real Unitree Go2 robot-view seed frames,
- label-safe real-photo-edit counterfactuals using real Go2 plates and real cube
  cutouts,
- transparent scorer labels.

Labels are not measured long-horizon execution outcomes.

## Metrics

Current local evaluation:

```text
test selection accuracy: 97.9%
test MAE: 0.0234
test R2: 0.944
random baseline: 25%
always-forward baseline: 31.2%
```

These metrics measure fit to the transparent scorer labels. They do not prove
real-world navigation success.

## Files

- `model.json`: feature names, weights, metadata, and training summary.
- `eval_report.json`: split metrics and baseline comparisons.
- `predictions_sample.json`: sample predicted scores.

## Intended Use

Use this for:

- demos of inspectable robot candidate scoring,
- WorldForge-style score-provider integration tests,
- dataset/model-card examples for future robot outcome collection.

Do not use it as a safety controller or as the sole basis for robot motion.
