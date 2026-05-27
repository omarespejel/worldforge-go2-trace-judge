---
license: mit
tags:
- robotics
- unitree-go2
- world-models
- jepa-style
- decision-traces
- candidate-ranking
pipeline_tag: tabular-regression
---

# Go2 Cube Micro JEPA-Style Latent Scorer

This is an experimental JEPA-style candidate-action scorer for the WorldForge Go2
Trace Judge hackathon artifact.

It predicts an action-conditioned latent first:

```text
current observation latent + candidate action
-> predicted outcome latent
-> candidate score
```

The predicted latent contains:

```text
goal_alignment
information_gain
progress
obstacle_risk
stuck_risk
execution_cost
```

The score is then computed from the predicted latent using the same transparent
WorldForge-style candidate scoring weights used by the trace dataset.

## Why This Is JEPA-Style

The model does not regress directly from observation to final action. It predicts
an abstract latent representation of the candidate outcome and scores that latent.

```text
predict future latent -> evaluate future latent
```

## Claim Boundary

This is **not** a trained V-JEPA model, not a Go2 foundation world model, and not
a safety-certified autonomous navigation policy.

It is a small action-conditioned latent predictor trained on transparent
action-outcome latents from real Go2 frames and label-safe real-photo-edit
counterfactual traces.

## Metrics

Current local evaluation against transparent labels:

```text
test selected-action accuracy: 97.9%
test score MAE from predicted latents: 0.0200
test score R2 from predicted latents: 0.951
real_seed selected-action accuracy: 8/8
random baseline: 25%
always-forward baseline: 31.2%
```

Selected latent prediction metrics on the test split:

```text
goal_alignment R2: 0.947
information_gain R2: 0.985
progress R2: 0.929
obstacle_risk R2: 0.995
stuck_risk R2: 0.993
```

These metrics measure fit to transparent trace labels. They do not prove
long-horizon real-world navigation success.

## Files

- `model.json`: feature names, latent names, weight matrix, scoring weights, and
  training summary.
- `eval_report.json`: latent prediction metrics, score metrics, selection
  accuracy, and baselines.
- `predictions_sample.json`: sample predicted outcome latents and derived scores.

## Intended Use

Use this for demos of action-conditioned latent scoring and WorldForge-style
score-provider integration tests.

Do not use it as a safety controller or as the sole basis for robot motion.
