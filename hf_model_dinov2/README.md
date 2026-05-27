---
license: mit
tags:
- robotics
- unitree-go2
- dinov2
- frozen-embeddings
- decision-traces
- candidate-ranking
pipeline_tag: tabular-regression
---

# Go2 Cube DINOv2 Hybrid Scorer

This is an ablation model for the WorldForge Go2 Trace Judge artifact.

It uses:

```text
frozen DINOv2 image embedding
+ geometry/action trace features
+ ridge scorer head
-> candidate score
```

The DINOv2 backbone is frozen. The only trained part is the small score head.

## Result

Current local evaluation against transparent trace labels:

```text
test selected-action accuracy: 97.9%
test MAE: 0.0234
test R2: 0.944
real_seed selected-action accuracy: 8/8
```

This essentially matches the geometry-only scorer rather than improving it.

## Interpretation

The result is useful because it prevents overclaiming. Full-frame DINOv2 features
do not add meaningful signal for this dataset because:

- the red cube is very small in the Go2 camera frame,
- the labels are generated from transparent geometry/risk features,
- the dataset is a scorer-interface demo, not measured real-world outcome data.

## Claim Boundary

This is not a fine-tuned DINOv2 model, not a Go2 foundation world model, and not
a validated autonomy policy.

It is a frozen-embedding ablation showing that pretrained visual features can be
plugged into the WorldForge-style score contract, while geometry/risk traces
remain the stronger signal for this specific dataset.
