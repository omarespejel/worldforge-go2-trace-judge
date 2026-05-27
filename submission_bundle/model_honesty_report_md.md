# Model Honesty Audit

This audit is intentionally adversarial. Its purpose is to prevent the
hackathon model from being oversold as a real Go2 foundation world model.

## Claim Boundary

The model distills transparent score labels from real Go2 frames and label-safe counterfactuals. It is not a Go2 foundation world model or validated autonomy policy.

## Main Scorer

- test selected-action accuracy: `0.9792`
- test MAE: `0.02337`
- test R2: `0.94378`

## Shuffled-Label Control

A model trained on randomly shuffled training labels should collapse toward
the simple baselines. If it does not, the evaluation is leaking too much.

- repeats: `20`
- mean test selected-action accuracy: `0.217708`
- min/max: `0.0` / `0.604167`
- random baseline: `0.25`
- always-forward baseline: `0.3125`

## Plate Holdout

Each row below trains on every real-photo plate except one, then evaluates on
the held-out plate. This is more honest than a random row split because it
tests whether the scorer survives a new camera background.

| held-out group | groups | selection acc | MAE | R2 |
|---|---:|---:|---:|---:|
| `artifacts/live_ciro/direct_camera_final_preflight.jpg` | 112 | 0.9286 | 0.029614 | 0.80193 |
| `artifacts/live_ciro/direct_camera_no_red.jpg` | 85 | 1.0000 | 0.016742 | 0.980977 |
| `artifacts/live_ciro/direct_camera_red_block_front.jpg` | 110 | 0.9909 | 0.016652 | 0.982034 |
| `artifacts/live_ciro/direct_camera_red_block_left.jpg` | 105 | 1.0000 | 0.026655 | 0.922228 |
| `artifacts/live_ciro/direct_camera_red_block_right.jpg` | 68 | 1.0000 | 0.022939 | 0.972438 |

## Interpretation

This model is a useful score-provider smoke test and trace distillation.
It is not evidence of learned long-horizon robot control.
