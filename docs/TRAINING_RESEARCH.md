# Training Research Note

## Recommendation

Do not spend the hackathon on GPU fine-tuning a Go2 world model. Use GPUs only after
we have collected candidate/outcome traces from the venue.

For the demo, the strongest claim is:

```text
transparent scorer now
robot trace dataset collected now
learned scorer later
```

## Why Not Train Tonight

WorldForge scoring wants counterfactual candidate data:

```text
observation + goal + candidate action -> predicted outcome quality
```

Most available robot datasets are policy or locomotion datasets:

```text
observation -> expert action
```

That is useful for imitation learning, but it does not directly answer "which of these
candidate futures is best in this venue?"

## Dataset Findings

### VLN-Go2-Matterport

Source: https://huggingface.co/datasets/Kennylhw/VLN-Go2-Matterport

Best use:
- future simulated Go2 visual navigation policy or representation work
- RGB/depth/instruction/action data for navigation

Limitations for this hackathon:
- simulated Matterport/Isaac Lab, not venue camera frames
- expert path data, not candidate outcome scoring labels
- dataset card links download outside normal HF files

### bolt-lab/go2-wm-indoor-1hr

Source: https://huggingface.co/datasets/bolt-lab/go2-wm-indoor-1hr

Best use:
- promising future inspection target because the repo contains `go2_wm_indoor_1hr.h5`

Limitations:
- README is basically empty
- the H5 is about 1.34 GB, so it is inspectable later but not something to
  blindly train against during robot time
- needs schema inspection before trusting it
- not enough evidence yet that it has goal/candidate/outcome labels

### m3/go2z1-grasp-vision-v2

Source: https://huggingface.co/datasets/m3/go2z1-grasp-vision-v2

Best use:
- Go2+Z1 manipulation / GR00T or LeRobot-style fine-tuning story
- useful if the hackathon has an arm-on-Go2 setup

Limitations:
- arm grasping, not plain Go2 visual navigation
- non-commercial license
- does not solve colored-target navigation
- one sampled episode parquet is tiny, but the full dataset is many episodes and
  targeted at Go2+Z1 arm grasping, not dog-only movement

### MIMUW-Robotics/kine2go

Source: https://huggingface.co/datasets/MIMUW-Robotics/kine2go

Best use:
- locomotion/motion-retargeting research
- Go2 gait/motion foundation model background

Limitations:
- motion/rollout data, not camera-target navigation
- not a score provider for this task
- useful for locomotion background, but disconnected from visual target selection

## What To Train Later

Once we collect traces, train a tiny ranker:

Input:

```text
target_confidence
target_bearing
target_area
distractor_confidence
unsafe_color_risk
candidate_id
candidate_features
```

Output:

```text
candidate quality score
```

Labels:

```text
target moved closer to center
target area increased
unsafe risk decreased
manual intervention false
selected action succeeded
```

This can be trained cheaply on CPU or a small GPU. It is not a foundation model; it is a
learned scorer that can replace the transparent scorer behind the same WorldForge trace contract.

## What Is Included Now

This package includes a deliberately small training smoke test:

```bash
make dataset
make ranker
```

It writes:

```text
dataset/go2_trace_candidates.jsonl
artifacts/ranker_smoke/model.json
```

The label is currently `transparent_score_label`, so the model is only distilling the
transparent scorer. That is still useful because it proves the evidence contract can be
consumed by a model without changing the robot loop.

Do not present this as a trained Go2 world model. Present it as:

> the first step toward replacing a transparent scorer with a learned scorer, using the
> same trace schema.

## Hyperstack Use

Use Hyperstack only if we collect enough real traces:

1. Extract frame features with a frozen encoder such as CLIP/DINO.
2. Train a small MLP or gradient-boosted ranker over candidate features.
3. Export as a lightweight score provider.

Do not use Hyperstack for:

- full Go2 world-model training from scratch
- GR00T unless the task is manipulation with matching data
- Cosmos as a decision engine

## Honest Hackathon Claim

> We built the trace contract and host loop that makes learned robot scoring possible.
> The live demo uses a transparent scorer. The artifacts are exactly the data shape
> needed to train a learned scorer later.
