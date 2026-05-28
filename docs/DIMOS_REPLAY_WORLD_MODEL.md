# DimOS Replay World-Model Stretch

This is the research-grade stretch layer for the hackathon submission. It uses
public DimOS Unitree Go2 replay data to build a small action-conditioned world
model dataset and a frozen-DINOv2 latent dynamics head.

Published artifacts:

- Dataset: https://huggingface.co/datasets/espejelomar/worldforge-go2-dimos-replay-world-pairs
- Model: https://huggingface.co/espejelomar/go2-dimos-replay-latent-dynamics

## Why This Shape

A real robot world model needs time-aligned observations and actions:

```text
observation_t + candidate_action_delta -> predicted observation_t+k
```

For this repo, the replay source is SQLite data from public DimOS LFS assets.
The builder exports:

- current robot-view frame,
- future robot-view frame,
- current/future pose,
- body-frame egomotion delta,
- side-by-side pair preview,
- explicit WorldForge scoring contract.

This is intentionally not trained from scratch. V-JEPA 2-style systems use huge
video pretraining and a separate action-conditioned robot phase; this repo trains
only a small action-conditioned head on top of frozen visual features.

## Source Replays

The build target uses:

```text
go2_short
markers_go2
go2_bigoffice
go2_china_office
```

`go2_china_office` is downloaded and inspected, but skipped for action-conditioned
pairs because its `color_image` pose fields are null in this replay export. The
script records this in `dataset_summary.json` instead of inventing labels.

Current derived dataset:

```text
pair_count: 540
train: 378
validation: 81
test: 81
usable source frames: 6478
unique exported frames: 1071
```

## Hugging Face Layout

The dataset is packaged two ways:

```text
hf_dataset_dimos_replay/
  data/train.jsonl
  data/validation.jsonl
  data/test.jsonl
  images/frames/
  images/pair_previews/
  imagefolder/train/metadata.jsonl
  imagefolder/validation/metadata.jsonl
  imagefolder/test/metadata.jsonl
  README.md
  dataset_summary.json
  provenance.json
```

The `data/*.jsonl` files are convenient for training scripts. The
`imagefolder/<split>/metadata.jsonl` layout follows Hugging Face ImageFolder
practice: preview JPEGs live next to split-local metadata rows with a `file_name`
field.

## Model

The model package is:

```text
hf_model_dimos_replay_latent/
  README.md
  model.json
  eval_report.json
  candidate_scores_sample.json
```

Architecture:

```text
frozen DINOv2 current-frame CLS latent
+ egomotion/action delta
-> residual future latent head
-> current latent + residual = predicted future latent
```

Only the ridge residual head is trained. The DINOv2 backbone is frozen.

## Current Metrics

Held-out chronological split metrics:

```text
validation future cosine: 0.531886
validation no-motion baseline: 0.518609
validation lift: +0.013276

test future cosine: 0.512470
test no-motion baseline: 0.506077
test lift: +0.006392

validation candidate-ranking accuracy: 44.4%
test candidate-ranking accuracy: 32.1%
random among six candidates: 16.7%
```

Interpretation:

- The learned head gives a small positive lift over visual persistence.
- Candidate ranking is above random but not strong enough to claim autonomous
  control.
- This is a dataset/model scaffold for future WorldForge score providers, not a
  safety controller.

## Rebuild

```bash
make dimos-replay-stretch
```

This downloads raw replay archives into the ignored cache:

```text
artifacts/dimos_replay_lfs_cache/
```

Only the derived HF-ready dataset/model artifacts are intended for publication.

## Claim Boundary

Say:

> We derived action-conditioned Go2 world-model pairs from public DimOS replays
> and trained a small frozen-DINOv2 residual dynamics head for WorldForge-style
> candidate scoring.

Do not say:

> We trained a Go2 foundation model, trained V-JEPA, or created a safety-certified
> robot controller.
