---
license: apache-2.0
task_categories:
- robotics
- image-to-image
tags:
- unitree-go2
- dimos
- world-model
- jepa-style
- worldforge
- egomotion
pretty_name: WorldForge Go2 DimOS Replay World Pairs
size_categories:
- 1K<n<10K
---

# WorldForge Go2 DimOS Replay World Pairs

This dataset is a compact, derived world-model dataset built from public
[`dimensionalOS/dimos`](https://github.com/dimensionalOS/dimos) Unitree Go2 replay assets.

It is designed for the WorldForge score contract:

```text
current robot-view image + candidate egomotion/action delta
-> predicted future visual latent
-> score against a goal/future latent
```

## Contents

- Source replay frames: `20918`
- Exported frame pairs: `2557`
- Unique exported frames: `5046`
- Splits: `{"test": 383, "train": 1791, "validation": 383}`
- Source pair counts: `{"go2_bigoffice": 500, "go2_china_office": 0, "go2_hongkong_office": 500, "go2_short": 203, "go2_slamabuse1": 500, "go2_slamabuse2": 500, "markers_go2": 354}`

Source replay assets:

- `go2_short`: `data/.lfs/go2_short.db.tar.gz` / `8a19846a0adf5755815fd039492c0255e0bc282e9df75a06648d7585cae8d2d2`
- `markers_go2`: `data/.lfs/markers_go2.db.tar.gz` / `5a43529f8dbc2aedcccca6ae89747235826123c2bc066e0dc8b87c2042219dae`
- `go2_bigoffice`: `data/.lfs/go2_bigoffice.db.tar.gz` / `e66f5472e72f370446d8dcd802f70f3c3c07e4e083c5d6a394873877dec4c88d`
- `go2_hongkong_office`: `data/.lfs/go2_hongkong_office.db.tar.gz` / `d1bb7de9a090b4053ba1ee4f36d776e439d970cba08ebb489f9311f26946f56c`
- `go2_slamabuse1`: `data/.lfs/go2_slamabuse1.db.tar.gz` / `a85feac43debdebf344c567483ab7d1bec12c3cf9e4df26034260a24e225f219`
- `go2_slamabuse2`: `data/.lfs/go2_slamabuse2.db.tar.gz` / `7d9a13596cf3d9a50e437fa89e8a3d68d843587116681564b4de7422b53c54dd`
- `go2_china_office`: `data/.lfs/go2_china_office.db.tar.gz` / `834539871fd325b15f3079a3490b278c54e78d0d40bfa1342dbdc983f6a3ee02`

Each row includes:

- `current_image`
- `future_image`
- `file_name` side-by-side preview for the Hugging Face image viewer
- timestamps and poses
- `egomotion_delta`
- the explicit world-model scoring contract

The repository also includes `imagefolder/train`, `imagefolder/validation`, and
`imagefolder/test` directories. Each split has pair-preview JPEGs plus a
`metadata.jsonl` file with the same labels, so it can be loaded with the standard
Hugging Face `imagefolder` builder.

## Provenance

The source material comes from `dimensionalOS/dimos`, whose checked-in `LICENSE`
file is Apache License 2.0. This repository currently reports license metadata
as `Other` on GitHub, so users should verify the source license text directly.

## Intended Use

This dataset is intended for:

- small latent-dynamics demos,
- action-conditioned future prediction experiments,
- WorldForge score-provider prototyping,
- educational robotics evidence-trace examples.

## Limitations

- This is not a broad robot foundation dataset.
- It is a small replay-derived dataset.
- The action labels are derived from pose deltas between frames, not raw joystick
  commands.
- It is not suitable for safety validation or direct robot control.
- Indoor replay imagery may contain real-world office context.

## Citation / Attribution

If you use this dataset, attribute both:

- DimensionalOS / DimOS as the source of the public replay data.
- WorldForge Go2 Trace Judge as the derived dataset/scoring package.
