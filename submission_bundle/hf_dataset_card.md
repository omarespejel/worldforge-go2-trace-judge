---
license: mit
task_categories:
- robotics
- reinforcement-learning
tags:
- robotics
- unitree-go2
- world-models
- decision-traces
- counterfactual-evaluation
- synthetic-data
pretty_name: WorldForge Go2 Trace Judge Dataset
size_categories:
- n<1K
---

# WorldForge Go2 Trace Judge Dataset

This is a small decision-trace dataset for inspectable Unitree Go2 autonomy. It is
not an imitation-learning policy dataset. Each row records:

```text
observation + goal + candidate actions + transparent scores + selected action
```

The dataset is designed for scorer integration tests, learned-ranker smoke tests,
and future world-model scoring research.

## Splits

| split | rows | source |
|---|---:|---|
| train | 336 | curated real-photo-edit counterfactuals |
| validation | 96 | curated real-photo-edit counterfactuals |
| test | 48 | curated real-photo-edit counterfactuals |
| real_seed | 8 | curated real Go2 camera trace artifacts |

## Schema

Important fields:

- `image`: relative path to the robot-view or synthetic-view frame.
- `goal_text`: natural-language task description.
- `observation_summary`: compact robot/world state summary.
- `action_candidates`: possible bounded Go2 actions.
- `candidate_scores`: transparent scorer outputs for every candidate.
- `selected_candidate_id`: selected action candidate.
- `label_source`: currently `transparent_heuristic_score`.
- `source_domain`: `real_photo_edit` or curated real single-frame sources.

## Intended Use

Use this for:

- validating a WorldForge-style decision trace contract,
- scorer-distillation smoke tests,
- candidate-ranking model prototypes,
- debugging dataset pipelines for later learned world models.

Do not use this as evidence that a Go2 navigation policy was trained or validated.

## Synthetic Data Caveat

Real-photo-edit rows are useful because they provide controllable counterfactual scenes:
target left/right/center, close/far, and unsafe colored markers in the path while
keeping the background and cube assets tied to the actual Go2 venue captures.

- `real_photo_edit`: real Go2 camera plates plus real cube cutouts placed at new
  positions and scales.

They should not be mixed with real rows without keeping `source_domain` as an
explicit feature or split.

## Real Seed Caveat

`real_seed` rows come from curated hackathon Go2 camera material and scorer traces.
They are small and should be quality-reviewed before any broad public redistribution.
Labels are transparent scorer labels, not measured long-horizon outcomes.

## Roadmap

The next valuable version would add measured execution outcomes:

```text
candidate action -> actual movement/result -> success/collision/stuck/progress label
```

That would turn this from a scorer-contract dataset into a true learned world-model
or candidate-ranker dataset.
