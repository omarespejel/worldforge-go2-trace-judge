# Final Submission

## Project Name

WorldForge Go2 Trace Judge

## One-Liner

An inspectable decision layer for Unitree Go2 autonomy: real robot-view frames,
counterfactual candidate scenes, small learned scorers, and replayable evidence.

## Links

- GitHub repo: https://github.com/omarespejel/worldforge-go2-trace-judge
- Release: https://github.com/omarespejel/worldforge-go2-trace-judge/releases/tag/v0.1-hackathon-2026-05-28
- Final voiced video: pending final voiceover upload.
- Source/artifact archive: https://github.com/omarespejel/worldforge-go2-trace-judge/releases/download/v0.1-hackathon-2026-05-28/worldforge-go2-trace-judge.zip
- HF replay dataset: https://huggingface.co/datasets/espejelomar/worldforge-go2-dimos-replay-world-pairs
- HF replay model: https://huggingface.co/espejelomar/go2-dimos-replay-latent-dynamics

## What It Does

Most robot demos show movement but hide the decision process. This project turns
a Unitree Go2 camera frame into candidate future actions, scores those candidates,
selects the best one, and writes the evidence trail.

```text
observation + goal + candidate action -> score -> selected action -> evidence
```

The robot execution boundary stays host-owned. The contribution is the scoring,
comparison, and replay layer around robot actions.

## What We Built

- Curated real Go2 robot-view frames from the venue.
- Label-safe real-photo-edit counterfactual dataset using real Go2 plates and
  real cube cutouts.
- Hugging Face-ready dataset package.
- Micro world scorer trained on decision traces.
- Micro JEPA-style latent scorer: predicts action-outcome latent, then scores.
- Frozen-DINOv2 hybrid scorer ablation.
- Public DimOS replay-derived world-model dataset: 540 action-conditioned
  current/future Go2 frame pairs from three usable replay DBs.
- Frozen-DINOv2 residual dynamics head trained on those replay pairs.
- Model honesty audit with shuffled-label and plate-holdout controls.
- One-command scorer demo that writes MP4 + JSON evidence.
- 60-second final judge video, designed for external voiceover and final upload.

## Key Results

```text
geometry micro scorer: 97.9% selection accuracy, R2 0.9438
micro JEPA-style scorer: 97.9% selection accuracy, R2 0.9514
frozen-DINOv2 hybrid scorer: 97.9% selection accuracy, R2 0.9443
shuffled-label control: 21.8% mean selection accuracy, R2 -0.0119
plate-holdout minimum selection accuracy: 92.9%
DimOS replay latent head validation lift vs no-motion: +0.0133 cosine
DimOS replay latent head test lift vs no-motion: +0.0064 cosine
DimOS replay candidate ranking accuracy: 44.4% validation, 32.1% test
```

## Claim Boundary

This is **not** a trained Go2 foundation world model, trained V-JEPA model, or
safety-certified autonomous controller.

The DimOS replay stretch trains only a small residual head on top of frozen
DINOv2 features. It is included as an honest world-model-data contribution, not
as a claim of robust learned Go2 control.

The DimOS media in the final video is used only to show the underlying robot OS
context. It is not presented as our robot run, training data, or proof of our
model behavior.

The honest claim is stronger for a hackathon:

> We built an inspectable robot decision trace system, generated counterfactual
> Go2 training data, trained small candidate scorers, and audited the models so
> the demo does not overclaim.

## 20-Second Pitch

Robots should not just emit actions. They should show the options they considered,
why one option won, and what evidence can be used to improve the scorer later.
WorldForge Go2 Trace Judge turns Go2 robot-view frames into replayable candidate
scoring traces.
