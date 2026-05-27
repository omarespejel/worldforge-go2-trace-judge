# Release Notes

## v0.1-hackathon-2026-05-28

Hackathon release for the WorldForge Go2 Trace Judge demo.

### Included

- Final 78-second demo video.
- One-command micro world scorer trace video.
- Hugging Face-ready dataset package.
- Hugging Face-ready model package.
- Optional JEPA-style and DINOv2-hybrid scorer packages.
- Model honesty audit with shuffled-label and plate-holdout controls.
- Submission bundle with evidence JSON artifacts.
- Reproducible `make hackathon-final` pipeline.

### Metrics

Current local evaluation against transparent trace labels:

```text
test selection accuracy: 97.9%
test MAE: 0.0234
test R2: 0.944
random baseline: 25%
always-forward baseline: 31.2%
real_seed selected-action accuracy: 8/8
```

Additional ML stretch:

```text
micro JEPA-style scorer: 97.9% selection accuracy, R2 0.951
DINOv2 hybrid scorer: 97.9% selection accuracy, R2 0.944
shuffled-label control: 21.8% mean selection accuracy, R2 -0.012
plate-holdout minimum selection accuracy: 92.9%
```

### Claim Boundary

This release demonstrates a small micro world scorer / latent action scorer over
real Go2 robot-view frames and label-safe counterfactual decision traces.

The optional JEPA-style scorer predicts an action-outcome latent before scoring.
The optional DINOv2 scorer uses a frozen visual backbone as an ablation. This
release does not claim to be a Go2 foundation world model, a trained V-JEPA model,
or a safety-certified autonomous robot controller.

### Key Commands

```bash
make hackathon-final
make micro-world-demo
make ml-stretch
make final-video
make package
```
