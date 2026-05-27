# Release Notes

## v0.1-hackathon-2026-05-28

Hackathon release for the WorldForge Go2 Trace Judge demo.

### Included

- Final 78-second demo video.
- One-command micro world scorer trace video.
- Hugging Face-ready dataset package.
- Hugging Face-ready model package.
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

### Claim Boundary

This release demonstrates a small micro world scorer / latent action scorer over
real Go2 robot-view frames and label-safe counterfactual decision traces.

It does not claim to be a Go2 foundation world model, a trained V-JEPA model, or
a safety-certified autonomous robot controller.

### Key Commands

```bash
make hackathon-final
make micro-world-demo
make final-video
make package
```
