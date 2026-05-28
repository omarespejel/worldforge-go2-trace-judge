---
license: apache-2.0
library_name: transformers
tags:
- unitree-go2
- dimos
- world-model
- latent-dynamics
- dinov2
- worldforge
---

# Go2 DimOS Replay Latent Dynamics Head

This is an experimental WorldForge-style world-model head trained on the derived
`WorldForge Go2 DimOS Replay World Pairs` dataset.

The current package uses 2,557 action-conditioned current/future Go2 replay pairs
from six usable public DimOS replay DBs.

## What Was Trained

Only a small `ridge` dynamics head was trained:

```text
frozen DINOv2 current-frame latent + egomotion/action delta
-> predicted residual future DINOv2 latent
-> current latent + residual = predicted future DINOv2 latent
```

The DINOv2 backbone remains frozen. This is not a trained V-JEPA model, not a
Go2 foundation model, and not a safety-certified controller.

## Evaluation

- Test future-latent cosine mean: `0.543147`
- Test no-motion cosine baseline: `0.524954`
- Test cosine lift vs no-motion: `0.018193`
- Test candidate scoring accuracy: `0.258486`
- Validation future-latent cosine mean: `0.631262`
- Validation no-motion cosine baseline: `0.580600`
- Validation cosine lift vs no-motion: `0.050662`
- Validation candidate scoring accuracy: `0.284595`

Candidate scoring uses the WorldForge-style contract:

```text
score(candidate) =
  cosine(predicted_future_latent(current_image, candidate_delta), goal_future_latent)
```

For evaluation, the real future frame provides the goal latent and the real
egomotion delta is ranked against counterfactual deltas.

## Limitations

- Tiny replay-derived dataset.
- Egomotion labels come from pose deltas, not raw joystick commands.
- The model is intended as an inspectable research/demo artifact.
- It should not be used for direct robot control or safety decisions.
