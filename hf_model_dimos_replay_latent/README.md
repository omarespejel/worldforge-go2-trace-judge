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

## What Was Trained

Only a small ridge dynamics head was trained:

```text
frozen DINOv2 current-frame latent + egomotion/action delta
-> predicted residual future DINOv2 latent
-> current latent + residual = predicted future DINOv2 latent
```

The DINOv2 backbone remains frozen. This is not a trained V-JEPA model, not a
Go2 foundation model, and not a safety-certified controller.

## Evaluation

- Test future-latent cosine mean: `0.51247`
- Test no-motion cosine baseline: `0.506077`
- Test cosine lift vs no-motion: `0.006392`
- Test candidate scoring accuracy: `0.320988`

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
