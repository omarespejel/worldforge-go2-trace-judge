# Model Audit

The trained models are intentionally small. They should be presented as scorer
interface proofs, not as evidence of learned long-horizon Go2 autonomy.

## Models

| model | input | test selected-action accuracy | test MAE | test R2 | interpretation |
|---|---|---:|---:|---:|---|
| random baseline | none | 25.0% | n/a | n/a | chance over four candidates |
| always-forward baseline | fixed action | 31.2% | n/a | n/a | simple behavior prior |
| geometry micro scorer | trace geometry + action | 97.9% | 0.0234 | 0.9438 | learns transparent scoring rule |
| micro JEPA-style scorer | predicts latent, then score | 97.9% | 0.0200 | 0.9514 | better architecture, still transparent-label distillation |
| DINOv2 hybrid scorer | frozen DINOv2 + geometry + action | 97.9% | 0.0234 | 0.9443 | visual features do not materially improve this dataset |

## Shuffled-Label Control

Training on shuffled labels collapses:

```text
mean selected-action accuracy: 21.8%
mean test R2: -0.0119
random baseline: 25.0%
```

This is good. It means the evaluation is not trivially high under destroyed
labels.

## Plate Holdout

Holding out entire source plates remains mostly strong:

```text
min selected-action accuracy: 92.9%
max selected-action accuracy: 100.0%
```

This shows the scorer transfers across the small set of real Go2 venue plates,
but it is still not evidence of general robot autonomy.

## Final Claim

The strongest honest claim is:

> We built a WorldForge-style decision trace interface, generated label-safe
> counterfactual Go2 scenes, trained small scorer heads, and audited the models
> to avoid overclaiming.

The weakest claim, which should be avoided:

> We trained a Go2 world model.
