# Hugging Face Upload Runbook

The dataset and model are upload-ready, but this machine is not currently logged
in to Hugging Face.

## Login

```bash
hf auth login
hf auth whoami
```

Use a token with write access.

## Dataset

```bash
hf upload \
  omarespejel/worldforge-go2-trace-judge-dataset \
  hf_dataset \
  . \
  --repo-type dataset \
  --commit-message "Add Go2 trace judge dataset"
```

Expected package:

```text
README.md
contact_sheet.jpg
dataset_summary.json
data/*.jsonl
images/*.jpg
masks/*.png
```

## Model

```bash
hf upload \
  omarespejel/go2-cube-micro-world-scorer \
  hf_model \
  . \
  --repo-type model \
  --commit-message "Add Go2 cube micro world scorer"
```

Expected package:

```text
README.md
model.json
eval_report.json
predictions_sample.json
```

## JEPA-Style Model

```bash
hf upload \
  omarespejel/go2-cube-micro-jepa-scorer \
  hf_model_jepa \
  . \
  --repo-type model \
  --commit-message "Add Go2 cube micro JEPA-style latent scorer"
```

## DINOv2 Hybrid Model

```bash
hf upload \
  omarespejel/go2-cube-dinov2-hybrid-scorer \
  hf_model_dinov2 \
  . \
  --repo-type model \
  --commit-message "Add Go2 cube DINOv2 hybrid scorer ablation"
```

## Claim Boundary

Use this wording:

> Experimental micro world scorer trained on transparent labels from real Go2
> frames and label-safe real-photo-edit counterfactuals.

Do not describe it as a Go2 foundation world model, trained V-JEPA model, or
safety-certified robot controller.
