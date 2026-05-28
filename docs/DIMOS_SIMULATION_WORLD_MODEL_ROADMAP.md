# DimOS Simulation And World-Model Roadmap

This note captures the post-hackathon path from the current replay-MPC artifact
to a repeatable no-hardware DimOS simulation loop.

## Short Answer

The strongest next step is not a larger ad hoc model. It is a closed evaluation
loop:

```text
DimOS replay or MuJoCo simulation
-> current robot observation and candidate egomotion actions
-> Go2 latent world model predicts candidate futures
-> WorldForge trace records scores and selected action
-> DimOS replay/sim validates the selected action path
```

The current project already has the first half:

- public DimOS Go2 replay data converted into action-conditioned frame pairs
- a frozen-DINOv2 residual latent dynamics head
- a no-robot replay-MPC demo that emits WorldForge-style trace artifacts
- Hugging Face dataset and model packages

The missing piece is a live DimOS simulation bridge that runs the same scorer
against MuJoCo or replay streams.

## What DimOS Gives Us

Primary DimOS docs confirm three relevant no-hardware surfaces:

1. Replay:

```bash
dimos --replay run unitree-go2
```

This replays a recorded Go2 navigation session and opens the command center with
Rerun visualization. Source: [DimOS Go2 guide](https://github.com/dimensionalOS/dimos/blob/main/docs/platforms/quadruped/go2/index.md).

2. MuJoCo simulation:

```bash
uv pip install 'dimos[base,unitree,sim]'
dimos --simulation run unitree-go2
```

DimOS describes this as the full Go2 navigation stack in MuJoCo, using the same
navigation code path as the robot stack. Source: [DimOS Go2 guide](https://github.com/dimensionalOS/dimos/blob/main/docs/platforms/quadruped/go2/index.md).

3. Agentic simulation:

```bash
dimos --simulation run unitree-go2-agentic
dimos --simulation run unitree-go2-agentic-ollama
```

The quickstart lists these as simulation runfiles with MCP/agent support.
Source: [DimOS quickstart](https://github.com/dimensionalOS/dimos/blob/main/docs/quickstart.mdx).

DimOS also exposes useful observation/control streams:

```text
registered_scan
odometry
way_point
nav_cmd_vel
corrected_odometry
global_map_pgo
```

The navigation stack uses `way_point` as the goal source, and `MovementManager`
can accept clicked points from viewer/agent sources while muxing nav and teleop
velocity commands. Source: [DimOS nav stack docs](https://github.com/dimensionalOS/dimos/blob/main/docs/capabilities/navigation/nav_stack.md).

## Where WorldForge Plugs In

WorldForge should stay above host control:

```text
DimOS owns:
  robot/sim connection
  perception streams
  navigation stack
  MCP skill execution
  safety limits

WorldForge owns:
  score_info
  candidate_scores
  selected_action
  outcome_after_execution
  replayable evidence
```

For simulation, the bridge should do this:

```text
DimOS frame/pose/costmap
-> generate candidate egomotion actions
-> latent world model predicts candidate futures
-> scorer ranks candidates
-> selected candidate becomes a DimOS waypoint or relative_move command
-> Rerun + JSON artifacts record the decision
```

That is the cleanest demonstration of "inspectable autonomy" because judges can
see both the robot/sim behavior and the evidence trail.

## Model Research Conclusion

### What We Have Now

The current model is a small action-conditioned latent world model:

```text
DINOv2(current_frame) + egomotion_delta
-> predicted DINOv2(future_frame)
```

It is small, auditable, and already evaluated against a no-motion baseline. The
published artifact is intentionally a residual dynamics head, not a foundation
model.

### DINOv2

DINOv2 is a strong frozen visual feature encoder. The official repository says
the models produce visual features that can be used directly with simple heads,
with no fine-tuning required for many tasks. Source: [DINOv2](https://github.com/facebookresearch/dinov2).

This supports our current architecture:

```text
frozen visual encoder + small action-conditioned prediction head
```

### DINOv3

DINOv3 is now the newer Meta line for dense visual features. It is relevant for a
future model upgrade, especially for dense scene understanding. The official repo
lists Hugging Face/Transformers support and multiple larger backbones, but model
weight access is more involved and it is not necessary for the current hackathon
artifact. Source: [DINOv3](https://github.com/facebookresearch/dinov3).

Recommended future experiment:

```text
DINOv3-S/B frozen features
-> same residual dynamics head
-> compare against current DINOv2-small ridge head
```

Ship it only if it improves test lift and replay candidate ranking.

### V-JEPA 2 / V-JEPA 2-AC

V-JEPA 2 is the right research north star, but not the right immediate training
target. Meta describes V-JEPA 2-AC as a latent action-conditioned world model
post-trained from V-JEPA 2 using robot trajectory data. Their blog says the
action-conditioned phase can work with 62 hours of robot data in their technical
report. Source: [Meta V-JEPA 2](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/),
[V-JEPA2 repo](https://github.com/facebookresearch/vjepa2).

Our replay dataset is useful, but it is still far smaller and pose-derived. So
the honest next step is:

```text
V-JEPA2/V-JEPA2.1 frozen video features
-> tiny action-conditioned head
-> replay/sim evaluation
```

Not:

```text
train V-JEPA from scratch
claim a Go2 foundation model
```

### LeWorldModel

LeWorldModel is closely aligned philosophically: encoder maps observations into
latents, predictor models dynamics in latent space, and planning scores action
sequences through predicted future embeddings. Source: [LeWorldModel](https://le-wm.github.io/).

It is a better paper to cite for our exact story than generic VLA policy papers,
because our project is not imitation control. It is candidate future scoring.

### LeRobot / Policy Models

LeRobot is valuable if we add arms or policy training. Its docs emphasize
standardized robotics datasets, hardware-agnostic robot interfaces, and policy
training/evaluation. Source: [LeRobot](https://github.com/huggingface/lerobot).

For Go2 navigation, LeRobot policies are not the immediate bottleneck. DimOS
already supplies navigation and MCP skills. Our differentiator is deciding among
candidate futures, not cloning a driver policy.

### Isaac Lab

Isaac Lab is useful for a longer sim-to-real program: GPU-accelerated robotics
simulation, sensors, RL, imitation learning, and motion planning. Source:
[Isaac Lab](https://github.com/isaac-sim/IsaacLab).

For this repo, it is not the next integration. DimOS already exposes MuJoCo Go2
simulation, which is closer to the hackathon stack and lower-friction.

### NVIDIA Cosmos

Cosmos may be interesting for future video prediction or synthetic world
generation, but the old `NVIDIA/Cosmos` repo is deprecated in favor of
`nvidia-cosmos`. It should not replace the current replay/simulation path.
Source: [NVIDIA Cosmos GitHub](https://github.com/NVIDIA/Cosmos).

## Definitive Next Steps

### Step 1 - Safe Local Probe

Run:

```bash
make dimos-sim-probe
```

Expected output:

```text
artifacts/dimos_simulation_probe/probe.json
artifacts/dimos_simulation_probe/next_commands.sh
```

This does not start simulation. It checks DimOS checkout availability, CLI
availability, Go2 blueprint names, and writes the exact next commands.

### Step 2 - Replay Visual Validation

Use this first because it does not need robot access or MuJoCo:

```bash
dimos --replay --viewer none run unitree-go2
```

Then compare the current repo artifact:

```bash
make replay-mpc-demo
open artifacts/replay_mpc_demo/predicted_vs_actual_future.jpg
```

Goal: show the selected candidate and actual future frame side-by-side.

### Step 3 - MuJoCo Smoke Test

Only after dependencies are installed:

```bash
uv pip install 'dimos[base,unitree,sim]'
dimos --simulation --viewer none run unitree-go2
```

If that works, run the agentic simulation:

```bash
dimos --simulation --viewer none run unitree-go2-agentic --daemon
dimos status
dimos mcp list-tools
```

### Step 4 - Closed-Loop Sim Bridge

Build a small bridge:

```text
DimOS simulated observation
-> six candidate relative moves
-> replay latent dynamics scorer
-> selected candidate
-> DimOS MCP `relative_move`
-> WorldForge trace artifacts
```

Start in dry-run mode:

```bash
python3 scripts/run_replay_mpc_demo.py \
  --dataset-dir hf_dataset_dimos_replay \
  --model-dir hf_model_dimos_replay_latent \
  --output-dir artifacts/replay_mpc_demo \
  --clean
```

Then add a DimOS-specific execution wrapper only after MCP tools are visible in
simulation.

### Step 5 - Stronger Model Experiments

Run these only after simulation is reproducible:

1. DINOv3 frozen encoder ablation.
2. V-JEPA2.1 frozen feature ablation.
3. More replay sources or longer replay horizons.
4. Candidate ranking objective instead of pure future-latent cosine.

Decision rule:

```text
publish only if test lift and replay candidate-ranking improve
```

## What This Enables

This moves the project from a hackathon artifact into a research loop:

```text
collect or replay robot trajectories
-> train small world-model scorer
-> evaluate in replay
-> validate in MuJoCo
-> run on physical Go2 with host-owned safety
```

That is aligned with WorldForge: the interface stays score/evidence first, while
DimOS remains the robot operating system.
