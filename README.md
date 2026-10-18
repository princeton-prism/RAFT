# RAFT

*RAFT* is two-stage post-training for the MolmoBot-Pi0 vision-language-action policy. Stage I
aligns the PaliGemma backbone to embodied reasoning about robot frames with a LoRA, which is then
merged into the backbone. Stage II finetunes the full policy with the pi0 flow-matching action
loss from that merged backbone, with the vision tower frozen. Two simulated Franka tasks are
covered, **pick-and-place** (iTHOR FloorPlan13) and **open drawer**; every script takes
`TASK=pickplace` (default) or `TASK=opendrawer`.

## Installation

### Clone the repository (recursively)

```bash
git clone --recursive <this repo> raft
cd raft
```

If you cloned without `--recursive`:

```bash
git submodule update --init --recursive
```

### Setup environment with uv

Requires Python 3.11, [uv](https://docs.astral.sh/uv/getting-started/installation/), CUDA 12 and
EGL for headless MuJoCo.

```bash
bash scripts/setup_molmospaces.sh          # patches the molmospaces submodule, `uv sync --extra eval`,
                                           # then copies openpi's transformers overrides
export MLSPACES_CACHE_DIR=/path/to/cache   # MolmoSpaces scene/object assets (downloaded on first use); must be writable
# optional: export MLSPACES_ASSETS_DIR=/path/to/assets   # where assets are linked (default: molmospaces/assets)
```

### Download checkpoints and data

| Artifact | What | Path expected by the scripts |
|---|---|---|
| MolmoBot-Pi0-DROID | base policy, [allenai/MolmoBot-Pi0-DROID](https://huggingface.co/allenai/MolmoBot-Pi0-DROID) | `checkpoints/MolmoBot-Pi0-DROID` |
| RAFT Stage I, pick-and-place | reasoning LoRA merged into the base | `checkpoints/RAFT-stage1-pickplace` |
| **RAFT, pick-and-place** | Stage II | `checkpoints/RAFT-pickplace` |
| RAFT Stage I, open drawer | reasoning LoRA merged into the base | `checkpoints/RAFT-stage1-opendrawer` |
| **RAFT, open drawer** | Stage II | `checkpoints/RAFT-opendrawer` |
| action data | MolmoSpaces demos (h5 + mp4); split files in `config/data_splits/` | `data/<task>/action_train*` |
| reasoning data | Gemini reasoning JSONL + frames | `data/<task>/reasoning*` |

Hugging Face links: _to be added at release_. Every path can be overridden with an env var (see
[scripts/env.sh](scripts/env.sh)).

## Getting Started

### Minimal example

Evaluate a released checkpoint on the first 8 in-distribution episodes. This checks that the
environment, assets, checkpoint and simulator are wired up:

```bash
TASK=pickplace MODEL_DIR=checkpoints/RAFT-pickplace N=8 bash scripts/tests/smoke_test.sh
```

### Evaluate a released checkpoint on all axes

```bash
TASK=pickplace  MODEL_DIR=checkpoints/RAFT-pickplace  bash eval_raft.sh
TASK=opendrawer MODEL_DIR=checkpoints/RAFT-opendrawer bash eval_raft.sh
# or one Slurm job per axis:
TASK=pickplace MODEL_DIR=checkpoints/RAFT-pickplace bash scripts/slurm/eval_axes.sh
```

Results land under `outputs/<task>/eval/<tag>/<axis>/`, and `eval_raft.sh` prints the table at the end.

### Train from scratch

The released reasoning data is already annotated. To re-annotate frames with Gemini
(`scripts/stage1/annotate_dataset_paligemma.py`), use a separate environment with
`google-generativeai` installed and `GOOGLE_API_KEY` set; it conflicts with the protobuf version
the simulator needs, so it is not part of the main environment.

```bash
export TASK=pickplace                               # or opendrawer
bash scripts/stage1/train_reasoning_lora.sh
bash scripts/stage1/merge.sh                        # -> checkpoints/RAFT-stage1-$TASK
NUM_TRAIN_STEPS=<steps> bash scripts/stage2/train_action.sh
MODEL_DIR=outputs/$TASK/train_output/molmobot_pi0_droid/raft_stage2_$TASK/final bash eval_raft.sh
```

Baselines use the same Stage II recipe and evaluation:

```bash
INIT_CKPT_DIR=checkpoints/MolmoBot-Pi0-DROID EXP_NAME=plainft NUM_TRAIN_STEPS=<steps> bash scripts/stage2/train_action.sh   # plain FT
FINETUNED=<plain-FT checkpoint dir> ALPHA=<alpha> bash scripts/baselines/retain.sh                                     # RETAIN
```

## Evaluation Tasks

| Task | Axis | What changes from training |
|:---|:---|:---|
| `pickplace` | `id` | nothing (held-out episodes of the training scene) |
| | `oodlight` | randomized lighting |
| | `texall` | randomized textures on all objects |
| | `oodobjv3` | novel pick-up objects |
| | `oodprompt` | paraphrased task instructions |
| `opendrawer` | `id` | nothing (held-out episodes of the training scenes) |
| | `oodlight` | randomized lighting |
| | `oodvis` | randomized lighting and textures |
| | `texall` | randomized lighting and textures on all objects |
| | `oodprompt` | instructions paraphrased away from the training phrasing `Open the {X}` |

Benchmarks are in `config/benchmarks/<task>/`; [scripts/eval/eval_axis.sh](scripts/eval/eval_axis.sh)
maps each axis to its benchmark and randomization flags.

## How it works

1. **Annotate.** Gemini labels robot frames with embodied reasoning (task VQA, spatial relations,
   bounding boxes, keypoints, end-effector traces); coordinates are normalized to one scale
   (`scripts/stage1/`).
2. **Stage I: reasoning alignment.** A LoRA on the PaliGemma backbone is trained with next-token
   cross-entropy on that reasoning ([src/agent/train_reasoning.py](src/agent/train_reasoning.py)),
   then merged into the backbone (`scripts/stage1/merge.sh`).
3. **Stage II: action finetuning.** The full policy is finetuned with the pi0 flow-matching action
   loss from the merged backbone, vision tower frozen
   ([src/agent/train_action.py](src/agent/train_action.py)).
4. **Evaluate.** Closed-loop rollouts in MolmoSpaces on the axes above
   ([src/agent/eval.py](src/agent/eval.py)).

## More details

Repository layout:

```
src/agent/          trainers (train_reasoning.py, train_action.py), simulation eval, policies
src/data/           action and reasoning datasets, augmentation, prompt templates
src/model/          openpi train config (molmobot_pi0_droid)
src/utils/          shared helpers
config/eval/        hydra config for simulation evaluation
config/benchmarks/  benchmark.json per task and axis
config/data_splits/ trajectory index + normalization stats of the exact training (and held-out val) splits
scripts/            stage1/, stage2/, eval/, slurm/, baselines/, data/, results/, tests/, setup
molmospaces/        allenai/molmospaces@cd23bec (submodule), patched by scripts/molmospaces_raft.patch
eval_raft.sh        all evaluation axes for one checkpoint
```

This codebase has been tested with CUDA 12.

## Acknowledgements

The policy, data loading and trainers are adapted from
[allenai/MolmoBot](https://github.com/allenai/MolmoBot/tree/main/MolmoBot-Pi0), which builds on
[openpi](https://github.com/Physical-Intelligence/openpi). Simulation environments and benchmarks
are from [MolmoSpaces](https://github.com/allenai/molmospaces). Reasoning annotations are generated
with Gemini.

## License

Code: Apache-2.0 (see `LICENSE`, inherited from allenai/MolmoBot). The model weights derive from
PaliGemma and MolmoBot-Pi0-DROID and are released under the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms).
