<div align="center">

# 🩺 runmedic

**A doctor for your training runs.** Point it at the logs, get a diagnosis with evidence and concrete fixes.
Works as a CLI and as a skill for coding agents (Claude Code, Codex and others).

[![CI](https://github.com/vincere-mori/runmedic/actions/workflows/ci.yml/badge.svg)](https://github.com/vincere-mori/runmedic/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Dependencies](https://img.shields.io/badge/dependencies-0-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

**English** · [Русский](README.ru.md)

</div>

---

Your loss went NaN at 3 a.m. The val loss started climbing two days ago. The GPU sits at 40% and nobody knows why. The curves are all there in TensorBoard or W&B, but reading them still takes an experienced eye.

runmedic reads the same logs and says what is wrong, shows the numbers that prove it, and tells you what to change:

```text
$ runmedic diagnose examples/logs/nan

Found: 1 critical, 1 warning

1. [CRITICAL] `train/loss` became NaN/inf at step 324
   · `train/loss` is inf at step 324 (last finite value 7.32e+33 at step 323)
   · grad norm jumped to 4.79e+11 before the NaN (4.85e+08x its median)
   · learning rate at that point: 2.49e-04
   · the learning rate was still rising when it broke: 124.5x its starting value
   Likely causes:
     - The LR schedule never stops increasing (warmup that never ends, wrong `total_steps`, or a custom lambda)
     - Exploding gradients: the learning rate is too high for this point of training
   What to try:
     1. Check the scheduler: print `scheduler.get_last_lr()` over steps and make sure it peaks and then decays
     2. Clip gradients: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
     3. Lower the peak learning rate 2-10x
     4. Resume from a checkpoint before the blow-up instead of restarting from scratch
```

That run is a real PyTorch training from [examples/](examples/broken_training.py) with a broken LR schedule, and the report points straight at it.

With the agent skill, your coding agent goes one step further: it runs the diagnosis, opens your training code, finds the line that causes it, proposes a minimal diff and checks the next run.

## What it catches

| Check | What it means | Needs |
|---|---|---|
| `non_finite` | Loss or grad norm became NaN/inf, and what led up to it: exploding loss, gradient blow-up, an LR that kept growing, or a bug from step 0 | loss |
| `divergence` | Loss went down and came back up, how much progress was lost | train loss |
| `no_learning` | Loss does not move; flags loss ≈ ln(k) (uniform prediction over k classes) and LR = 0 | train loss |
| `loss_spikes` | Sudden jumps vs the trend, with the steps to inspect; links early spikes to missing warmup | train loss |
| `overfitting` | Val loss rising while train loss falls, with the best checkpoint step | train + val loss |
| `val_below_train` | Val consistently below train: usually fine, sometimes leakage | train + val loss |
| `grad_spikes` | Gradient norm far above its median, suggested clipping threshold | grad norm |
| `constant_lr_plateau` | Loss stalled with no LR decay | LR + train loss |
| `input_bound` | GPU idle or most of the step spent waiting for data | GPU util or data/step time |
| `slow_steps` | Slow outlier steps, detects periodic ones (eval or checkpoint inside the loop) | step time |
| `slowdown` | Throughput dropping over the run | throughput |
| `memory_growth` | GPU or host memory growing steadily (leak patterns explained) | memory |
| `log:*` | Known failures in stdout: CUDA OOM, DataLoader worker killed / shm, AMP overflow, NCCL timeout, device mismatch, backward twice, in-place autograd errors, DDP unused params | text log |

Thresholds are deliberately conservative: a quiet report is more useful than a noisy one you learn to ignore. A healthy run gets "No problems found".

## Supported logs

| Source | Point it at |
|---|---|
| TensorBoard | a log dir or an `events.out.tfevents*` file (own parser, no TensorFlow needed) |
| MLflow | `mlflow.db` (default since MLflow 3) or a `mlruns/` dir |
| Weights & Biases | a run URL or `wandb:entity/project/run_id` (needs `pip install "runmedic[wandb]"` and `wandb login`) |
| PyTorch Lightning | `lightning_logs/version_N/` (TensorBoard) or `metrics.csv` |
| HF Transformers Trainer | the output dir (`trainer_state.json` from the latest checkpoint) |
| CSV / JSON Lines | any table with a step column |
| Plain stdout | `step 100 loss=0.52`, `{'loss': 0.52, ...}` dicts, nanoGPT-style `iter 100: loss 2.34`, Lightning progress bars |

A directory can mix them: runmedic finds every run inside and diagnoses each one. Metric names are recognized automatically (`train/loss`, `val_loss`, `eval_loss`, `learning_rate`, `grad_norm`, `system/gpu_0_utilization_percentage` and so on). `runmedic series PATH` shows how it interpreted them, and `--train-loss NAME` etc. override the guess.

## Install

```bash
pip install git+https://github.com/vincere-mori/runmedic
```

or run without installing:

```bash
uvx --from git+https://github.com/vincere-mori/runmedic runmedic diagnose runs/
```

Zero dependencies, Python 3.10+.

## Usage

```bash
runmedic diagnose runs/exp1                 # human-readable report
runmedic diagnose mlflow.db --run my-run    # only runs whose name contains "my-run"
runmedic diagnose runs/ --json              # for agents, scripts and dashboards
runmedic diagnose train.log --fail-on critical   # exit code 1 on critical findings, for CI
runmedic series runs/exp1                   # which series were found and how they were read
```

## Agent skill

The skill teaches a coding agent the whole loop: find the logs, run runmedic, read the training code, explain the cause with evidence, propose a minimal fix, verify on the next run. It ships with a [playbook](plugins/runmedic/skills/runmedic/references/playbook.md) of confirmations and fixes per check, and a logging snippet for runs that lack the metrics.

**Install in one line.** Send this to your agent:

```
Install the runmedic skill following these instructions: https://raw.githubusercontent.com/vincere-mori/runmedic/main/INSTALL.md
```

<details>
<summary>Install manually</summary>

**Claude Code:**

```
/plugin marketplace add vincere-mori/runmedic
/plugin install runmedic@runmedic
```

**Codex and other agents with `SKILL.md` support:**

```bash
git clone --depth 1 https://github.com/vincere-mori/runmedic.git
mkdir -p ~/.agents/skills && cp -r runmedic/plugins/runmedic/skills/runmedic ~/.agents/skills/
```

</details>

Then just ask: *"why did my training blow up?"*, *"is it overfitting?"*, *"why is the GPU at 40%?"*.

## Try it on the demo runs

```bash
git clone https://github.com/vincere-mori/runmedic && cd runmedic
pip install -e .
runmedic diagnose examples/logs
```

`examples/logs/` has four real PyTorch runs: `nan` (broken LR schedule), `overfit` (big model, tiny noisy dataset), `slow-data` (slow `__getitem__`, synchronous checkpoint every 50 steps) and `healthy`. Regenerate them with `python examples/broken_training.py <scenario>` (needs `torch` and `tensorboard`).

## How it works

1. **Load.** Each logger has a small reader. TensorBoard event files are parsed directly (TFRecord framing + a minimal protobuf decoder for scalar summaries), MLflow is read from its file store or sqlite database, W&B through its public API.
2. **Interpret.** Series names are mapped to roles (train loss, val loss, LR, grad norm, GPU util, memory, step/data time, throughput).
3. **Check.** Each check is a small heuristic over smoothed curves and robust statistics (EMA, median/MAD z-scores, rank correlation), looking for one failure mode and collecting evidence.
4. **Explain.** Findings carry evidence, likely causes ranked by what the evidence supports, and fixes.

No ML model inside, so it is fast, deterministic and works offline.

## Limitations

- It reads curves, not your code. It tells you *what* happened and the likely *why*; confirming the cause in code is the agent skill's job, or yours.
- Diagnosis quality depends on what you log. Without grad norm or LR, NaN reports are less specific; without step/data time, input bottlenecks are invisible. The playbook has a logging snippet.
- W&B offline runs (`wandb/offline-run-*`) are not read directly yet; sync them or export the history to CSV.

## Related work

Research tools such as AutoTrainer (ICSE 2021) and TrainCheck (OSDI 2025) detect training problems by instrumenting the training process. runmedic works after the fact on the logs you already have, across loggers, and is built to hand its findings to a coding agent that can then fix the code.

## Contributing

Bug reports with a log that was misdiagnosed are the most valuable contribution: open an issue with the run (or a CSV export of its metrics) and what was actually wrong. New checks and log formats are welcome; every check needs a test with a synthetic curve that triggers it and one that must not.

```bash
pip install -e . pytest tensorboard ruff
pytest && ruff check . && ruff format --check .
```

## License

MIT
