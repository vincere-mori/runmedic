---
name: runmedic
description: Diagnose a machine learning training run from its logs and fix the code. Use when the user's training has a problem or they ask why it behaves a certain way - loss is NaN or inf, loss explodes, does not go down, spikes, plateaus, validation gets worse (overfitting), training is slow, the GPU is idle, the DataLoader is the bottleneck, memory grows, CUDA out of memory, NCCL timeouts. Works with TensorBoard, MLflow, Weights & Biases, Lightning CSV, HF Trainer, JSONL and plain stdout logs. Do not use for inference-only or data analysis questions.
---

# runmedic: training run doctor

You diagnose training runs like a senior ML engineer: evidence from the logs first, then the code, then a minimal fix, then a check that the fix worked. Never guess a cause the logs do not support.

## 1. Find the logs

Look in the project before asking. Typical places:

| Logger | Where |
|---|---|
| TensorBoard | `runs/`, `logs/`, `tb_logs/`, `lightning_logs/version_*/`, any dir with `events.out.tfevents*` |
| MLflow | `mlflow.db` (default since MLflow 3) or `mlruns/` |
| Weights & Biases | needs a run URL `https://wandb.ai/<entity>/<project>/runs/<id>` and `wandb login` |
| Lightning CSVLogger | `lightning_logs/version_*/metrics.csv` |
| HF Trainer | `<output_dir>/checkpoint-*/trainer_state.json` |
| stdout | `*.log`, `nohup.out`, SLURM `*.out`, or ask the user to paste / save the console output |

If several runs exist, ask which one, or diagnose the latest by modification time and say so.

## 2. Run the diagnosis

```bash
runmedic diagnose <path> --json
```

If `runmedic` is not installed, run it without installing:

```bash
uvx runmedic diagnose <path> --json
```

or install it: `pip install runmedic` (`pip install "runmedic[wandb]"` for W&B runs).

Check `roles` in the JSON: runmedic guesses which series is the train loss, val loss, learning rate, grad norm and so on. If a guess is wrong, list the series with `runmedic series <path>` and pass the right ones: `--train-loss NAME --val-loss NAME --lr NAME --grad-norm NAME --gpu-util NAME --step-time NAME`.

Every finding has `check`, `severity`, `evidence`, `likely_causes` and `fixes`. Treat `likely_causes` as hypotheses to confirm in the code, not as facts.

If runmedic cannot work (no logs at all, or the problem is not in the curves), fall back to reading the logs and code yourself with the same playbook.

## 3. Read the training code

For each finding, open the code that controls it and look for the concrete reason. [references/playbook.md](references/playbook.md) lists, per check, what to look at, how to confirm it and the usual fixes with code. Read the section for every check you got.

Typical places: optimizer and scheduler setup, the training step (`zero_grad` / `backward` / `step` order, clipping, autocast and GradScaler), the loss function, the `DataLoader` arguments, the `Dataset.__getitem__`, the evaluation loop, what gets appended to lists across steps, and config files.

## 4. Missing metrics

Some problems cannot be confirmed without a metric that is not logged (grad norm, learning rate, data wait time, GPU memory). Then propose a small logging patch from the "Instrumentation" section of the playbook and a short rerun (a few hundred steps) before changing the training itself. Ask the user before starting anything that takes long or uses a GPU they might need.

## 5. Report and fix

Write to the user, in their language:

1. **Diagnosis**: each problem in one sentence, with the evidence (step, values) from the report.
2. **Cause in the code**: file and line, and why it leads to this curve. If you could not find it in the code, say so.
3. **Fix**: a minimal diff, most likely and cheapest first. Change one thing at a time so the effect is measurable. Do not rewrite the training script.
4. **How to verify**: what to run and what the curve should look like after the fix.

If the report is clean, say that the curves look healthy for the checks that ran, and mention which checks could not run because their metrics are missing.

## 6. Verify

After the user (or you, if they agreed) reruns training, run `runmedic diagnose` on the new logs and compare: the finding should be gone or weaker. Report the before/after numbers.

## Rules

- Evidence first. Quote numbers from the report; do not invent steps or values.
- One problem can cause several findings (exploding gradients produce grad spikes and then NaN). Explain the chain once, starting from the root.
- Fix the root cause, not the symptom: do not hide NaN with `nan_to_num`, do not just raise the batch size to make an OOM go away if the real issue is a leak.
- Respect the user's setup: do not switch frameworks, optimizers or precision modes without saying why.
