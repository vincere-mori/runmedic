# Playbook

Per runmedic check: what to look at in the code, how to confirm, how to fix. Snippets are PyTorch; translate them to the user's framework (Lightning, HF Trainer, JAX) when needed.

## non_finite: loss or grad norm became NaN/inf

**Look at:** the learning rate and schedule around the NaN step, gradient clipping, mixed precision setup, the loss function, input normalization.

**Confirm:**
- NaN at the very first steps → a bug: check one batch with `torch.isfinite(x).all()` for inputs and labels; look for `log`, `sqrt`, division in the loss.
- Loss or grad norm climbs for many steps before NaN → exploding gradients.
- NaN out of nowhere → find the op: run a few steps under `torch.autograd.set_detect_anomaly(True)` (slow, only for debugging).

**Fix:**
```python
# clip after backward, before step (with GradScaler: after scaler.unscale_(optimizer))
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```
```python
# warmup + decay
from transformers import get_cosine_schedule_with_warmup  # or write a LambdaLR
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05 * total), num_training_steps=total)
```
- Use `F.cross_entropy(logits, y)` instead of `log(softmax(...))`; `torch.log(x.clamp_min(1e-8))`.
- fp16: prefer `torch.autocast("cuda", dtype=torch.bfloat16)` on Ampere+; with fp16 keep `GradScaler`.
- Lightning: `Trainer(gradient_clip_val=1.0)`; HF Trainer: `max_grad_norm=1.0`, `warmup_ratio=0.05`.

## divergence: loss went down, then back up

**Look at:** the LR schedule (constant? decays too late?), weight decay on norm/bias params, data shuffling across epochs, any mid-run change of data source.

**Confirm:** compare the LR curve with the turning point; check `shuffle=True` and `sampler.set_epoch(epoch)` for `DistributedSampler`.

**Fix:** decaying schedule or lower peak LR; clipping; exclude norms/biases from weight decay:
```python
decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}], lr=lr)
```
Resume from the checkpoint closest to the best step instead of restarting.

## loss_spikes: isolated jumps in the training loss

**Look at:** warmup, LR, clipping, the data for the spike steps.

**Confirm:** log sample ids per batch (`return x, y, idx` from `__getitem__`), then inspect the batches at the reported steps: corrupted files, wrong labels, extreme lengths.

**Fix:** clipping, warmup, lower LR; filter bad samples; for LLM pretraining spikes, skipping the batch and lowering LR briefly is a known practice.

## no_learning: loss does not move

**Look at, in this order:**
1. `optimizer.zero_grad(); loss.backward(); optimizer.step()` all present and in this order.
2. Parameters passed to the optimizer are the ones being trained: `optimizer = Adam(model.parameters())` created after the model is final (not before `model = model.to(device)` re-wrapping, not before replacing the head).
3. `requires_grad` and frozen layers: `sum(p.numel() for p in model.parameters() if p.requires_grad)`.
4. LR value actually applied: `optimizer.param_groups[0]["lr"]`.
5. Labels match inputs (no separate shuffles, right column, right dtype for the loss).
6. Loss ≈ ln(num_classes) → uniform predictions: check the head size, the activations before the loss (no softmax before `CrossEntropyLoss`).

**Confirm:** overfit a single batch:
```python
xb, yb = next(iter(train_loader))
for i in range(300):
    loss = criterion(model(xb), yb)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
print(loss.item())  # must approach 0; if not, the bug is in model/loss/optimizer, not in data or hyperparameters
```

## overfitting: val loss rises while train loss falls

**Look at:** number of epochs vs dataset size, regularization (weight decay, dropout, augmentation), model size, whether the best checkpoint is kept.

**Fix:** early stopping and keeping the best checkpoint first (cheapest, no retraining needed if checkpoints exist):
- Lightning: `EarlyStopping(monitor="val_loss", patience=3)` + `ModelCheckpoint(monitor="val_loss")`
- HF Trainer: `load_best_model_at_end=True, metric_for_best_model="eval_loss", eval_strategy="steps"` + `EarlyStoppingCallback(early_stopping_patience=3)`

Then regularization and data: weight decay, augmentation, dropout, label smoothing, more data, smaller model / LoRA. Mild overfitting (info severity) is often fine; mention it, do not push big changes.

## val_below_train: val loss lower than train loss

Usually normal (dropout/augmentation only in train; train loss averaged over the epoch). Only dig deeper if val metrics look too good: check train/val overlap (hash samples, near-duplicate search), and whether val goes through the same preprocessing.

## grad_spikes: gradient norm spikes

Same family as loss spikes and NaN. Clip at about 1-2x the typical norm, add warmup, inspect the batches at spike steps. If spikes come from one layer, log per-layer norms to find it.

## constant_lr_plateau: no LR decay and the loss stalled

Add a decaying schedule (cosine to ~10% of peak, linear decay, or `ReduceLROnPlateau` on val loss), or stop earlier to save compute.

## input_bound: GPU waits for data

**Look at:** `DataLoader(num_workers=..., pin_memory=..., persistent_workers=..., prefetch_factor=...)`, the cost of `Dataset.__getitem__` (image decoding, augmentations, tokenization), storage (network drive, many small files), `.item()` / `.cpu()` / `print` every step.

**Confirm:**
```python
import time
t = time.perf_counter(); batch = next(it); data_t = time.perf_counter() - t
# ... forward/backward/step ...; torch.cuda.synchronize(); step_t = time.perf_counter() - t
```
or `torch.profiler.profile(...)` and look for gaps between CUDA kernels.

**Fix:**
```python
DataLoader(ds, batch_size=bs, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
x = x.to(device, non_blocking=True)
```
Pre-process once (resize, decode, tokenize offline into a fast format like WebDataset/Arrow/numpy memmap), move augmentations to GPU (kornia, torchvision v2 on tensors), log every N steps.

## slow_steps / slowdown: slow outlier steps or training slowing down

- Periodic slow steps every N steps → something in the loop runs every N steps: evaluation, checkpoint saving, heavy logging. Save asynchronously or less often; evaluate on a subset.
- Slow steps at epoch boundaries → workers restarting: `persistent_workers=True`.
- Gradual slowdown → memory growth (see below), growing sequence lengths, thermal throttling (`nvidia-smi -q -d PERFORMANCE`), another process on the GPU.

## memory_growth: memory keeps growing

**GPU:** look for tensors stored across steps with their graph:
```python
losses.append(loss)          # keeps the whole graph alive
losses.append(loss.item())   # fine
preds.append(out)            # in eval: use torch.no_grad() and out.detach().cpu()
```
Compare `torch.cuda.memory_summary()` at two steps. Variable input shapes can grow the CUDA cache without a leak: `torch.cuda.max_memory_allocated()` plateaus in that case.

**Host (RSS):** DataLoader workers + a Dataset holding Python lists/dicts of objects → copy-on-read from refcount updates. Store metadata in numpy arrays / Arrow. Unbounded caches, lists of metrics kept for the whole run. Use `tracemalloc` or `memray`.

## log checks (from stdout)

- **log:cuda_oom** → lower batch size + gradient accumulation, bf16 autocast, activation checkpointing (`model.gradient_checkpointing_enable()` for HF), check for leaks above, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for fragmentation.
- **log:dataloader_killed** → Docker `--shm-size=8g` or `--ipc=host`; fewer workers; check `dmesg` for the OOM killer.
- **log:amp_overflow** → a few skipped steps at the start are normal for fp16; constant overflow means LR too high or fp16 range too small → bf16.
- **log:nccl_timeout** → ranks out of sync: equal number of batches per rank (`drop_last=True`), evaluation/saving on all ranks or behind `dist.barrier()`; debug with `NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL`.
- **log:device_mismatch**, **log:backward_twice**, **log:inplace_grad**, **log:unused_params** → the message itself names the tensor or parameter; fix at that line.

## Instrumentation

Minimal logging that makes every check work. Plain PyTorch with TensorBoard:

```python
from torch.utils.tensorboard import SummaryWriter
import time
writer = SummaryWriter("runs/exp")
t_prev = time.perf_counter()
for step, (x, y) in enumerate(loader):
    t_data = time.perf_counter()
    loss = criterion(model(x.to(device, non_blocking=True)), y.to(device, non_blocking=True))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # returns the norm before clipping
    optimizer.step(); scheduler.step()
    if step % 10 == 0:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        now = time.perf_counter()
        writer.add_scalar("train/loss", loss.item(), step)
        writer.add_scalar("train/grad_norm", gnorm.item(), step)
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
        writer.add_scalar("perf/data_time", t_data - t_prev, step)
        writer.add_scalar("perf/step_time", now - t_prev, step)
        if torch.cuda.is_available():
            writer.add_scalar("perf/gpu_mem_allocated_mb", torch.cuda.memory_allocated() / 2**20, step)
    t_prev = time.perf_counter()
```

- **Lightning:** `self.log("train_loss", loss)`, `LearningRateMonitor`, `DeviceStatsMonitor`; grad norm via `self.log("grad_norm", ...)` in `on_before_optimizer_step`.
- **HF Trainer:** `logging_steps=10`, `report_to="tensorboard"`; `trainer_state.json` already has `loss`, `grad_norm`, `learning_rate`, `eval_loss`.
- **W&B:** `wandb.log({...}, step=step)`; system metrics (GPU util, memory) are logged automatically.
- **MLflow:** `mlflow.log_metric(key, value, step=step)`; `mlflow.enable_system_metrics_logging()` for GPU/CPU.
