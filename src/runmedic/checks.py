"""Heuristic checks over a run. Each one looks for a single failure mode and explains it with evidence from the logs.

Thresholds are deliberately conservative: a quiet report is better than a noisy one people learn to ignore.
"""

from __future__ import annotations

import bisect
import math
import statistics
from collections import defaultdict

from runmedic import logpatterns
from runmedic.model import SEVERITY_ORDER, Finding, Run, Series
from runmedic.stats import ema, fmt, fmt_step, robust_z, spearman


class Ctx:
    def __init__(self, run: Run, roles: dict[str, list[str]]):
        self.run = run
        self.roles = roles
        self.log_ids = {p.id for p in (logpatterns.match(e.text) for e in run.log_lines) if p}

    def raw(self, role: str) -> Series | None:
        names = self.roles.get(role)
        return self.run.series[names[0]] if names else None

    def finite(self, role: str) -> Series | None:
        s = self.raw(role)
        return s.finite() if s else None

    def all_finite(self, role: str) -> list[Series]:
        return [self.run.series[n].finite() for n in self.roles.get(role, [])]

    def value_at(self, role: str, step: float) -> float | None:
        s = self.finite(role)
        if not s or not len(s):
            return None
        i = bisect.bisect_right(s.steps, step) - 1
        return s.values[max(i, 0)]

    def before_blowup(self, s: Series) -> Series:
        """Drops the run-up to the first NaN/inf so spike checks do not repeat what non_finite already explains."""
        first_bad = None
        for role in ("train_loss", "grad_norm"):
            raw = self.raw(role)
            if raw:
                bad = [st for st, v in zip(raw.steps, raw.values) if not math.isfinite(v)]
                if bad and (first_bad is None or bad[0] < first_bad):
                    first_bad = bad[0]
        if first_bad is None:
            return s
        keep = [i for i, st in enumerate(s.steps) if st < first_bad]
        cut = max(0, len(keep) - max(10, len(keep) // 10))
        out = Series(s.name)
        for i in keep[:cut]:
            out.add(s.steps[i], s.values[i])
        return out

    def lr_warmup(self) -> bool | None:
        """True if the LR grows at the start, False if it starts at its peak, None if unknown."""
        lr = self.finite("lr")
        if not lr or len(lr) < 5:
            return None
        head = lr.values[: max(3, len(lr) // 10)]
        return head[-1] > head[0] * 1.5


def _head_tail(values: list[float], frac: float = 0.1) -> tuple[float, float]:
    k = max(1, int(len(values) * frac))
    return statistics.fmean(values[:k]), statistics.fmean(values[-k:])


# ---------------------------------------------------------------- loss health


def check_non_finite(ctx: Ctx) -> list[Finding]:
    for role in ("train_loss", "val_loss", "grad_norm"):
        s = ctx.raw(role)
        if not s:
            continue
        bad = [i for i, v in enumerate(s.values) if not math.isfinite(v)]
        if not bad:
            continue
        i = bad[0]
        step = s.steps[i]
        evidence = [
            f"`{s.name}` is {s.values[i]} at step {fmt_step(step)}"
            + (f" (last finite value {fmt(s.values[i - 1])} at step {fmt_step(s.steps[i - 1])})" if i else "")
        ]
        if len(bad) > 1:
            evidence.append(f"{len(bad)} of {len(s)} logged points are NaN/inf after that")

        exploding = False
        before = [v for v in s.values[:i] if math.isfinite(v)]
        if len(before) >= 10:
            window = before[-max(5, len(before) // 10) :]
            base = statistics.median(before[: len(before) // 2])
            if base > 0 and max(window) > 3 * base:
                exploding = True
                evidence.append(f"`{s.name}` climbed to {fmt(max(window))} right before, {fmt(max(window) / base)}x its earlier median")
        gn = ctx.finite("grad_norm") if role != "grad_norm" else None
        if gn and len(gn) >= 10:
            prior = [v for st, v in zip(gn.steps, gn.values) if st <= step]
            if len(prior) >= 10:
                base = statistics.median(prior[: len(prior) // 2])
                peak = max(prior[-max(5, len(prior) // 10) :])
                if base > 0 and peak > 5 * base:
                    exploding = True
                    evidence.append(f"grad norm jumped to {fmt(peak)} before the NaN ({fmt(peak / base)}x its median)")
        lr = ctx.value_at("lr", step)
        warmup = ctx.lr_warmup()
        lr_growing = False
        if lr is not None:
            evidence.append(f"learning rate at that point: {fmt(lr)}" + (" (no warmup)" if warmup is False else ""))
            lr_s = ctx.finite("lr")
            lr0 = lr_s.values[0] if lr_s and len(lr_s) else 0
            if lr0 > 0 and lr > 3 * lr0 and lr >= 0.99 * max(v for st, v in zip(lr_s.steps, lr_s.values) if st <= step):
                lr_growing = True
                evidence.append(f"the learning rate was still rising when it broke: {fmt(lr / lr0)}x its starting value")

        from_start = i == 0 or len(before) < 3
        causes, fixes = [], []
        if lr_growing:
            causes.append("The LR schedule never stops increasing (warmup that never ends, wrong `total_steps`, or a custom lambda)")
            fixes.append("Check the scheduler: print `scheduler.get_last_lr()` over steps and make sure it peaks and then decays")
        if from_start:
            causes += [
                "NaN from the very first steps points to a bug, not optimization: NaN/inf in inputs or labels, "
                "log(0) or division by zero in the loss, or bad initialization"
            ]
            fixes += [
                "Check a batch: `assert torch.isfinite(x).all()` for inputs and labels",
                "Use numerically stable ops: `F.cross_entropy` on logits instead of `log(softmax)`, `torch.clamp(x, min=eps)` before `log`/division",
                "Run one step under `torch.autograd.set_detect_anomaly(True)` to find the op that produced NaN",
            ]
        if exploding:
            causes.append(
                "Exploding gradients: the learning rate is too high for this point of training"
                + (" and there is no warmup" if warmup is False else "")
            )
        elif not from_start:
            causes.append(
                "Sudden blow-up with no visible lead-up: a single bad batch, or an unstable op "
                "(log/sqrt/division near zero, attention logits overflowing)"
            )
            fixes.append(
                "Log the ids of the samples in each batch and inspect the batch at that step; "
                "rerun it under `torch.autograd.set_detect_anomaly(True)`"
            )
        if exploding or not from_start:
            fixes += [
                "Clip gradients: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`",
                "Lower the peak learning rate 2-10x" + ("; add linear warmup for the first ~5% of steps" if warmup is not True else ""),
                "Resume from a checkpoint before the blow-up instead of restarting from scratch",
            ]
        if "amp_overflow" in ctx.log_ids or not from_start:
            causes.append("fp16 overflow under mixed precision (if you use AMP with float16)")
            fixes.append("Prefer bf16 autocast; with fp16 keep `GradScaler` and clip after `scaler.unscale_(optimizer)`")
        return [
            Finding(
                "non_finite", "critical", f"`{s.name}` became NaN/inf at step {fmt_step(step)}", evidence, causes, fixes, step=step, metric=s.name
            )
        ]
    return []


def check_divergence(ctx: Ctx) -> list[Finding]:
    s = ctx.finite("train_loss")
    if not s or len(s) < 20:
        return []
    raw = ctx.raw("train_loss")
    if raw and any(not math.isfinite(v) for v in raw.values):
        return []  # already reported as non_finite
    sm = ema(s.values)
    start, _ = _head_tail(sm, 0.05)
    ib = min(range(len(sm)), key=sm.__getitem__)
    best, final = sm[ib], sm[-1]
    progress = start - best
    if progress <= 0 or ib > 0.9 * len(sm) or final - best <= 0.5 * progress:
        return []
    lost = (final - best) / progress
    severity = "critical" if lost >= 1 else "warning"
    evidence = [
        f"smoothed `{s.name}` went from {fmt(start)} down to {fmt(best)} at step {fmt_step(s.steps[ib])} "
        f"and back up to {fmt(final)} at step {fmt_step(s.steps[-1])}",
        f"the run lost {lost:.0%} of the progress it had made",
    ]
    lr_best, lr_end = ctx.value_at("lr", s.steps[ib]), ctx.value_at("lr", s.steps[-1])
    if lr_best is not None and lr_end is not None:
        evidence.append(f"learning rate {fmt(lr_best)} at the best point, {fmt(lr_end)} at the end")
    return [
        Finding(
            "divergence",
            severity,
            f"Training loss diverged after step {fmt_step(s.steps[ib])}",
            evidence,
            [
                "Learning rate too high for the later phase of training (constant LR or a schedule that decays too late)",
                "Optimizer state blew up (Adam with very small eps, or weight decay applied to norms/biases)",
                "Data order changed: unshuffled data or a new data source mixed in mid-run",
            ],
            [
                "Use a decaying schedule (cosine or linear) and/or lower the peak LR",
                "Clip gradients (`clip_grad_norm_(..., 1.0)`) and exclude norms/biases from weight decay",
                "Check that the DataLoader shuffles (`shuffle=True` or `DistributedSampler.set_epoch(epoch)`)",
                f"Resume from the checkpoint closest to step {fmt_step(s.steps[ib])}",
            ],
            step=s.steps[ib],
            metric=s.name,
        )
    ]


def check_spikes(ctx: Ctx) -> list[Finding]:
    s = ctx.finite("train_loss")
    s = ctx.before_blowup(s) if s else None
    if not s or len(s) < 30:
        return []
    trend = ema(s.values, span=max(5, len(s) // 20))
    resid = [v - t for v, t in zip(s.values, [trend[0]] + trend[:-1])]
    z = robust_z(resid)
    spikes = [(s.steps[i], s.values[i], trend[i - 1]) for i in range(1, len(s)) if z[i] > 8 and trend[i - 1] > 0 and s.values[i] > 1.5 * trend[i - 1]]
    if not spikes:
        return []
    worst = max(spikes, key=lambda x: x[1] / x[2])
    severity = "warning" if len(spikes) >= 3 or worst[1] > 3 * worst[2] else "info"
    evidence = [
        f"{len(spikes)} sudden spike(s) in `{s.name}`; the worst is {fmt(worst[1])} at step "
        f"{fmt_step(worst[0])} against a trend of {fmt(worst[2])} ({fmt(worst[1] / worst[2])}x)"
    ]
    evidence.append("spike steps: " + ", ".join(fmt_step(st) for st, _, _ in spikes[:10]) + (" ..." if len(spikes) > 10 else ""))
    early = sum(1 for st, _, _ in spikes if st <= s.steps[len(s) // 10])
    warmup = ctx.lr_warmup()
    causes = ["Individual bad batches: corrupted samples, wrong labels, extreme outliers", "Learning rate too high, so single steps overshoot"]
    if early and warmup is False:
        causes.insert(0, f"{early} spike(s) happen in the first 10% of training and the LR starts at its peak: missing warmup")
    return [
        Finding(
            "loss_spikes",
            severity,
            f"Loss spikes ({len(spikes)})",
            evidence,
            causes,
            [
                "Log sample ids/paths per batch and look at the batches at the spike steps",
                "Clip gradients (`clip_grad_norm_(..., 1.0)`)",
                "Add LR warmup or lower the peak LR",
                "If a few samples are to blame, filter them or cap the per-sample loss",
            ],
            step=worst[0],
            metric=s.name,
        )
    ]


def check_no_learning(ctx: Ctx) -> list[Finding]:
    s = ctx.finite("train_loss")
    if not s or len(s) < 20:
        return []
    sm = ema(s.values)
    start, end = _head_tail(sm, 0.1)
    if start == 0:
        return []
    improvement = (start - min(sm)) / abs(start)
    if improvement >= 0.02:
        return []
    evidence = [f"`{s.name}` stayed around {fmt(start)} → {fmt(end)} over {len(s)} logged points (best improvement {improvement:.1%})"]
    causes = []
    lr = ctx.finite("lr")
    if lr and len(lr) and max(abs(v) for v in lr.values) == 0:
        causes.append("The learning rate is zero for the whole run (scheduler misconfigured or warmup never ends)")
        evidence.append("learning rate is 0 everywhere")
    k = math.exp(end) if 0.5 < end < 12 else 0
    if k and abs(k - round(k)) / k < 0.02 and round(k) >= 2:
        evidence.append(f"loss ≈ ln({round(k)}): if this is cross-entropy, the model predicts a uniform distribution over {round(k)} classes")
        causes.append("The model outputs the same prediction for every input (collapsed or not connected to the loss)")
    causes += [
        "Weights are not updated: `optimizer.step()` missing, `zero_grad()` called after `backward()`, "
        "parameters frozen (`requires_grad=False`) or not passed to the optimizer",
        "Learning rate far too low",
        "Labels are misaligned with inputs (shuffled separately, wrong column, augmentation mixes them)",
    ]
    return [
        Finding(
            "no_learning",
            "warning",
            "The model is not learning",
            evidence,
            causes,
            [
                "Overfit a single batch first: 200 steps on one fixed batch must drive the loss near zero",
                "Check `sum(p.numel() for p in model.parameters() if p.requires_grad)` and that those params are in the optimizer",
                "Make sure the order is `zero_grad() → forward → backward() → step()`",
                "Try a 10x higher learning rate for a few hundred steps",
            ],
            metric=s.name,
        )
    ]


def check_overfitting(ctx: Ctx) -> list[Finding]:
    v = ctx.finite("val_loss")
    t = ctx.finite("train_loss")
    if not v or len(v) < 4 or not t or len(t) < 4:
        return []
    iv = min(range(len(v)), key=v.values.__getitem__)
    best, last = v.values[iv], v.values[-1]
    if iv > len(v) - 4 or best == 0:
        return []
    # judge by the last few evaluations, not a single noisy one
    recent = statistics.fmean(v.values[-3:])
    rise = (recent - best) / abs(best)
    if rise < 0.05:
        return []
    t_sm = ema(t.values)
    t_at_best = t_sm[max(bisect.bisect_right(t.steps, v.steps[iv]) - 1, 0)]
    t_end = t_sm[-1]
    if t_at_best == 0 or (t_at_best - t_end) / abs(t_at_best) < 0.03:
        return []  # val got worse but train did not improve either: not classic overfitting
    return [
        Finding(
            "overfitting",
            "warning" if rise >= 0.15 else "info",
            ("Overfitting" if rise >= 0.15 else "Mild overfitting") + f" since step {fmt_step(v.steps[iv])}",
            [
                f"`{v.name}` was best at step {fmt_step(v.steps[iv])} ({fmt(best)}) and is now {fmt(last)} "
                f"(last 3 evaluations average +{rise:.0%}) after {len(v) - 1 - iv} more evaluations",
                f"meanwhile `{t.name}` kept falling: {fmt(t_at_best)} → {fmt(t_end)}",
            ],
            ["The model memorizes the training set: too many epochs for the data size, or not enough regularization"],
            [
                f"Keep the checkpoint from step {fmt_step(v.steps[iv])} and add early stopping on `{v.name}`",
                "Add regularization: weight decay, dropout, stronger augmentation, label smoothing",
                "Get more data or use a smaller model / fewer trainable parameters (e.g. LoRA instead of full fine-tune)",
            ],
            step=v.steps[iv],
            metric=v.name,
        )
    ]


def check_val_below_train(ctx: Ctx) -> list[Finding]:
    v = ctx.finite("val_loss")
    t = ctx.finite("train_loss")
    if not v or len(v) < 3 or not t or len(t) < 10:
        return []
    t_sm = ema(t.values)
    pairs = [(val, t_sm[max(bisect.bisect_right(t.steps, st) - 1, 0)]) for st, val in zip(v.steps, v.values)]
    below = [1 for val, tr in pairs if tr > 0 and val < 0.9 * tr]
    if len(below) < 0.7 * len(pairs):
        return []
    return [
        Finding(
            "val_below_train",
            "info",
            "Validation loss is consistently lower than training loss",
            [f"`{v.name}` is >10% below `{t.name}` in {len(below)} of {len(pairs)} evaluations"],
            [
                "Usually harmless: dropout/augmentation only run in train mode, and train loss is averaged over an epoch "
                "while val is measured at its end",
                "Sometimes a red flag: the validation set is easier, or overlaps with train (duplicates, leakage)",
            ],
            [
                "Measure loss on a train subset in `model.eval()` mode; if that is also above val, check the split",
                "Check for duplicates between train and val (exact hashes, near-duplicates for images/text)",
            ],
            metric=v.name,
        )
    ]


def check_grad_explosion(ctx: Ctx) -> list[Finding]:
    s = ctx.finite("grad_norm")
    if not s or len(s) < 20:
        return []
    # large gradients in the first steps are normal, judge the rest of the run
    s = ctx.before_blowup(s)
    skip = max(5, len(s) // 20)
    steps, values = s.steps[skip:], s.values[skip:]
    if len(values) < 10:
        return []
    med = statistics.median(values)
    if med <= 0:
        return []
    spikes = [(st, val) for st, val in zip(steps, values) if val > 10 * med]
    if not spikes:
        return []
    worst = max(spikes, key=lambda x: x[1])
    return [
        Finding(
            "grad_spikes",
            "warning",
            f"Gradient norm spikes ({len(spikes)})",
            [
                f"`{s.name}` median is {fmt(med)}, but {len(spikes)} step(s) exceed 10x that; max {fmt(worst[1])} at step {fmt_step(worst[0])}",
                "spike steps: " + ", ".join(fmt_step(st) for st, _ in spikes[:10]) + (" ..." if len(spikes) > 10 else ""),
            ],
            ["Learning rate too high or no warmup", "Outlier batches", "Unstable layers (no normalization, attention logits growing)"],
            [
                "Clip gradients at roughly the median norm: `clip_grad_norm_(..., max_norm=~" + fmt(med * 2) + ")`",
                "Add warmup / lower the peak LR",
                "Inspect the batches at the spike steps",
            ],
            step=worst[0],
            metric=s.name,
        )
    ]


def check_lr_plateau(ctx: Ctx) -> list[Finding]:
    lr = ctx.finite("lr")
    t = ctx.finite("train_loss")
    if not lr or len(lr) < 10 or not t or len(t) < 30:
        return []
    if max(lr.values) == 0 or (max(lr.values) - min(lr.values)) / max(lr.values) > 0.01:
        return []
    sm = ema(t.values)
    tail = sm[int(len(sm) * 0.7) :]
    total = sm[0] - min(sm)
    if total <= 0 or (tail[0] - tail[-1]) > 0.02 * total:
        return []
    if sm[0] > 0 and min(sm) < 0.05 * sm[0]:
        return []  # the loss is already near zero: nothing left for LR decay to win
    return [
        Finding(
            "constant_lr_plateau",
            "info",
            "Loss plateaued with a constant learning rate",
            [
                f"learning rate is constant at {fmt(lr.values[0])} for the whole run",
                f"`{t.name}` improved less than 2% of its total progress over the last 30% of training",
            ],
            ["Without LR decay the optimizer keeps bouncing around the minimum instead of settling into it"],
            [
                "Add a decaying schedule (cosine to ~10% of the peak, or ReduceLROnPlateau)",
                "Or stop earlier: the last 30% of compute bought almost nothing",
            ],
            metric=t.name,
        )
    ]


# ---------------------------------------------------------------- performance


def _util_percent(values: list[float]) -> list[float]:
    return [v * 100 for v in values] if values and max(values) <= 1.0 else values


def check_gpu_util(ctx: Ctx) -> list[Finding]:
    gpus = [s for s in ctx.all_finite("gpu_util") if len(s) >= 10]
    data_t, step_t = ctx.finite("data_time"), ctx.finite("step_time")
    evidence = []
    worst_mean = None
    for s in gpus:
        vals = _util_percent(s.values[len(s) // 10 :])
        mean = statistics.fmean(vals)
        low = sum(1 for x in vals if x < 30) / len(vals)
        if worst_mean is None or mean < worst_mean:
            worst_mean = mean
        if mean < 75:
            evidence.append(f"`{s.name}` averages {mean:.0f}% and is below 30% for {low:.0%} of the time")
    data_share = None
    if data_t and step_t and len(data_t) >= 5 and len(step_t) >= 5:
        d, st = statistics.median(data_t.values), statistics.median(step_t.values)
        if st > 0:
            data_share = d / st
            if data_share > 0.2:
                evidence.append(f"waiting for data takes {data_share:.0%} of each step (`{data_t.name}` / `{step_t.name}`)")
    if not evidence:
        return []
    bad = (worst_mean is not None and worst_mean < 50) or (data_share or 0) > 0.3
    return [
        Finding(
            "input_bound",
            "warning" if bad else "info",
            ("GPU is waiting" if gpus else "Training is waiting") + ": the input pipeline is likely the bottleneck",
            evidence,
            [
                "Too few DataLoader workers for the per-sample work (decoding, augmentation, tokenization)",
                "Host-to-device copies are not overlapped (no pinned memory)",
                "Slow storage or many small files; heavy CPU augmentations",
                "Batch too small to keep the GPU busy, or Python overhead between steps (logging, `.item()` every step)",
            ],
            [
                "Raise `num_workers` (start with the number of CPU cores per GPU), set `pin_memory=True`, "
                "`persistent_workers=True`, `prefetch_factor=4`, and `.to(device, non_blocking=True)`",
                "Pre-process once (resize/decode/tokenize offline) or move augmentations to the GPU",
                "Increase batch size if memory allows; avoid `.item()`/`.cpu()` syncs every step",
                "Confirm with `torch.profiler` or by timing `next(iter(loader))` against the training step",
            ],
            metric=gpus[0].name if gpus else (data_t.name if data_t else None),
        )
    ]


def check_step_time(ctx: Ctx) -> list[Finding]:
    findings = []
    st = ctx.finite("step_time")
    if st and len(st) >= 30:
        vals = st.values[len(st) // 10 :]
        steps = st.steps[len(st) // 10 :]
        med = statistics.median(vals)
        slow = [i for i, v in enumerate(vals) if med > 0 and v > 3 * med]
        if len(slow) >= 3:
            extra = sum(vals[i] - med for i in slow) / sum(vals)
            slow_steps = [steps[i] for i in slow]
            gaps = [b - a for a, b in zip(slow_steps, slow_steps[1:])]
            period = statistics.mode(gaps) if gaps else 0
            on_grid = sum(1 for x in slow_steps if period and abs((x - slow_steps[0]) / period - round((x - slow_steps[0]) / period)) < 0.05)
            periodic = period > 1 and len(slow) >= 3 and on_grid >= 0.7 * len(slow)
            evidence = [f"{len(slow)} steps take >3x the median `{st.name}` ({fmt(med)}); they cost {extra:.0%} of total time"]
            if periodic:
                evidence.append(f"slow steps repeat every {fmt_step(period)} steps")
            findings.append(
                Finding(
                    "slow_steps",
                    "warning" if extra > 0.15 else "info",
                    "Periodic slow steps" if periodic else "Slow outlier steps",
                    evidence,
                    (["Something runs every N steps inside the training loop: evaluation, checkpoint saving, heavy logging"] if periodic else [])
                    + [
                        "DataLoader stalls when a worker hits slow samples or a new epoch spins up workers",
                        "Python garbage collection or CUDA allocator re-organizing memory",
                    ],
                    [
                        "Save checkpoints asynchronously or less often; run evaluation less frequently or on a subset",
                        "Use `persistent_workers=True` so workers are not recreated every epoch",
                        "Log scalars every N steps and avoid image/histogram logging in the hot loop",
                    ],
                    metric=st.name,
                )
            )
    tp = ctx.finite("throughput")
    if tp and len(tp) >= 20:
        q = len(tp) // 4
        first, last = statistics.median(tp.values[q // 2 : q + 1]), statistics.median(tp.values[-q:])
        if first > 0 and last < 0.8 * first and spearman(tp.steps, tp.values) < -0.6:
            findings.append(
                Finding(
                    "slowdown",
                    "warning",
                    "Training gets slower over time",
                    [f"`{tp.name}` dropped from ~{fmt(first)} to ~{fmt(last)} ({1 - last / first:.0%} slower)"],
                    [
                        "Memory growth causing allocator churn or swapping (see memory checks)",
                        "Growing sequence lengths or a curriculum that makes batches heavier",
                        "Thermal throttling or another process competing for the GPU",
                    ],
                    [
                        "Check memory curves for a leak; check `nvidia-smi -q -d PERFORMANCE` for throttling",
                        "If the data gets heavier by design, this is expected",
                    ],
                    metric=tp.name,
                )
            )
    return findings


def check_memory_growth(ctx: Ctx) -> list[Finding]:
    findings = []
    for role, where in (("gpu_mem", "GPU"), ("host_mem", "host")):
        for s in ctx.all_finite(role):
            if len(s) < 10:
                continue
            vals, steps = s.values[len(s) // 10 :], s.steps[len(s) // 10 :]
            q = max(1, len(vals) // 4)
            first, last = statistics.median(vals[:q]), statistics.median(vals[-q:])
            if first <= 0 or last < 1.2 * first or spearman(steps, vals) < 0.8:
                continue
            causes = (
                [
                    "Tensors that keep the autograd graph alive are stored across steps (`losses.append(loss)` instead of `loss.item()`)",
                    "Predictions/embeddings accumulated on the GPU during evaluation",
                    "Growing caches or variable input shapes fragmenting the CUDA allocator (not a true leak)",
                ]
                if role == "gpu_mem"
                else [
                    "DataLoader workers touching Python lists/dicts in the Dataset: refcount updates copy pages (copy-on-read)",
                    "Metrics, predictions or logs accumulated in Python lists for the whole run",
                    "A cache in the Dataset that never evicts",
                ]
            )
            fixes = (
                [
                    "Store `loss.item()` / `.detach().cpu()` values, run evaluation under `torch.no_grad()`",
                    "Move accumulated outputs to CPU every step; print `torch.cuda.memory_summary()` at two points to compare",
                ]
                if role == "gpu_mem"
                else [
                    "Keep Dataset metadata in numpy arrays or Arrow instead of Python lists of objects",
                    "Bound caches (`functools.lru_cache(maxsize=...)`), write long-running metrics to disk",
                    "Track memory with `tracemalloc` or `memray` to find what grows",
                ]
            )
            findings.append(
                Finding(
                    "memory_growth",
                    "warning",
                    f"{where} memory keeps growing",
                    [f"`{s.name}` grew from ~{fmt(first)} to ~{fmt(last)} (+{last / first - 1:.0%}) and rises steadily through the run"],
                    causes,
                    fixes,
                    metric=s.name,
                )
            )
    return findings


def check_log_patterns(ctx: Ctx) -> list[Finding]:
    groups = defaultdict(list)
    for e in ctx.run.log_lines:
        p = logpatterns.match(e.text)
        if p:
            groups[p.id].append(e)
    out = []
    by_id = {p.id: p for p in logpatterns.PATTERNS}
    for pid, events in groups.items():
        p = by_id[pid]
        evidence = [f"{len(events)} matching line(s) in the log, first at line {events[0].line_no}:"]
        evidence += [f"  {e.text}" for e in events[:3]]
        out.append(Finding(f"log:{pid}", p.severity, p.title, evidence, list(p.causes), list(p.fixes)))
    return out


ALL_CHECKS = [
    check_log_patterns,
    check_non_finite,
    check_divergence,
    check_no_learning,
    check_overfitting,
    check_spikes,
    check_grad_explosion,
    check_val_below_train,
    check_lr_plateau,
    check_gpu_util,
    check_step_time,
    check_memory_growth,
]


def run_checks(run: Run, roles: dict[str, list[str]]) -> list[Finding]:
    ctx = Ctx(run, roles)
    findings = [f for check in ALL_CHECKS for f in check(ctx)]
    return sorted(findings, key=lambda f: SEVERITY_ORDER[f.severity])
