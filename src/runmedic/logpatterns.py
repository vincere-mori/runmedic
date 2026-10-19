"""Known failure messages that show up in training stdout/stderr."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Pattern:
    id: str
    severity: str
    regex: re.Pattern
    title: str
    causes: tuple[str, ...]
    fixes: tuple[str, ...]


def _p(id, severity, regex, title, causes, fixes):
    return Pattern(id, severity, re.compile(regex, re.IGNORECASE), title, tuple(causes), tuple(fixes))


PATTERNS = [
    _p(
        "cuda_oom",
        "critical",
        r"CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED",
        "GPU ran out of memory",
        [
            "Batch size or sequence length too large for the GPU",
            "Activations kept alive between steps (e.g. appending `loss` instead of `loss.item()` to a list)",
            "Memory fragmentation after variable-size batches",
        ],
        [
            "Lower the batch size and keep the effective batch with gradient accumulation",
            "Enable mixed precision (`torch.autocast(..., dtype=torch.bfloat16)`) and gradient checkpointing",
            "Log `loss.item()`, not the tensor; wrap evaluation in `torch.no_grad()`",
            "Try `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` against fragmentation",
        ],
    ),
    _p(
        "dataloader_killed",
        "critical",
        r"DataLoader worker \(pid.*\) (is killed|exited unexpectedly)|Bus error|insufficient shared memory|/dev/shm",
        "DataLoader worker crashed",
        [
            "Not enough shared memory (/dev/shm), common inside Docker",
            "Worker killed by the OS out-of-memory killer (too many workers or heavy samples)",
        ],
        [
            "In Docker run with `--shm-size=8g` or `--ipc=host`",
            "Reduce `num_workers` or the per-sample memory; avoid holding big Python objects in the Dataset",
            "Check `dmesg` for OOM killer messages",
        ],
    ),
    _p(
        "amp_overflow",
        "warning",
        r"Gradient overflow|Skipping step, loss scaler|found inf|loss scale.*(reduc|decreas)",
        "Mixed precision is overflowing and skipping steps",
        ["fp16 range is too small for these activations or gradients", "Learning rate too high, so gradients blow up"],
        [
            "Switch to bf16 if the GPU supports it (Ampere and newer)",
            "Lower the learning rate or add warmup; clip gradients after `scaler.unscale_(optimizer)`",
            "Keep losses and softmax in fp32 (autocast does this for most ops)",
        ],
    ),
    _p(
        "anomaly_nan",
        "critical",
        r"(?:Function '\w+' returned nan|detect_anomaly|nan values? (?:in|found)|loss is nan)",
        "NaN detected during training",
        ["Exploding gradients, log/sqrt/division of zero, or fp16 overflow"],
        [
            "See the NaN section of the report for the step and what preceded it",
            "Run a few steps with `torch.autograd.set_detect_anomaly(True)` to find the op",
        ],
    ),
    _p(
        "nccl_timeout",
        "critical",
        r"NCCL.*(timeout|timed out|Watchdog)|ProcessGroupNCCL.*(error|timeout)",
        "Distributed training hung or timed out in NCCL",
        [
            "Ranks diverged: one rank skipped a collective (different number of batches, a rank-only `if`)",
            "One rank crashed or is much slower (data loading, evaluation only on rank 0 without a barrier)",
        ],
        [
            "Make sure every rank sees the same number of batches (`DistributedSampler`, `drop_last=True`)",
            "Run evaluation and checkpoint saving the same way on all ranks or add `dist.barrier()`",
            "Set `NCCL_DEBUG=INFO` and `TORCH_DISTRIBUTED_DEBUG=DETAIL` to see which rank stalls",
        ],
    ),
    _p(
        "device_mismatch",
        "critical",
        r"Expected all tensors to be on the same device",
        "Tensors on different devices",
        ["A tensor created inside forward/loss without `device=`, or a buffer not registered on the module"],
        ["Create tensors with `device=x.device`; register constants with `self.register_buffer`"],
    ),
    _p(
        "backward_twice",
        "critical",
        r"Trying to backward through the graph a second time",
        "Backward through a freed graph",
        ["A tensor from the previous step (hidden state, running loss) is reused in the next step's graph"],
        ["`.detach()` recurrent state between steps; accumulate `loss.item()` for logging"],
    ),
    _p(
        "inplace_grad",
        "critical",
        r"modified by an inplace operation",
        "In-place operation broke autograd",
        ["An in-place op (`+=`, `relu_`, `x[...] = ...`) changed a tensor needed for backward"],
        ["Replace the in-place op with an out-of-place one; the error message names the tensor version"],
    ),
    _p(
        "unused_params",
        "warning",
        r"Expected to have finished reduction in the prior iteration|find_unused_parameters",
        "DDP found parameters without gradients",
        ["Some parameters are not used in forward on some steps (conditional branches, frozen heads)"],
        ["Freeze them with `requires_grad_(False)` or pass `find_unused_parameters=True` (slower)"],
    ),
]


def match(line: str) -> Pattern | None:
    for p in PATTERNS:
        if p.regex.search(line):
            return p
    return None
