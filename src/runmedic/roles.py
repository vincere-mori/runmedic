"""Figures out which series is the train loss, the val loss, the learning rate and so on, from their names."""

from __future__ import annotations

import re

from runmedic.model import Run

VAL = re.compile(r"(^|[^a-z])(val|valid|validation|eval|evaluation|test|dev|holdout)([^a-z]|$)", re.I)
LOSS = re.compile(r"loss", re.I)
PRIMARY_LOSS = {
    "loss",
    "train_loss",
    "train/loss",
    "loss/train",
    "training_loss",
    "train/loss_epoch",
    "train_loss_epoch",
    "train/loss_step",
    "train_loss_step",
    "trainer/loss",
}
PRIMARY_VAL = {
    "val_loss",
    "val/loss",
    "loss/val",
    "eval_loss",
    "eval/loss",
    "validation_loss",
    "valid_loss",
    "valid/loss",
    "test_loss",
    "val_loss_epoch",
}

ROLE_PATTERNS = {
    "lr": re.compile(r"(^|[^a-z])(lr|learning_rate|learning-rate)([^a-z]|$)", re.I),
    "grad_norm": re.compile(r"grad(ient)?[_/.]?norm|gnorm|grad_l2", re.I),
    "gpu_util": re.compile(r"gpu.*util|^system\.gpu\.\d+\.gpu$|gpu_\d+_utilization", re.I),
    "gpu_mem": re.compile(r"(gpu|cuda).*(mem|memory|allocated)|memory_allocated|max_memory_allocated", re.I),
    "host_mem": re.compile(r"system\.proc\.memory\.rss|system_memory_usage|cpu.*mem|host.*mem|rss", re.I),
    "step_time": re.compile(r"step_time|iter_time|batch_time|time[/_]step|sec(onds)?_per_(step|iter)|time_per_(step|iter)", re.I),
    "data_time": re.compile(r"data_time|data_load|dataloader_time|data_wait", re.I),
    "throughput": re.compile(r"samples?_per_s|samples?/s|tokens?_per_s|tokens?/s|it_per_s|throughput", re.I),
}


def _pick(names: list[str], primary: set[str], run: Run) -> str | None:
    if not names:
        return None
    for n in names:
        if n.lower() in primary:
            return n
    # prefer the generic name over components like loss_cls / loss_bbox, then the longest series
    return min(names, key=lambda n: (n.lower().count("_") + n.count("/"), -len(run.series[n])))


def detect(run: Run, overrides: dict[str, str] | None = None) -> dict[str, list[str]]:
    roles: dict[str, list[str]] = {}
    losses = [n for n in run.series if LOSS.search(n)]
    val = [n for n in losses if VAL.search(n)]
    train = [n for n in losses if n not in val]

    t = _pick(train, PRIMARY_LOSS, run)
    v = _pick(val, PRIMARY_VAL, run)
    if t:
        roles["train_loss"] = [t]
    if v:
        roles["val_loss"] = [v]

    for role, pattern in ROLE_PATTERNS.items():
        hits = [n for n in run.series if pattern.search(n) and n not in losses]
        if role == "host_mem":
            hits = [n for n in hits if n not in roles.get("gpu_mem", [])]
        if role in ("throughput", "step_time"):
            hits = [n for n in hits if not VAL.search(n)]  # eval speed says nothing about the training loop
        if hits:
            roles[role] = hits

    for role, name in (overrides or {}).items():
        if name not in run.series:
            raise KeyError(f"series '{name}' not found in run '{run.name}'")
        roles[role] = [name]
    return roles
