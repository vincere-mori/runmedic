"""Weights & Biases runs through the public API. Needs `pip install runmedic[wandb]` and `wandb login`."""

from __future__ import annotations

import math
import re

from runmedic.model import Run

RUN_URL = re.compile(r"wandb\.ai/(?P<entity>[^/]+)/(?P<project>[^/]+)/runs/(?P<run>[^/?#]+)")
RUN_PATH = re.compile(r"^wandb:(?P<path>[^/\s]+/[^/\s]+/[^/\s]+)$")


def parse_ref(ref: str) -> str | None:
    """Accepts a run URL or `wandb:entity/project/run_id`, returns `entity/project/run_id`."""
    m = RUN_URL.search(ref)
    if m:
        return f"{m['entity']}/{m['project']}/{m['run']}"
    m = RUN_PATH.match(ref)
    return m["path"] if m else None


def load(run_path: str) -> Run:
    try:
        import wandb
    except ImportError as e:
        raise SystemExit("W&B support needs the wandb package: pip install 'runmedic[wandb]'") from e

    api_run = wandb.Api().run(run_path)
    run = Run(name=api_run.name or run_path, source="wandb", path=run_path)
    for row in api_run.scan_history():
        step = row.get("_step")
        for key, val in row.items():
            if key.startswith("_") or isinstance(val, bool) or not isinstance(val, (int, float)):
                continue
            run.get(key).add(float(step if step is not None else len(run.get(key))), float(val))
    # system metrics (GPU utilization, memory) live in a separate stream
    for row in api_run.history(stream="system", pandas=False):
        t = row.get("_runtime")
        for key, val in row.items():
            if key.startswith("system.") and isinstance(val, (int, float)) and t is not None and math.isfinite(t):
                run.get(key).add(float(t), float(val))
    return run.finalize()
