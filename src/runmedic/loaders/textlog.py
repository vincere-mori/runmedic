"""Plain training stdout: `step 100 loss=0.52`, HF Trainer dicts, Lightning progress bars, tracebacks."""

from __future__ import annotations

import re
from pathlib import Path

from runmedic import logpatterns
from runmedic.model import LogEvent, Run

NUM = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?|[-+]?(?:nan|inf(?:inity)?)"
# "loss=0.5", "'loss': 0.5", "Loss: 0.5", "loss 0.5", and two-word keys like "val loss 2.0" (nanoGPT)
KV = re.compile(
    rf"(?P<key>(?:(?:train|training|val|valid|validation|eval|test)\s+)?[A-Za-z_][\w./-]*)['\"]?(?:\s*[:=]\s*|\s+)['\"]?(?P<val>{NUM})(?![\w.])",
    re.IGNORECASE,
)
STEP = re.compile(r"\b(?:global_step|step|iter(?:ation)?)\b['\"]?\s*[:=]?\s*(\d+)", re.IGNORECASE)
METRIC_HINT = re.compile(
    r"loss|acc|lr\b|learning_rate|grad_norm|gnorm|ppl|perplexity|f1|auc|mae|mse|rmse|bleu|rouge|"
    r"gpu|mem|samples_per_sec|tokens_per_sec|throughput|it_per_sec|step_time|batch_time|data_time",
    re.IGNORECASE,
)
STEP_NAMES = {"step", "global_step", "iter", "iteration", "epoch", "batch"}


def parse_lines(lines, run: Run) -> None:
    current_step: float | None = None
    for no, line in enumerate(lines, 1):
        if logpatterns.match(line):
            run.log_lines.append(LogEvent(no, line.strip()[:300]))
        m = STEP.search(line)
        line_step = float(m.group(1)) if m else None
        if line_step is not None:
            current_step = line_step
        for kv in KV.finditer(line):
            key = re.sub(r"\s+", "_", kv.group("key").strip("'\"")).lower()
            if key in STEP_NAMES or not METRIC_HINT.search(key):
                continue
            series = run.get(key)
            step = line_step if line_step is not None else current_step
            # no step on this line (HF dicts, progress bars): continue the series after its last point,
            # an eval line printed right after "step N" still lands on N
            if line_step is None and (step is None or (series.steps and series.steps[-1] >= step)):
                step = series.steps[-1] + 1 if series.steps else 0.0
            series.add(step, float(kv.group("val")))


def load(path: Path) -> Run:
    run = Run(name=path.stem, source="text", path=str(path))
    text = path.read_text(encoding="utf-8", errors="replace")
    parse_lines(re.split(r"\r\n|\r|\n", text), run)
    return run.finalize()
