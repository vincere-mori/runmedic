"""Table-like logs: CSV (Lightning CSVLogger, W&B/MLflow exports), JSON Lines, HF Trainer `trainer_state.json`."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from runmedic.model import Run

STEP_KEYS = ("step", "global_step", "_step", "iteration", "iter", "trainer/global_step")
SKIP_KEYS = {"epoch", "_runtime", "_timestamp", "timestamp", "time", "wall_time", "datetime", "date", "created_at"}


def _number(v) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        v = v.strip()
        if not v:
            return None
        try:
            return float(v)  # handles 'nan', 'inf', '1e-3'
        except ValueError:
            return None
    return None


def _add_rows(run: Run, rows) -> None:
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        step = None
        for k in STEP_KEYS:
            if k in row:
                step = _number(row[k])
                if step is not None:
                    break
        if step is None:
            step = float(i)
        for key, raw in row.items():
            if key in STEP_KEYS or key.lower() in SKIP_KEYS:
                continue
            val = _number(raw)
            if val is not None:
                run.get(key).add(step, val)


def load_csv(path: Path) -> Run:
    run = Run(name=path.stem, source="csv", path=str(path))
    with path.open(encoding="utf-8", newline="") as f:
        _add_rows(run, csv.DictReader(f))
    return run.finalize()


def load_jsonl(path: Path) -> Run:
    run = Run(name=path.stem, source="jsonl", path=str(path))
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    _add_rows(run, rows)
    return run.finalize()


def load_json(path: Path) -> Run | None:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("log_history"), list):
        # HF Trainer: train rows have 'loss', eval rows have 'eval_loss'
        run = Run(name=path.parent.name, source="hf-trainer", path=str(path))
        # the final summary row (train_runtime, average train_loss) is not a point on the curve
        rows = [r for r in data["log_history"] if "train_runtime" not in r]
        rows = [{("train_loss" if k == "loss" else k): v for k, v in r.items()} for r in rows]
        _add_rows(run, rows)
        return run.finalize()
    if isinstance(data, list) and data and isinstance(data[0], dict):
        run = Run(name=path.stem, source="json", path=str(path))
        _add_rows(run, data)
        return run.finalize()
    return None
