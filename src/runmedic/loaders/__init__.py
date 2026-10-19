"""Turns a path (or a W&B reference) into runs with metric series, whatever logger produced them."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from runmedic.loaders import mlflow, tabular, tensorboard, textlog, wandb
from runmedic.model import Run

TEXT_SUFFIXES = {".log", ".txt", ".out", ".err"}
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", "wandb"}


def _is_mlflow_db(path: Path) -> bool:
    try:
        con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        try:
            tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        finally:
            con.close()
    except sqlite3.DatabaseError:
        return False
    return {"metrics", "runs"} <= tables


def load_file(path: Path) -> list[Run]:
    name = path.name
    if name.startswith("events.out.tfevents"):
        run = Run(name=path.parent.name, source="tensorboard", path=str(path))
        tensorboard.read_events(path, run)
        return [run.finalize()]
    if path.suffix in {".db", ".sqlite", ".sqlite3"} and _is_mlflow_db(path):
        return mlflow.load_sqlite(path)
    if path.suffix == ".csv":
        return [tabular.load_csv(path)]
    if path.suffix == ".jsonl":
        return [tabular.load_jsonl(path)]
    if path.suffix == ".json":
        run = tabular.load_json(path)
        return [run] if run else []
    return [textlog.load(path)]


def _walk(root: Path):
    for p in root.rglob("*"):
        if not SKIP_DIRS.intersection(p.relative_to(root).parts):
            yield p


def load_dir(root: Path) -> list[Run]:
    runs: list[Run] = []
    files = [p for p in _walk(root) if p.is_file()]

    for d in sorted({f.parent for f in files if f.name.startswith("events.out.tfevents")}):
        run = tensorboard.load_dir(d)
        run.name = d.relative_to(root).as_posix() if d != root else root.name
        runs.append(run)

    for d in mlflow.find_run_dirs(root):
        runs.append(mlflow.load_run_dir(d))
    for f in files:
        if f.suffix in {".db", ".sqlite", ".sqlite3"} and _is_mlflow_db(f):
            runs += mlflow.load_sqlite(f)

    # HF Trainer copies trainer_state.json into every checkpoint; the longest one has the full history
    states = [f for f in files if f.name == "trainer_state.json"]
    if states:
        best = max(states, key=lambda f: f.stat().st_size)
        run = tabular.load_json(best)
        if run:
            # the Trainer also mirrors the same metrics to TensorBoard under <output_dir>/runs/
            hf_tb = best.parent.parent / "runs"
            runs = [r for r in runs if not (r.source == "tensorboard" and hf_tb in Path(r.path).parents)]
            runs.append(run)

    # a stdout log next to TensorBoard/MLflow data usually repeats the same metrics: keep the structured
    # series and only take the error lines (OOM, NCCL, ...) from the text log
    structured = {Path(r.path): r for r in runs if r.series and Path(r.path).is_dir()}
    for f in files:
        owner = structured.get(f.parent)
        if owner and f.suffix in TEXT_SUFFIXES:
            owner.log_lines += textlog.load(f).log_lines
        elif not owner and f.suffix in {".csv", ".jsonl"} | TEXT_SUFFIXES:
            runs += load_file(f)
    return [r for r in runs if r.series or r.log_lines]


def load(target: str) -> list[Run]:
    ref = wandb.parse_ref(target)
    if ref:
        return [wandb.load(ref)]
    path = Path(target)
    if not path.exists():
        raise FileNotFoundError(target)
    runs = load_dir(path) if path.is_dir() else load_file(path)
    return [r for r in runs if r.series or r.log_lines]
