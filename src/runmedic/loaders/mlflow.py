"""MLflow runs, read straight from the tracking store: the `mlruns/` file store or a sqlite `mlflow.db`."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from runmedic.model import Run


def _run_name(run_dir: Path) -> str:
    tag = run_dir / "tags" / "mlflow.runName"
    if tag.is_file():
        return tag.read_text(encoding="utf-8").strip() or run_dir.name
    return run_dir.name


def load_run_dir(run_dir: Path) -> Run:
    """A file-store run directory: <mlruns>/<experiment_id>/<run_id>/metrics/<key> with 'timestamp value step' lines."""
    run = Run(name=_run_name(run_dir), source="mlflow", path=str(run_dir))
    metrics = run_dir / "metrics"
    for f in sorted(p for p in metrics.rglob("*") if p.is_file()):
        key = f.relative_to(metrics).as_posix()  # keys with '/' are stored as subfolders
        for line in f.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            step = float(parts[2]) if len(parts) > 2 else float(len(run.get(key)))
            run.get(key).add(step, float(parts[1]))
    return run.finalize()


def find_run_dirs(root: Path) -> list[Path]:
    return sorted(p.parent for p in root.rglob("metrics") if p.is_dir() and (p.parent / "meta.yaml").is_file())


def load_sqlite(db: Path) -> list[Run]:
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        names = dict(con.execute("SELECT run_uuid, name FROM runs").fetchall())
        rows = con.execute("SELECT run_uuid, key, value, step, is_nan FROM metrics ORDER BY run_uuid, step, timestamp")
        runs: dict[str, Run] = {}
        for run_id, key, value, step, is_nan in rows:
            if run_id not in runs:
                runs[run_id] = Run(name=names.get(run_id) or run_id, source="mlflow", path=f"{db}#{run_id}")
            runs[run_id].get(key).add(float(step), float("nan") if is_nan else float(value))
    finally:
        con.close()
    return [r.finalize() for r in runs.values()]
