from __future__ import annotations

import argparse
import sys

from runmedic import __version__, loaders, report, roles
from runmedic.checks import run_checks
from runmedic.model import SEVERITY_ORDER

ROLE_FLAGS = ["train_loss", "val_loss", "lr", "grad_norm", "gpu_util", "step_time"]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="runmedic",
        description="Diagnose training runs from their logs: TensorBoard, MLflow, W&B, CSV/JSONL, HF Trainer, plain stdout.",
    )
    p.add_argument("--version", action="version", version=f"runmedic {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("diagnose", help="find problems in a run and explain how to fix them")
    d.add_argument("path", help="log dir, log file, mlflow.db, or a W&B run (URL or wandb:entity/project/run_id)")
    d.add_argument("--json", action="store_true", help="machine-readable output (for agents and CI)")
    d.add_argument("--run", help="only runs whose name contains this text")
    d.add_argument("--fail-on", choices=["critical", "warning", "info"], help="exit with code 1 if a finding of this severity or worse is found")
    for role in ROLE_FLAGS:
        d.add_argument(f"--{role.replace('_', '-')}", dest=role, metavar="NAME", help=f"series to use as {role.replace('_', ' ')}")

    s = sub.add_parser("series", help="list the series runmedic found and how it interpreted them")
    s.add_argument("path")
    s.add_argument("--run")
    return p


def _load(path: str, only: str | None):
    try:
        runs = loaders.load(path)
    except FileNotFoundError:
        sys.exit(f"runmedic: no such file or directory: {path}")
    if only:
        runs = [r for r in runs if only in r.name]
    if not runs:
        sys.exit(f"runmedic: no training metrics or logs found in {path}")
    return runs


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    runs = _load(args.path, args.run)

    if args.cmd == "series":
        print("\n\n".join(report.series_table(r, roles.detect(r)) for r in runs))
        return 0

    overrides = {role: getattr(args, role) for role in ROLE_FLAGS if getattr(args, role)}
    results = []
    for run in runs:
        try:
            run_roles = roles.detect(run, overrides)
        except KeyError as e:
            sys.exit(f"runmedic: {e.args[0]}")
        results.append((run, run_roles, run_checks(run, run_roles)))

    if args.json:
        print(report.to_json(results))
    else:
        sep = "\n\n" + "─" * 72 + "\n\n"
        print(sep.join(report.to_text(*r) for r in results))

    if args.fail_on:
        limit = SEVERITY_ORDER[args.fail_on]
        if any(SEVERITY_ORDER[f.severity] <= limit for _, _, fs in results for f in fs):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
