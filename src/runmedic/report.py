from __future__ import annotations

import json
import math

from runmedic.model import Finding, Run
from runmedic.stats import fmt, fmt_step

LABEL = {"critical": "CRITICAL", "warning": "WARNING", "info": "INFO"}
ROLE_TITLES = {
    "train_loss": "train loss",
    "val_loss": "val loss",
    "lr": "learning rate",
    "grad_norm": "grad norm",
    "gpu_util": "GPU util",
    "gpu_mem": "GPU memory",
    "host_mem": "host memory",
    "step_time": "step time",
    "data_time": "data time",
    "throughput": "throughput",
}


def _series_summary(run: Run) -> dict:
    out = {}
    for name, s in run.series.items():
        last = s.values[-1]
        out[name] = {"points": len(s), "first_step": s.steps[0], "last_step": s.steps[-1], "last_value": last if math.isfinite(last) else str(last)}
    return out


def to_dict(run: Run, roles: dict[str, list[str]], findings: list[Finding]) -> dict:
    return {
        "run": run.name,
        "source": run.source,
        "path": run.path,
        "roles": roles,
        "series": _series_summary(run),
        "findings": [f.to_dict() for f in findings],
    }


def to_json(results: list[tuple[Run, dict, list[Finding]]]) -> str:
    return json.dumps([to_dict(*r) for r in results], indent=2, ensure_ascii=False)


def to_text(run: Run, roles: dict[str, list[str]], findings: list[Finding]) -> str:
    lines = [f"runmedic · {run.name}  ({run.source}, {len(run.series)} series)", f"  {run.path}", ""]
    if roles:
        lines.append("Detected:")
        for role, names in roles.items():
            lines.append(f"  {ROLE_TITLES.get(role, role):<14} {', '.join(names)}")
        lines.append("")
    if "train_loss" not in roles:
        lines += ["No training loss series found. Pass it explicitly with --train-loss NAME (see `runmedic series PATH`).", ""]

    if not findings:
        lines.append("No problems found. The curves look healthy for the checks runmedic knows about.")
        return "\n".join(lines)

    counts = {sev: sum(1 for f in findings if f.severity == sev) for sev in LABEL}
    lines.append("Found: " + ", ".join(f"{n} {sev}" for sev, n in counts.items() if n))
    for i, f in enumerate(findings, 1):
        lines += ["", f"{i}. [{LABEL[f.severity]}] {f.title}"]
        lines += [f"   · {e}" for e in f.evidence]
        if f.causes:
            lines.append("   Likely causes:")
            lines += [f"     - {c}" for c in f.causes]
        if f.fixes:
            lines.append("   What to try:")
            lines += [f"     {j}. {x}" for j, x in enumerate(f.fixes, 1)]
    return "\n".join(lines)


def series_table(run: Run, roles: dict[str, list[str]]) -> str:
    role_of = {n: r for r, names in roles.items() for n in names}
    width = max((len(n) for n in run.series), default=10)
    lines = [f"{run.name}  ({run.source})", f"  {'series':<{width}}  {'points':>6}  {'steps':>17}  {'last':>10}  role"]
    for name, s in run.series.items():
        span = f"{fmt_step(s.steps[0])}..{fmt_step(s.steps[-1])}"
        lines.append(f"  {name:<{width}}  {len(s):>6}  {span:>17}  {fmt(s.values[-1]):>10}  {role_of.get(name, '')}")
    return "\n".join(lines)
