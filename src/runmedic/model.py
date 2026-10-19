from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Series:
    name: str
    steps: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    def add(self, step: float, value: float) -> None:
        self.steps.append(step)
        self.values.append(value)

    def sort(self) -> None:
        pairs = sorted(zip(self.steps, self.values), key=lambda p: p[0])
        self.steps = [p[0] for p in pairs]
        self.values = [p[1] for p in pairs]

    def finite(self) -> Series:
        out = Series(self.name)
        for s, v in zip(self.steps, self.values):
            if math.isfinite(v):
                out.add(s, v)
        return out

    def __len__(self) -> int:
        return len(self.values)


@dataclass
class LogEvent:
    line_no: int
    text: str


@dataclass
class Run:
    name: str
    source: str  # tensorboard, mlflow, csv, ...
    path: str
    series: dict[str, Series] = field(default_factory=dict)
    log_lines: list[LogEvent] = field(default_factory=list)

    def get(self, name: str) -> Series:
        if name not in self.series:
            self.series[name] = Series(name)
        return self.series[name]

    def finalize(self) -> Run:
        self.series = {k: s for k, s in self.series.items() if len(s)}
        for s in self.series.values():
            s.sort()
        return self


SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}


@dataclass
class Finding:
    check: str
    severity: str  # critical | warning | info
    title: str
    evidence: list[str]
    causes: list[str]
    fixes: list[str]
    step: float | None = None
    metric: str | None = None

    def to_dict(self) -> dict:
        return {
            "check": self.check,
            "severity": self.severity,
            "title": self.title,
            "metric": self.metric,
            "step": self.step,
            "evidence": self.evidence,
            "likely_causes": self.causes,
            "fixes": self.fixes,
        }
