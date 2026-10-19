from __future__ import annotations

import math
import statistics


def ema(values: list[float], span: int | None = None) -> list[float]:
    """Exponential moving average; the span adapts to the series length so short runs are not over-smoothed."""
    if not values:
        return []
    span = span or max(2, min(50, len(values) // 10))
    alpha = 2 / (span + 1)
    out, acc = [], values[0]
    for v in values:
        acc = alpha * v + (1 - alpha) * acc
        out.append(acc)
    return out


def mad(values: list[float]) -> float:
    med = statistics.median(values)
    return statistics.median(abs(v - med) for v in values)


def robust_z(values: list[float]) -> list[float]:
    med = statistics.median(values)
    scale = 1.4826 * mad(values)
    if scale == 0:
        scale = statistics.pstdev(values) or 1.0
    return [(v - med) / scale for v in values]


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    for r, i in enumerate(order):
        ranks[i] = float(r)
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return 0.0
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    vy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return cov / (vx * vy) if vx and vy else 0.0


def fmt(v: float) -> str:
    if not math.isfinite(v):
        return str(v)
    if v == 0:
        return "0"
    if abs(v) >= 1000 or abs(v) < 1e-3:
        return f"{v:.2e}"
    return f"{v:.4g}"


def fmt_step(s: float) -> str:
    return str(int(s)) if float(s).is_integer() else f"{s:.2f}"
