import math
import random

from runmedic import roles
from runmedic.checks import run_checks
from runmedic.model import Run


def make_run(**series) -> Run:
    """make_run(train_loss=[...], lr=[...]) -> Run with steps 0..n-1, or pass (steps, values) tuples."""
    run = Run(name="test", source="test", path="-")
    for name, data in series.items():
        steps, values = data if isinstance(data, tuple) else (list(range(len(data))), data)
        for s, v in zip(steps, values):
            run.get(name).add(s, v)
    return run.finalize()


def findings(run: Run) -> dict:
    return {f.check: f for f in run_checks(run, roles.detect(run))}


def healthy_loss(n=300, start=2.5, end=0.3, noise=0.03, seed=0):
    rng = random.Random(seed)
    return [end + (start - end) * math.exp(-5 * i / n) + rng.gauss(0, noise) for i in range(n)]
