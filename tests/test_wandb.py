import sys
import types

from runmedic import loaders
from runmedic.loaders import wandb as wb


def test_parse_ref():
    assert wb.parse_ref("https://wandb.ai/me/proj/runs/abc123?nw=1") == "me/proj/abc123"
    assert wb.parse_ref("wandb:me/proj/abc123") == "me/proj/abc123"
    assert wb.parse_ref("runs/exp1") is None


def test_load_through_api(monkeypatch):
    class FakeRun:
        name = "fancy-run"

        def scan_history(self):
            yield {"_step": 0, "train/loss": 1.0, "_runtime": 1.0, "flag": True, "text": "x"}
            yield {"_step": 1, "train/loss": 0.5}

        def history(self, stream, pandas):
            assert stream == "system" and pandas is False
            return [{"_runtime": 10.0, "system.gpu.0.gpu": 40.0}, {"_runtime": 20.0, "system.gpu.0.gpu": 42.0}]

    fake = types.SimpleNamespace(Api=lambda: types.SimpleNamespace(run=lambda path: FakeRun()))
    monkeypatch.setitem(sys.modules, "wandb", fake)
    (run,) = loaders.load("https://wandb.ai/me/proj/runs/abc123")
    assert run.name == "fancy-run"
    assert run.series["train/loss"].values == [1.0, 0.5]
    assert run.series["system.gpu.0.gpu"].steps == [10.0, 20.0]
    assert "flag" not in run.series and "text" not in run.series
