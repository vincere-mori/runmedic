import json

import pytest

from runmedic.cli import main


@pytest.fixture
def nan_log(tmp_path):
    lines = [f"step {i} loss {2.0 - i * 0.01:.4f} lr 0.001" for i in range(100)] + ["step 100 loss nan lr 0.001"]
    f = tmp_path / "train.log"
    f.write_text("\n".join(lines))
    return f


def test_text_report(nan_log, capsys):
    assert main(["diagnose", str(nan_log)]) == 0
    out = capsys.readouterr().out
    assert "[CRITICAL]" in out and "step 100" in out


def test_json_report(nan_log, capsys):
    main(["diagnose", str(nan_log), "--json"])
    data = json.loads(capsys.readouterr().out)
    assert data[0]["roles"]["train_loss"] == ["loss"]
    assert data[0]["findings"][0]["check"] == "non_finite"
    assert data[0]["findings"][0]["step"] == 100


def test_fail_on(nan_log, capsys):
    assert main(["diagnose", str(nan_log), "--fail-on", "critical"]) == 1


def test_healthy_run_passes_fail_on(tmp_path, capsys):
    f = tmp_path / "ok.log"
    f.write_text("\n".join(f"step {i} loss {1 / (i + 1):.5f}" for i in range(100)))
    assert main(["diagnose", str(f), "--fail-on", "warning"]) == 0
    assert "No problems found" in capsys.readouterr().out


def test_override_and_series(nan_log, capsys):
    main(["series", str(nan_log)])
    assert "train_loss" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["diagnose", str(nan_log), "--train-loss", "nope"])


def test_empty_dir(tmp_path):
    with pytest.raises(SystemExit):
        main(["diagnose", str(tmp_path)])
