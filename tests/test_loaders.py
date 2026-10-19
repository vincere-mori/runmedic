import json
import math
import sqlite3
import struct

import pytest

from runmedic import loaders
from runmedic.loaders import textlog
from runmedic.model import Run


def by_name(runs):
    return {r.name: r for r in runs}


def test_tensorboard_scalars_and_tensors(tmp_path):
    pytest.importorskip("tensorboard")
    from tensorboard.compat.proto import event_pb2, summary_pb2, tensor_pb2, types_pb2
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    w = EventFileWriter(str(tmp_path / "run1"))
    for i in range(3):
        v = summary_pb2.Summary.Value(tag="train/loss", simple_value=1.0 / (i + 1))
        w.add_event(event_pb2.Event(step=i * 10, summary=summary_pb2.Summary(value=[v])))
    t = tensor_pb2.TensorProto(dtype=types_pb2.DT_DOUBLE, tensor_content=struct.pack("<d", 0.25))
    w.add_event(event_pb2.Event(step=30, summary=summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag="val/loss", tensor=t)])))
    t = tensor_pb2.TensorProto(dtype=types_pb2.DT_FLOAT, float_val=[float("nan")])
    w.add_event(event_pb2.Event(step=40, summary=summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag="train/loss", tensor=t)])))
    w.close()

    run = by_name(loaders.load(str(tmp_path)))["run1"]
    assert run.source == "tensorboard"
    assert run.series["train/loss"].steps == [0, 10, 20, 40]
    assert math.isnan(run.series["train/loss"].values[-1])
    assert run.series["val/loss"].values == [0.25]


def test_tensorboard_truncated_file(tmp_path):
    pytest.importorskip("tensorboard")
    from tensorboard.compat.proto import event_pb2, summary_pb2
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    w = EventFileWriter(str(tmp_path))
    for i in range(5):
        v = summary_pb2.Summary.Value(tag="loss", simple_value=float(i))
        w.add_event(event_pb2.Event(step=i, summary=summary_pb2.Summary(value=[v])))
    w.close()
    f = next(tmp_path.glob("events.out.tfevents*"))
    f.write_bytes(f.read_bytes()[:-7])  # a run that is still writing
    (run,) = loaders.load(str(f))
    assert run.series["loss"].values == [0, 1, 2, 3]


def test_mlflow_file_store(tmp_path):
    run_dir = tmp_path / "mlruns" / "0" / "abc123"
    (run_dir / "metrics" / "train").mkdir(parents=True)
    (run_dir / "tags").mkdir()
    (run_dir / "meta.yaml").write_text("run_id: abc123\n")
    (run_dir / "tags" / "mlflow.runName").write_text("my-run")
    (run_dir / "metrics" / "train" / "loss").write_text("1700000000000 1.5 0\n1700000000001 1.0 1\n1700000000002 nan 2\n")
    (run_dir / "metrics" / "val_loss").write_text("1700000000003 1.2 2\n")
    run = by_name(loaders.load(str(tmp_path)))["my-run"]
    assert run.series["train/loss"].steps == [0, 1, 2]
    assert math.isnan(run.series["train/loss"].values[2])
    assert run.series["val_loss"].values == [1.2]


def test_mlflow_sqlite(tmp_path):
    db = tmp_path / "mlflow.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE runs (run_uuid TEXT, name TEXT)")
    con.execute("CREATE TABLE metrics (key TEXT, value REAL, timestamp INT, run_uuid TEXT, step INT, is_nan BOOL)")
    con.execute("INSERT INTO runs VALUES ('r1', 'sqlite-run')")
    con.executemany("INSERT INTO metrics VALUES (?, ?, ?, 'r1', ?, ?)", [("loss", 2.0, 1, 0, 0), ("loss", 1.0, 2, 1, 0), ("loss", 0.0, 3, 2, 1)])
    con.commit()
    con.close()
    (run,) = loaders.load(str(db))
    assert run.name == "sqlite-run"
    assert run.series["loss"].values[:2] == [2.0, 1.0]
    assert math.isnan(run.series["loss"].values[2])


def test_lightning_csv_with_sparse_columns(tmp_path):
    f = tmp_path / "metrics.csv"
    f.write_text("epoch,step,train_loss,val_loss\n0,9,1.2,\n0,19,1.0,\n0,19,,1.1\n1,29,0.8,\n")
    (run,) = loaders.load(str(f))
    assert run.series["train_loss"].steps == [9, 19, 29]
    assert run.series["val_loss"].steps == [19]
    assert "epoch" not in run.series


def test_jsonl(tmp_path):
    f = tmp_path / "log.jsonl"
    f.write_text('{"step": 1, "loss": 0.9}\nnot json\n{"step": 2, "loss": "NaN", "note": "x"}\n')
    (run,) = loaders.load(str(f))
    assert run.series["loss"].steps == [1, 2]
    assert math.isnan(run.series["loss"].values[1])


def test_hf_trainer_state_uses_longest_checkpoint(tmp_path):
    def state(n):
        rows = [{"loss": 1.0 / (i + 1), "learning_rate": 1e-4, "epoch": i / 10, "step": i * 10} for i in range(n)]
        rows.append({"eval_loss": 0.5, "epoch": 1.0, "step": n * 10})
        rows.append({"train_runtime": 12.0, "train_loss": 0.4, "epoch": 1.0, "step": n * 10})
        return json.dumps({"log_history": rows})

    for name, n in (("checkpoint-50", 5), ("checkpoint-100", 10)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "trainer_state.json").write_text(state(n))
    (run,) = loaders.load(str(tmp_path))
    assert run.source == "hf-trainer"
    assert len(run.series["train_loss"]) == 10  # summary row with average train_loss is dropped
    assert run.series["eval_loss"].values == [0.5]


def parse(*lines) -> Run:
    run = Run("t", "text", "-")
    textlog.parse_lines(lines, run)
    return run.finalize()


def test_text_nanogpt_style():
    run = parse("iter 100: loss 2.3456, time 12.3ms, mfu 30.1%", "step 250: train loss 1.9012, val loss 2.0034", "iter 300: loss nan, time 12.0ms")
    assert run.series["loss"].steps == [100, 300]
    assert run.series["train_loss"].values == [1.9012]
    assert run.series["val_loss"].steps == [250]
    assert "time" not in run.series


def test_text_hf_dicts_without_steps():
    run = parse(
        "{'loss': 0.9, 'grad_norm': 1.3, 'learning_rate': 5e-05, 'epoch': 0.1}",
        "{'loss': 0.7, 'grad_norm': 1.1, 'learning_rate': 4e-05, 'epoch': 0.2}",
        "{'eval_loss': 0.61, 'epoch': 0.2}",
    )
    assert run.series["loss"].values == [0.9, 0.7]
    assert run.series["loss"].steps == [0, 1]
    assert run.series["learning_rate"].values == [5e-05, 4e-05]
    assert "epoch" not in run.series


def test_text_case_and_separators():
    run = parse("Epoch 1 | Step 300 | Loss: 0.5 | lr=3.0e-04", "Epoch 1 | Step 310 | LOSS: 0.4")
    assert run.series["loss"].steps == [300, 310]
    assert run.series["lr"].values == [3e-4]


def test_text_failure_lines_are_kept():
    run = parse(
        "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
        "ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).",
    )
    assert len(run.log_lines) == 2


def test_directory_with_mixed_logs(tmp_path):
    (tmp_path / "train.log").write_text("step 1 loss 1.0\nstep 2 loss 0.9\n")
    (tmp_path / "metrics.csv").write_text("step,loss\n1,1.0\n")
    (tmp_path / "README.md").write_text("loss 5")
    names = {r.name for r in loaders.load(str(tmp_path))}
    assert names == {"train", "metrics"}


def test_missing_path():
    with pytest.raises(FileNotFoundError):
        loaders.load("does/not/exist")


def test_text_log_next_to_tensorboard_only_adds_errors(tmp_path):
    pytest.importorskip("tensorboard")
    from tensorboard.compat.proto import event_pb2, summary_pb2
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    w = EventFileWriter(str(tmp_path))
    for i in range(3):
        v = summary_pb2.Summary.Value(tag="loss", simple_value=float(i))
        w.add_event(event_pb2.Event(step=i, summary=summary_pb2.Summary(value=[v])))
    w.close()
    (tmp_path / "train.log").write_text("step 1 loss 5.0\ntorch.OutOfMemoryError: CUDA out of memory.\n")
    (run,) = loaders.load(str(tmp_path))
    assert run.source == "tensorboard"
    assert run.series["loss"].values == [0, 1, 2]
    assert len(run.log_lines) == 1


def test_text_hf_values_as_strings():
    # transformers 5 prints metric values as strings
    run = parse("{'loss': '0.9637', 'grad_norm': '2.371', 'learning_rate': '0.04867', 'epoch': '0.1'}")
    assert run.series["loss"].values == [0.9637]
    assert run.series["learning_rate"].values == [0.04867]
