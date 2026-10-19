import math
import random

from helpers import findings, healthy_loss, make_run


def test_healthy_run_is_quiet():
    n = 300
    val_steps = list(range(0, n, 30))
    val = [healthy_loss(n, start=2.6, end=0.35, noise=0)[s] for s in val_steps]
    run = make_run(train_loss=healthy_loss(n), val_loss=(val_steps, val), lr=[1e-3 * min(1, i / 20) * (1 - i / n) + 1e-5 for i in range(n)])
    assert findings(run) == {}


def test_nan_after_exploding_loss():
    loss = healthy_loss(200)[:150] + [1.0 * 2**k for k in range(10)] + [float("nan")] * 40
    gn = [1.0] * 150 + [5.0 * 3**k for k in range(10)] + [float("nan")] * 40
    f = findings(make_run(train_loss=loss, grad_norm=gn, lr=[3e-3] * 200))
    nf = f["non_finite"]
    assert nf.severity == "critical"
    assert nf.step == 160
    assert any("Exploding gradients" in c for c in nf.causes)
    assert any("no warmup" in e for e in nf.evidence)
    assert "divergence" not in f  # NaN is reported once, not again as divergence


def test_nan_from_first_step_is_a_bug():
    f = findings(make_run(train_loss=[float("nan")] * 50))
    nf = f["non_finite"]
    assert nf.step == 0
    assert "bug" in nf.causes[0]


def test_sudden_nan_without_lead_up():
    loss = healthy_loss(100) + [float("inf")]
    nf = findings(make_run(train_loss=loss))["non_finite"]
    assert any("no visible lead-up" in c for c in nf.causes)


def test_divergence():
    n = 300
    loss = [2.0 - 1.5 * i / 150 for i in range(150)] + [0.5 + 2.0 * (i / 150) for i in range(150)]
    f = findings(make_run(train_loss=loss))
    d = f["divergence"]
    assert d.severity == "critical"
    assert 140 <= d.step <= 160
    assert "no_learning" not in f
    assert len(loss) == n


def test_partial_regression_is_warning_only():
    loss = [2.0 - 1.5 * i / 200 for i in range(200)] + [0.5 + 1.0 * i / 100 for i in range(100)]
    assert findings(make_run(train_loss=loss))["divergence"].severity == "warning"


def test_spikes():
    loss = healthy_loss(400)
    for i in (50, 120, 300):
        loss[i] *= 4
    f = findings(make_run(train_loss=loss))
    sp = f["loss_spikes"]
    assert sp.severity == "warning"
    assert "50" in sp.evidence[1] and "300" in sp.evidence[1]


def test_spikes_blame_missing_warmup():
    loss = healthy_loss(400)
    for i in (5, 12, 20):
        loss[i] *= 5
    lr = [1e-3 * (1 - i / 400) for i in range(400)]
    sp = findings(make_run(train_loss=loss, lr=lr))["loss_spikes"]
    assert "missing warmup" in sp.causes[0]


def test_no_learning_and_uniform_prediction():
    rng = random.Random(1)
    loss = [math.log(10) + rng.gauss(0, 0.005) for _ in range(200)]
    nl = findings(make_run(train_loss=loss))["no_learning"]
    assert any("ln(10)" in e for e in nl.evidence)
    assert "single batch" in nl.fixes[0]


def test_zero_lr():
    nl = findings(make_run(train_loss=[1.0] * 100, lr=[0.0] * 100))["no_learning"]
    assert "zero" in nl.causes[0]


def test_overfitting():
    n = 500
    train = healthy_loss(n, end=0.05, noise=0)
    val_steps = list(range(0, n, 50))
    val = [1.5, 0.9, 0.6, 0.5, 0.48, 0.52, 0.6, 0.7, 0.8, 0.9]
    ov = findings(make_run(train_loss=train, val_loss=(val_steps, val)))["overfitting"]
    assert ov.step == 200
    assert "early stopping" in ov.fixes[0]


def test_val_worse_but_train_flat_is_not_overfitting():
    train = [0.5] * 200
    val = ([0, 50, 100, 150, 199], [0.6, 0.55, 0.6, 0.7, 0.8])
    assert "overfitting" not in findings(make_run(train_loss=train, val_loss=val))


def test_val_below_train():
    train = healthy_loss(300, start=1.0, end=0.5, noise=0)
    val = (list(range(0, 300, 30)), [0.3] * 10)
    assert findings(make_run(train_loss=train, val_loss=val))["val_below_train"].severity == "info"


def test_grad_norm_spikes():
    gn = [1.0 + 0.1 * math.sin(i) for i in range(100)]
    gn[40] = gn[70] = 50.0
    g = findings(make_run(train_loss=healthy_loss(100), grad_norm=gn))["grad_spikes"]
    assert g.step in (40, 70)


def test_constant_lr_plateau():
    loss = [2.0 - 1.8 * min(1, i / 120) for i in range(300)]
    f = findings(make_run(train_loss=loss, lr=[1e-3] * 300))
    assert f["constant_lr_plateau"].severity == "info"


def test_low_gpu_util_and_data_wait():
    f = findings(make_run(train_loss=healthy_loss(100), gpu_util=[0.35] * 100, data_time=[0.2] * 100, step_time=[0.3] * 100))
    g = f["input_bound"]
    assert g.severity == "warning"
    assert any("35%" in e for e in g.evidence)
    assert any("67%" in e for e in g.evidence)


def test_busy_gpu_is_fine():
    assert "input_bound" not in findings(make_run(train_loss=healthy_loss(100), gpu_util=[95.0] * 100))


def test_periodic_slow_steps():
    st = [0.1] * 500
    for i in range(50, 500, 100):
        st[i] = 2.0
    s = findings(make_run(train_loss=healthy_loss(500), step_time=st))["slow_steps"]
    assert "every 100 steps" in s.evidence[1]
    assert s.severity == "warning"


def test_throughput_slowdown():
    tp = [1000 - 1.5 * i for i in range(400)]
    assert "slowdown" in findings(make_run(train_loss=healthy_loss(400), samples_per_sec=tp))


def test_gpu_memory_leak():
    mem = [4000 + 10 * i for i in range(200)]
    m = findings(make_run(train_loss=healthy_loss(200), cuda_memory_allocated=mem))["memory_growth"]
    assert "GPU" in m.title
    assert "loss.item()" in m.causes[0]


def test_flat_memory_is_fine():
    assert "memory_growth" not in findings(make_run(train_loss=healthy_loss(200), cuda_memory_allocated=[4000.0] * 200))


def test_nan_with_ever_growing_lr():
    loss = healthy_loss(200)[:150] + [1.0 * 2**k for k in range(10)] + [float("nan")] * 40
    lr = [1e-5 * 1.02**i for i in range(200)]
    nf = findings(make_run(train_loss=loss, lr=lr))["non_finite"]
    assert any("still rising" in e for e in nf.evidence)
    assert "never stops increasing" in nf.causes[0]
