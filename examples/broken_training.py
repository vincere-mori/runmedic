"""Small real PyTorch trainings that fail in classic ways, to see what runmedic says about them.

    python examples/broken_training.py nan        # LR too high, no warmup, no clipping -> NaN
    python examples/broken_training.py overfit    # tiny noisy dataset, big model, many epochs
    python examples/broken_training.py slow-data  # slow __getitem__, num_workers=0, checkpoint every 50 steps
    python examples/broken_training.py healthy    # the same model trained sensibly

Logs go to examples/logs/<scenario>/ (TensorBoard events + train.log), then:

    runmedic diagnose examples/logs/nan
"""

import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.tensorboard import SummaryWriter

LOGS = Path(__file__).parent / "logs"


class Logger:
    def __init__(self, name: str):
        self.out = LOGS / name
        self.out.mkdir(parents=True, exist_ok=True)
        for old in self.out.glob("events.out.tfevents*"):
            old.unlink()
        self.tb = SummaryWriter(str(self.out))
        self.txt = (self.out / "train.log").open("w", encoding="utf-8")

    def log(self, step: int, **metrics):
        for k, v in metrics.items():
            self.tb.add_scalar(k, v, step)
        line = f"step {step} " + " ".join(f"{k}={v:.6g}" for k, v in metrics.items())
        self.txt.write(line + "\n")

    def close(self):
        self.tb.close()
        self.txt.close()
        # the default file name contains the hostname; keep the committed demo logs neutral
        for f in self.out.glob("events.out.tfevents*"):
            f.rename(self.out / "events.out.tfevents.demo")


def grad_norm(model: nn.Module) -> float:
    norms = [p.grad.norm() for p in model.parameters() if p.grad is not None]
    return torch.stack(norms).norm().item() if norms else 0.0


def nan_run():
    torch.manual_seed(0)
    x = torch.randn(4096, 32) * 20  # unnormalized features
    y = (x @ torch.randn(32, 1)) + torch.randn(4096, 1)
    model = nn.Sequential(nn.Linear(32, 256), nn.ReLU(), nn.Linear(256, 1))
    opt = torch.optim.SGD(model.parameters(), lr=2e-6, momentum=0.9)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.015**s)  # LR keeps growing: a broken schedule
    log = Logger("nan")
    loader = DataLoader(TensorDataset(x, y), batch_size=64, shuffle=True)
    step = 0
    for _ in range(20):
        for xb, yb in loader:
            loss = nn.functional.mse_loss(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            log.log(step, **{"train/loss": loss.item(), "train/grad_norm": grad_norm(model), "train/lr": sched.get_last_lr()[0]})
            opt.step()
            sched.step()
            step += 1
            if step >= 400:
                log.close()
                return


def classification_data(n, seed, noise=0.25):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 20, generator=g)
    y = (x[:, :5].sum(1) > 0).long()
    flip = torch.rand(n, generator=g) < noise
    y[flip] = 1 - y[flip]
    return x, y


def overfit_run(healthy: bool = False):
    torch.manual_seed(0)
    xtr, ytr = classification_data(4000 if healthy else 300, 1)
    xva, yva = classification_data(2000, 2)
    hidden = 64 if healthy else 1024
    model = nn.Sequential(nn.Linear(20, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1 if healthy else 0.0)
    epochs = 12 if healthy else 150
    loader = DataLoader(TensorDataset(xtr, ytr), batch_size=64, shuffle=True)
    total = epochs * len(loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=total, pct_start=0.05)
    log = Logger("healthy" if healthy else "overfit")
    step = 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            loss = nn.functional.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            log.log(step, **{"train/loss": loss.item(), "train/lr": sched.get_last_lr()[0]})
            step += 1
        model.eval()
        with torch.no_grad():
            log.log(step, **{"val/loss": nn.functional.cross_entropy(model(xva), yva).item()})
    log.close()


class SlowDataset(Dataset):
    def __init__(self):
        self.x, self.y = classification_data(2000, 3)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        time.sleep(0.0006)  # decoding / augmentation done in Python, per sample
        return self.x[i], self.y[i]


def slow_data_run():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 2))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loader = DataLoader(SlowDataset(), batch_size=32, shuffle=True, num_workers=0)
    log = Logger("slow-data")
    step = 0
    for _ in range(5):
        t_prev = time.perf_counter()
        for xb, yb in loader:
            t_data = time.perf_counter()
            loss = nn.functional.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 50 == 49:
                time.sleep(0.3)  # "checkpoint" saved synchronously inside the loop
            t_end = time.perf_counter()
            log.log(step, **{"train/loss": loss.item(), "perf/data_time": t_data - t_prev, "perf/step_time": t_end - t_prev})
            t_prev = t_end
            step += 1
    log.close()


if __name__ == "__main__":
    scenario = sys.argv[1] if len(sys.argv) > 1 else "nan"
    {"nan": nan_run, "overfit": overfit_run, "slow-data": slow_data_run, "healthy": lambda: overfit_run(healthy=True)}[scenario]()
    print(f"logs written to {LOGS / scenario}")
