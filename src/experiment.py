"""
Label-efficiency study on EuroSAT-MS.

For each label budget (images per class) and seed, a class-balanced training subset is drawn from the
training pool (at least 250 images per class) and every method is trained with a fixed recipe (no tuning on the test set), then scored
on the full held-out EuroSAT test split (5,400 images). Results are appended to results/runs.csv so the study can be resumed.

    python src/experiment.py            # full grid
    python src/experiment.py --quick    # one seed, two budgets (smoke test)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from data import AGRI_CLASSES, CLASSES, RGB, ROOT, band_statistics, load_split, sample_per_class
from models import random_forest, resnet18_for, spectral_features

RESULTS = ROOT / "results"
BUDGETS: list[int | None] = [10, 25, 50, 100, 250]
SEEDS = [0, 1, 2]
METHODS = ["rf_rgb", "rf_ms", "cnn_rgb_scratch", "cnn_rgb_imagenet", "cnn_ms_scratch", "cnn_ms_imagenet"]
FIELDS = ["method", "budget", "seed", "n_train", "accuracy", "macro_f1", "agri_f1", "seconds"] + \
         [f"f1_{i}" for i in range(len(CLASSES))]


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def epochs_for(n_per_class: int) -> int:
    """Roughly constant optimisation effort: more passes over smaller subsets, capped for large ones."""
    return int(min(100, max(15, math.ceil(3000 / (10 * n_per_class) * 2))))


def augment(xb: torch.Tensor) -> torch.Tensor:
    """Random flips and 90-degree rotations: satellite patches have no canonical orientation."""
    if torch.rand(1).item() < 0.5:
        xb = xb.flip(-1)
    if torch.rand(1).item() < 0.5:
        xb = xb.flip(-2)
    k = int(torch.randint(0, 4, (1,)).item())
    return torch.rot90(xb, k, dims=(-2, -1)) if k else xb


def train_cnn(x_tr, y_tr, x_te, method, seed, dev, n_per_class) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rgb = "_rgb_" in method
    bands = RGB if rgb else list(range(13))
    mu, sd = band_statistics(x_tr[:, bands])
    mu_t = torch.tensor(mu, device=dev).view(1, -1, 1, 1)
    sd_t = torch.tensor(sd, device=dev).view(1, -1, 1, 1)

    def prep(x):
        t = torch.from_numpy(x[:, bands].astype(np.float32)).to(dev)
        return (t - mu_t) / sd_t

    xt, yt = prep(x_tr), torch.from_numpy(y_tr).to(dev)
    model = resnet18_for(len(bands), pretrained=method.endswith("imagenet")).to(dev)
    epochs = epochs_for(n_per_class)
    bs = min(64, len(xt))
    steps = epochs * math.ceil(len(xt) / bs)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.1)

    model.train()
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(xt), generator=g).to(dev)
        for i in range(0, len(xt), bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            loss = F.cross_entropy(model(augment(xt[idx])), yt[idx], label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x_te), 256):
            preds.append(model(prep(x_te[i:i + 256])).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def evaluate(y_true, y_pred) -> dict:
    per_class = f1_score(y_true, y_pred, average=None, labels=list(range(len(CLASSES))), zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": float(per_class.mean()),
        "agri_f1": float(per_class[AGRI_CLASSES].mean()),
        **{f"f1_{i}": float(v) for i, v in enumerate(per_class)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    budgets = [25, 100] if args.quick else BUDGETS
    seeds = [0] if args.quick else SEEDS
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / ("runs_quick.csv" if args.quick else "runs.csv")

    x_tr_all, y_tr_all = load_split("train")
    x_te, y_te = load_split("test")
    dev = device()
    print(f"train {x_tr_all.shape}, test {x_te.shape}, device {dev}")

    done = set()
    if out.exists():
        with open(out) as fh:
            done = {(r["method"], r["budget"], r["seed"]) for r in csv.DictReader(fh)}
    new_file = not out.exists()
    feats_te = {"rf_rgb": spectral_features(x_te, rgb_only=True), "rf_ms": spectral_features(x_te)}
    confusions = {}

    with open(out, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for budget in budgets:
            for seed in seeds:
                idx = sample_per_class(y_tr_all, budget, seed)
                x_tr, y_tr = x_tr_all[idx], y_tr_all[idx]
                n_pc = budget or int(np.bincount(y_tr).min())
                for method in METHODS:
                    key = (method, str(budget or "all"), str(seed))
                    if key in done:
                        continue
                    t0 = time.time()
                    if method.startswith("rf"):
                        rf = random_forest(seed).fit(spectral_features(x_tr, rgb_only=method == "rf_rgb"), y_tr)
                        pred = rf.predict(feats_te[method])
                    else:
                        pred = train_cnn(x_tr, y_tr, x_te, method, seed, dev, n_pc)
                    m = evaluate(y_te, pred)
                    row = {"method": method, "budget": budget or "all", "seed": seed, "n_train": len(idx),
                           "seconds": round(time.time() - t0, 1), **{k: round(v, 4) for k, v in m.items()}}
                    writer.writerow(row)
                    fh.flush()
                    if seed == 0 and budget in (25, None):
                        confusions[f"{method}|{budget or 'all'}"] = confusion_matrix(
                            y_te, pred, labels=list(range(len(CLASSES)))).tolist()
                    print(f"{method:18s} budget={str(budget or 'all'):>4s} seed={seed} "
                          f"acc={m['accuracy']:.3f} macroF1={m['macro_f1']:.3f} agriF1={m['agri_f1']:.3f} "
                          f"({row['seconds']}s)", flush=True)

    if confusions:
        path = RESULTS / ("confusions_quick.json" if args.quick else "confusions.json")
        existing = json.loads(path.read_text()) if path.exists() else {}
        existing.update(confusions)
        path.write_text(json.dumps(existing))


if __name__ == "__main__":
    main()
