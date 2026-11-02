"""Summarise results/runs.csv into figures and a Markdown results table."""

from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data import AGRI_CLASSES, CLASSES, ROOT  # noqa: E402

RESULTS = ROOT / "results"
FIG = RESULTS / "figures"
LABELS = {
    "rf_rgb": "Random forest, RGB statistics",
    "rf_ms": "Random forest, 13-band statistics + indices",
    "cnn_rgb_scratch": "ResNet-18, RGB, from scratch",
    "cnn_rgb_imagenet": "ResNet-18, RGB, ImageNet-pretrained",
    "cnn_ms_scratch": "ResNet-18, 13 bands, from scratch",
    "cnn_ms_imagenet": "ResNet-18, 13 bands, ImageNet-initialised",
}
STYLE = {
    "rf_rgb": ("#9AA5B1", "v", ":"), "rf_ms": ("#52606D", "s", ":"),
    "cnn_rgb_scratch": ("#F0B429", "^", "--"), "cnn_rgb_imagenet": ("#DE911D", "o", "-"),
    "cnn_ms_scratch": ("#3EBD93", "D", "--"), "cnn_ms_imagenet": ("#00793E", "o", "-"),
}
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "savefig.bbox": "tight"})


def main(path: str = "runs.csv") -> None:
    df = pd.read_csv(RESULTS / path, dtype={"budget": str})
    full_n = int(df.loc[df["budget"] == "all", "n_train"].max() / 10) if (df["budget"] == "all").any() else None
    df["n_pc"] = df["budget"].apply(lambda b: full_n if b == "all" else int(b))
    agg = df.groupby(["method", "n_pc"]).agg(
        acc=("accuracy", "mean"), acc_sd=("accuracy", "std"),
        f1=("macro_f1", "mean"), f1_sd=("macro_f1", "std"),
        agri=("agri_f1", "mean"), agri_sd=("agri_f1", "std"), seeds=("seed", "nunique")).reset_index()
    FIG.mkdir(parents=True, exist_ok=True)

    for metric, ylabel, name in [("f1", "Macro-F1 (10 classes)", "learning_curve_macro_f1"),
                                 ("agri", "Mean F1, agricultural classes", "learning_curve_agri_f1")]:
        fig, ax = plt.subplots(figsize=(6.2, 3.4))
        for m in LABELS:
            a = agg[agg["method"] == m].sort_values("n_pc")
            if a.empty:
                continue
            c, mk, ls = STYLE[m]
            ax.errorbar(a["n_pc"], a[metric], yerr=a[f"{metric}_sd"].fillna(0), color=c, marker=mk, ls=ls,
                        ms=4, capsize=2, label=LABELS[m])
        ax.set_xscale("log")
        ax.set_xlabel("Labelled training images per class (log scale)")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=7, loc="lower right")
        fig.savefig(FIG / f"{name}.png", dpi=160)
        plt.close(fig)

    conf_path = RESULTS / path.replace("runs", "confusions").replace(".csv", ".json")
    if conf_path.exists():
        conf = json.loads(conf_path.read_text())
        key = next((k for k in ("cnn_ms_imagenet|25", "cnn_rgb_imagenet|25") if k in conf), None)
        if key:
            cm = np.array(conf[key])[np.ix_(AGRI_CLASSES, AGRI_CLASSES)].astype(float)
            cmn = cm / cm.sum(1, keepdims=True)
            names = [CLASSES[i] for i in AGRI_CLASSES]
            fig, ax = plt.subplots(figsize=(4.2, 3.6))
            ax.imshow(cmn, cmap="Greens", vmin=0, vmax=1)
            ax.grid(False)
            for i in range(len(names)):
                for j in range(len(names)):
                    ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                            color="white" if cmn[i, j] > 0.6 else "black", fontsize=8)
            ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
            ax.set_yticks(range(len(names)), names)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"{LABELS[key.split('|')[0]]}, 25 labels/class\n(rows normalised, agricultural classes)", fontsize=8)
            fig.savefig(FIG / "confusion_agri_25.png", dpi=160)
            plt.close(fig)

    lines = ["| Method | " + " | ".join(f"{n}/class" for n in sorted(agg["n_pc"].unique())) + " |",
             "|---|" + "---|" * agg["n_pc"].nunique()]
    for m in LABELS:
        a = agg[agg["method"] == m].set_index("n_pc")
        if a.empty:
            continue
        cells = [f"{a.loc[n, 'f1']:.3f} ± {0 if np.isnan(a.loc[n, 'f1_sd']) else a.loc[n, 'f1_sd']:.3f}"
                 if n in a.index else "" for n in sorted(agg["n_pc"].unique())]
        lines.append(f"| {LABELS[m]} | " + " | ".join(cells) + " |")
    agri_lines = ["| Method | " + " | ".join(f"{n}/class" for n in sorted(agg["n_pc"].unique())) + " |",
                  "|---|" + "---|" * agg["n_pc"].nunique()]
    for m in LABELS:
        a = agg[agg["method"] == m].set_index("n_pc")
        if a.empty:
            continue
        agri_lines.append(f"| {LABELS[m]} | " + " | ".join(
            f"{a.loc[n, 'agri']:.3f}" if n in a.index else "" for n in sorted(agg["n_pc"].unique())) + " |")

    per_class = df[df["budget"] == "25"].groupby("method")[[f"f1_{i}" for i in range(10)]].mean() \
        if (df["budget"] == "25").any() else None
    summary = {
        "macro_f1_table": "\n".join(lines),
        "agri_f1_table": "\n".join(agri_lines),
        "per_class_f1_at_25": None if per_class is None else per_class.round(3).rename(
            columns={f"f1_{i}": c for i, c in enumerate(CLASSES)}).to_dict(orient="index"),
        "full_n_per_class": full_n,
        "seeds": int(df["seed"].nunique()),
        "test_images": None,
    }
    (RESULTS / path.replace(".csv", "_summary.json")).write_text(json.dumps(summary, indent=2))
    print(summary["macro_f1_table"])
    print()
    print(summary["agri_f1_table"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs.csv")
