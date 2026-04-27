"""
Plot the sweep metrics vs N with 95% CI bands across seeds. Reads
results/sweep.csv. CLI flags `--n-min`, `--n-max`, `--out-dir` allow
filtering the N range and writing to a specific output directory.

Outputs (under --out-dir, default results/figures):
    V_oracle_vs_N.png
    V_IPW_eval_vs_N.png
    V_DR_eval_vs_N.png
    cap_viol_sup_vs_N.png
    summary.png        (1x4 panel)
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SWEEP_CSV = "results/sweep.csv"
FIGURES_DIR = "results/figures"

METRICS = ["V_oracle", "V_IPW_eval", "V_DR_eval", "cap_viol_sup"]
METRIC_LABELS = {
    "V_oracle":     r"$V_{\mathrm{oracle}}$",
    "V_IPW_eval":   r"$V_{\mathrm{IPW}}$ (eval)",
    "V_DR_eval":    r"$V_{\mathrm{DR}}$ (eval)",
    "cap_viol_sup": "capacity violation (sup)",
}

TRAINED_METHODS = [
    "F",
    "S2-linear",
    "S2-lasso",
    "S2-tree",
    "S2-knn",
]
REFERENCE_METHODS = {}


def _summarise(df, method, metric):
    sub = df[df["method"] == method].copy()
    if sub.empty:
        return None
    g = sub.groupby("N")[metric].agg(["mean", "std", "count"]).reset_index()
    g["se"] = g["std"] / np.sqrt(g["count"].clip(lower=1))
    g["ci"] = 1.96 * g["se"]
    return g


def _plot_metric(df, metric, ax):
    cmap = plt.get_cmap("tab10")

    # Reserve cmap(0) for G/G-mu so the F-and-onwards colors stay consistent
    # whether or not G is in TRAINED_METHODS.
    g_aliases = {"G", "G-mu"}
    for i, method in enumerate(TRAINED_METHODS):
        g = _summarise(df, method, metric)
        if g is None:
            continue
        color = cmap(i) if method in g_aliases or "G" in TRAINED_METHODS \
            or "G-mu" in TRAINED_METHODS else cmap(i + 1)
        ax.plot(g["N"], g["mean"], marker="o", color=color, label=method, lw=1.6)

    for method, style in REFERENCE_METHODS.items():
        g = _summarise(df, method, metric)
        if g is None:
            continue
        ax.plot(g["N"], g["mean"], color=style["color"],
                linestyle=style["linestyle"], lw=1.4, label=method)

    ax.set_xlabel("N (training size)")
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.grid(True, alpha=0.3)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-min", type=int, default=None,
                   help="Filter sweep CSV to N >= n-min")
    p.add_argument("--n-max", type=int, default=None,
                   help="Filter sweep CSV to N <= n-max")
    p.add_argument("--out-dir", type=str, default=FIGURES_DIR,
                   help="Output directory for plots (default: results/figures)")
    return p.parse_args()


def main():
    args = _parse_args()
    if not os.path.exists(SWEEP_CSV):
        raise SystemExit(f"{SWEEP_CSV} not found — run sweep + aggregate first.")

    df = pd.read_csv(SWEEP_CSV)
    if df.empty:
        raise SystemExit(f"{SWEEP_CSV} is empty.")

    if args.n_min is not None:
        df = df[df["N"] >= args.n_min]
    if args.n_max is not None:
        df = df[df["N"] <= args.n_max]
    if df.empty:
        raise SystemExit("No rows after N filter.")

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    n_seeds_per_cell = (
        df.groupby(["method", "N"]).size().reset_index(name="n").pivot(
            index="method", columns="N", values="n"
        )
    )
    print("[plot] seeds per (method, N):")
    print(n_seeds_per_cell.to_string())

    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(8, 5))
        _plot_metric(df, metric, ax)
        ax.set_title(f"{METRIC_LABELS[metric]} vs N (mean over 20 seeds)")
        ax.legend(loc="best", fontsize=8, ncols=2)
        out = os.path.join(out_dir, f"{metric}_vs_N.png")
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"[plot] {out}")

    # Combined 1xN summary.
    fig, axes = plt.subplots(1, len(METRICS), figsize=(6.5 * len(METRICS), 5.5),
                             sharex=True)
    for ax, metric in zip(axes, METRICS):
        _plot_metric(df, metric, ax)
        ax.set_title(METRIC_LABELS[metric])
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
                    fontsize=8, frameon=False)
    fig.suptitle("Method comparison vs training size (mean over 20 seeds)",
                 y=1.02)
    out = os.path.join(out_dir, "summary.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out}")


if __name__ == "__main__":
    main()
