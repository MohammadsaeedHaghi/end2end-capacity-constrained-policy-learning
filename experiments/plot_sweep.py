"""
Plot the three sweep metrics (V_oracle, V_IPW_eval, V_DR_eval) vs N with 95%
CI bands across seeds. Reads results/sweep.csv.

Outputs:
    results/figures/V_oracle_vs_N.png
    results/figures/V_IPW_eval_vs_N.png
    results/figures/V_DR_eval_vs_N.png
    results/figures/summary.png        (1x3 panel of the three above)
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SWEEP_CSV = "results/sweep.csv"
FIGURES_DIR = "results/figures"

METRICS = ["V_oracle", "V_IPW_eval", "V_DR_eval"]
METRIC_LABELS = {
    "V_oracle":   r"$V_{\mathrm{oracle}}$",
    "V_IPW_eval": r"$V_{\mathrm{IPW}}$ (eval)",
    "V_DR_eval":  r"$V_{\mathrm{DR}}$ (eval)",
}

TRAINED_METHODS = [
    "G",
    "F",
    "S2-linear",
    "S2-lasso",
    "S2-tree",
    "S2-knn",
    "S2-dr",
]
REFERENCE_METHODS = {
    "random":               {"color": "gray",  "linestyle": "--"},
    "oracle_greedy_no_cap": {"color": "black", "linestyle": "--"},
}


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

    for i, method in enumerate(TRAINED_METHODS):
        g = _summarise(df, method, metric)
        if g is None:
            continue
        color = cmap(i)
        ax.plot(g["N"], g["mean"], marker="o", color=color, label=method, lw=1.6)
        ax.fill_between(g["N"], g["mean"] - g["ci"], g["mean"] + g["ci"],
                        color=color, alpha=0.15, linewidth=0)

    for method, style in REFERENCE_METHODS.items():
        g = _summarise(df, method, metric)
        if g is None:
            continue
        ax.plot(g["N"], g["mean"], color=style["color"],
                linestyle=style["linestyle"], lw=1.4, label=method)

    ax.set_xlabel("N (training size)")
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.grid(True, alpha=0.3)


def main():
    if not os.path.exists(SWEEP_CSV):
        raise SystemExit(f"{SWEEP_CSV} not found — run sweep + aggregate first.")

    df = pd.read_csv(SWEEP_CSV)
    if df.empty:
        raise SystemExit(f"{SWEEP_CSV} is empty.")

    os.makedirs(FIGURES_DIR, exist_ok=True)

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
        ax.set_title(f"{METRIC_LABELS[metric]} vs N (mean ± 95% CI)")
        ax.legend(loc="best", fontsize=8, ncols=2)
        out = os.path.join(FIGURES_DIR, f"{metric}_vs_N.png")
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"[plot] {out}")

    # Combined 1x3 summary.
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5), sharex=True)
    for ax, metric in zip(axes, METRICS):
        _plot_metric(df, metric, ax)
        ax.set_title(METRIC_LABELS[metric])
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
                    fontsize=8, frameon=False)
    fig.suptitle("Off-policy value vs training size (mean ± 95% CI, 20 seeds)",
                 y=1.02)
    out = os.path.join(FIGURES_DIR, "summary.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out}")


if __name__ == "__main__":
    main()
