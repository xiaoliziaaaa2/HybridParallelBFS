#!/usr/bin/env python3
"""
plot_speedup.py

Read results/timing.csv and generate speedup / scaling plots.

Produces:
  results/speedup.png  — 4-panel figure:
    1. Pure MPI: speedup vs. number of MPI ranks (strong scaling)
    2. Hybrid: speedup vs. total cores (np × threads)
    3. Pure MPI vs. Hybrid at matched total-core counts
    4. Runtime (ms) vs. total cores (log–log)

Usage:
    python scripts/plot_speedup.py [--csv results/timing.csv] [-o results/speedup.png]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Styling — clean, publication-ready look
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi":      150,
    "savefig.dpi":     150,
    "font.size":       10,
    "axes.titlesize":  11,
    "axes.labelsize":  10,
    "legend.fontsize": 8,
    "figure.figsize":  (12, 9),
    "lines.linewidth": 1.6,
    "lines.markersize":5,
})

COLORS  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
MARKERS = ["o", "s", "D", "^", "v"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def compute_speedup(df):
    """Compute speedup relative to T(1) = pure MPI np=1 median time, per graph."""
    baseline = {}
    for gs in df["graph_size"].unique():
        mask = (df["graph_size"] == gs) & (df["mode"] == "mpi") & (df["np"] == 1)
        sub = df[mask]
        if not sub.empty:
            baseline[gs] = sub["time_ms"].median()
        else:
            baseline[gs] = df[df["graph_size"] == gs]["time_ms"].min()

    df["speedup"] = df.apply(
        lambda r: baseline.get(r["graph_size"], r["time_ms"]) / r["time_ms"]
        if r["time_ms"] > 0 else np.nan, axis=1
    )
    # Clip for visual clarity — speedup below 0.5 is noise
    df["speedup"] = df["speedup"].clip(lower=0.5)
    return df


def median_agg(df, group_cols):
    """Group by the given columns and compute median of time_ms & speedup."""
    return df.groupby(group_cols, as_index=False).agg(
        time_ms=("time_ms", "median"),
        speedup=("speedup", "median"),
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_pure_mpi_speedup(ax, df):
    """Panel 1: Pure MPI speedup vs. np, one line per graph size."""
    mpi = df[df["mode"] == "mpi"]
    agg = median_agg(mpi, ["graph_size", "np"])

    for i, gs in enumerate(sorted(agg["graph_size"].unique())):
        sub = agg[agg["graph_size"] == gs]
        ax.plot(sub["np"], sub["speedup"],
                marker=MARKERS[i % len(MARKERS)], color=COLORS[i % len(COLORS)],
                label=gs, markersize=6)

    # Ideal line
    x_ideal = sorted(agg["np"].unique())
    ax.plot(x_ideal, x_ideal, "k--", alpha=0.3, label="Ideal")
    ax.set_xlabel("Number of MPI ranks")
    ax.set_ylabel("Speedup")
    ax.set_title("Pure MPI — Strong Scaling")
    ax.legend(title="Graph size", framealpha=0.7)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))


def plot_hybrid_speedup(ax, df):
    """Panel 2: Hybrid speedup vs. total cores, one line per graph size."""
    hyb = df[df["mode"] == "hybrid"]
    agg = median_agg(hyb, ["graph_size", "total_cores"])

    for i, gs in enumerate(sorted(agg["graph_size"].unique())):
        sub = agg[agg["graph_size"] == gs]
        ax.plot(sub["total_cores"], sub["speedup"],
                marker=MARKERS[i % len(MARKERS)], color=COLORS[i % len(COLORS)],
                label=gs, markersize=6)

    x_ideal = sorted(agg["total_cores"].unique())
    ax.plot(x_ideal, x_ideal, "k--", alpha=0.3, label="Ideal")
    ax.set_xlabel("Total cores (np × threads)")
    ax.set_ylabel("Speedup")
    ax.set_title("Hybrid MPI+OpenMP — Strong Scaling")
    ax.legend(title="Graph size", framealpha=0.7)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))


def plot_mpi_vs_hybrid(ax, df):
    """Panel 3: Bar chart — Pure MPI vs. Hybrid at same total-core count."""
    # For each total_core that appears in both modes, compare
    mpi_agg  = median_agg(df[df["mode"] == "mpi"], ["graph_size", "np"])
    mpi_agg  = mpi_agg.rename(columns={"np": "total_cores"})
    hyb_agg  = median_agg(df[df["mode"] == "hybrid"], ["graph_size", "total_cores"])

    # Merge on (graph_size, total_cores)
    merged = pd.merge(
        mpi_agg[["graph_size", "total_cores", "speedup"]],
        hyb_agg[["graph_size", "total_cores", "speedup"]],
        on=["graph_size", "total_cores"], suffixes=("_mpi", "_hybrid"),
        how="inner"
    )

    if merged.empty:
        ax.text(0.5, 0.5, "No matched core counts for comparison",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("MPI vs. Hybrid (matched cores)")
        return

    gs_list = sorted(merged["graph_size"].unique())
    x       = np.arange(len(merged))
    width   = 0.35

    ax.bar(x - width/2, merged["speedup_mpi"],    width,
           label="Pure MPI",   color=COLORS[1], alpha=0.85)
    ax.bar(x + width/2, merged["speedup_hybrid"], width,
           label="Hybrid",     color=COLORS[0], alpha=0.85)

    # Build tick labels: "graph_size\nnp=N"
    labels = [f"{r['graph_size']}\nnp={int(r['total_cores'])}"
              for _, r in merged.iterrows()]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Speedup")
    ax.set_title("Pure MPI vs. Hybrid (matched total cores)")
    ax.legend(framealpha=0.7)
    ax.grid(True, alpha=0.3, axis="y")


def plot_runtime(ax, df):
    """Panel 4: Runtime (ms) vs. total cores, log–log."""
    df["total_cores"] = df["np"] * df["threads"]
    agg = df.groupby(["graph_size", "mode", "total_cores"], as_index=False).agg(
        time_ms=("time_ms", "median")
    )

    for i, gs in enumerate(sorted(agg["graph_size"].unique())):
        sub = agg[agg["graph_size"] == gs]
        # Separate pure MPI and hybrid with solid vs dashed
        for mode, ls in [("mpi", "-"), ("hybrid", "--")]:
            sub2 = sub[sub["mode"] == mode]
            if sub2.empty:
                continue
            ax.plot(sub2["total_cores"], sub2["time_ms"],
                    marker=MARKERS[i % len(MARKERS)],
                    color=COLORS[i % len(COLORS)],
                    linestyle=ls, markersize=5,
                    label=f"{gs} {mode}")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_xlabel("Total cores")
    ax.set_ylabel("Runtime (ms)")
    ax.set_title("Runtime vs. Total Cores")
    ax.legend(fontsize=7, framealpha=0.7, ncol=2)
    ax.grid(True, alpha=0.3, which="both")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Plot Hybrid BFS speedup curves")
    parser.add_argument("--csv", type=str,
                        default="results/timing.csv",
                        help="Path to timing CSV")
    parser.add_argument("-o", "--output", type=str,
                        default="results/speedup.png",
                        help="Output figure path")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        # Try project-relative
        alt = Path(__file__).resolve().parent.parent / args.csv
        if alt.exists():
            csv_path = alt
        else:
            print(f"ERROR: {args.csv} not found", file=sys.stderr)
            sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} runs from {csv_path}")

    # Basic validation
    required_cols = {"graph_size", "mode", "np", "threads", "time_ms"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"ERROR: CSV missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    # Ensure total_cores column
    if "total_cores" not in df.columns:
        df["total_cores"] = df["np"] * df["threads"]

    # Compute speedup
    df = compute_speedup(df)

    # Build figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Hybrid Parallel BFS — Scaling Analysis", fontsize=13,
                 fontweight="bold", y=0.98)

    plot_pure_mpi_speedup(axes[0, 0], df)
    plot_hybrid_speedup(axes[0, 1], df)
    plot_mpi_vs_hybrid(axes[1, 0], df)
    plot_runtime(axes[1, 1], df)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Figure saved to {out_path}")


if __name__ == "__main__":
    main()
