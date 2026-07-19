#!/usr/bin/env python3
"""
run_experiments.py

Automated experiment runner for Hybrid Parallel BFS.

Runs pure MPI and Hybrid (MPI+OpenMP) BFS on multiple graph sizes with
multiple configurations.  Each configuration is repeated NRUNS times.
Results are written to results/timing.csv.

Usage:
    python scripts/run_experiments.py [--nruns 5] [--mpiexec mpiexec]
"""

import subprocess
import sys
import os
import csv
import re
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BFS_EXE      = PROJECT_ROOT / "bin" / "bfs.exe"
DATA_DIR     = PROJECT_ROOT / "data"
RESULTS_DIR  = PROJECT_ROOT / "results"

# Experiment configuration
GRAPH_SIZES = {
    "20k":  20000,
    "40k":  40000,
    "80k":  80000,
    "160k": 160000,
}

AVG_DEGREE = 16

MPI_NP_CONFIGS = [1, 2, 4, 8]                   # Pure MPI
HYBRID_CONFIGS = [                                # (np, threads)
    (1, 2), (1, 4),
    (2, 2), (2, 4),
    (4, 2),
]

NRUNS = 5


def run_bfs(mpiexec, mode, graph_path, np, threads=1, root=0):
    """Run BFS once and return parsed result dict, or None on failure."""
    cmd = [
        mpiexec, "-np", str(np),
        str(BFS_EXE),
        "--mode", mode,
        "--graph", str(graph_path),
        "--root", str(root),
        "--csv",
    ]
    if mode == "hybrid":
        cmd += ["--threads", str(threads)]

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600
        )
        # Look for RESULT: line in stdout or stderr
        combined = proc.stdout + proc.stderr
        for line in combined.splitlines():
            m = re.match(
                r"RESULT:(\w+),(\d+),(\d+),(\d+),(\d+),([\d.]+),(\d+),(\d+)",
                line
            )
            if m:
                return {
                    "mode":         m.group(1),
                    "np":           int(m.group(2)),
                    "threads":      int(m.group(3)),
                    "V":            int(m.group(4)),
                    "E":            int(m.group(5)),
                    "time_ms":      float(m.group(6)),
                    "levels":       int(m.group(7)),
                    "visited":      int(m.group(8)),
                }
        print(f"  WARNING: no RESULT line in output", file=sys.stderr)
        return None
    except subprocess.TimeoutExpired:
        print(f"  WARNING: timeout", file=sys.stderr)
        return None
    except FileNotFoundError:
        print(f"  ERROR: {mpiexec} not found", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Hybrid BFS Experiment Runner")
    parser.add_argument("--nruns", type=int, default=NRUNS,
                        help=f"Number of runs per config (default: {NRUNS})")
    parser.add_argument("--mpiexec", type=str, default="mpiexec",
                        help="MPI launcher (default: mpiexec)")
    parser.add_argument("--graph-sizes", type=str, nargs="+",
                        default=list(GRAPH_SIZES.keys()),
                        help="Graph sizes to test (default: all)")
    args = parser.parse_args()

    if not BFS_EXE.exists():
        print(f"ERROR: {BFS_EXE} not found. Build first with: make all")
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "timing.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "graph_size", "mode", "np", "threads", "total_cores",
            "run_id", "time_ms", "levels", "visited"
        ])

    total_runs = 0

    for gs_label in args.graph_sizes:
        if gs_label not in GRAPH_SIZES:
            print(f"WARNING: unknown graph size '{gs_label}', skipping")
            continue

        V = GRAPH_SIZES[gs_label]
        graph_path = DATA_DIR / f"graph_{gs_label}.csr"

        if not graph_path.exists():
            print(f"WARNING: {graph_path} not found, skipping {gs_label}")
            continue

        print(f"\n{'='*50}")
        print(f"  Graph: {gs_label} ({V} vertices)")
        print(f"{'='*50}")

        # ---- Pure MPI ----
        for np in MPI_NP_CONFIGS:
            print(f"  Pure MPI: np={np}")
            for run_id in range(1, args.nruns + 1):
                result = run_bfs(args.mpiexec, "mpi", graph_path, np)
                if result:
                    with open(csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            gs_label, result["mode"], result["np"],
                            result["threads"], result["np"] * result["threads"],
                            run_id, result["time_ms"],
                            result["levels"], result["visited"]
                        ])
                    total_runs += 1
                    print(f"    run {run_id}: {result['time_ms']:.2f} ms")

        # ---- Hybrid ----
        for np, threads in HYBRID_CONFIGS:
            print(f"  Hybrid: np={np} x threads={threads}")
            for run_id in range(1, args.nruns + 1):
                result = run_bfs(args.mpiexec, "hybrid", graph_path,
                                 np, threads)
                if result:
                    with open(csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            gs_label, result["mode"], result["np"],
                            result["threads"], result["np"] * result["threads"],
                            run_id, result["time_ms"],
                            result["levels"], result["visited"]
                        ])
                    total_runs += 1
                    print(f"    run {run_id}: {result['time_ms']:.2f} ms")

    print(f"\n{'='*50}")
    print(f"  Done! {total_runs} runs saved to {csv_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
