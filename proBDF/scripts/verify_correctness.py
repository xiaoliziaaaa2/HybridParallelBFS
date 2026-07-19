#!/usr/bin/env python3
"""
verify_correctness.py

Quick correctness check: generates a small graph, runs pure MPI and hybrid BFS,
and compares the visited vertex counts.

Usage:
    python scripts/verify_correctness.py
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRAPH_GEN   = PROJECT_ROOT / "bin" / "graph_gen.exe"
BFS_EXE     = PROJECT_ROOT / "bin" / "bfs.exe"
TEST_GRAPH  = PROJECT_ROOT / "data" / "test_verify.csr"

V    = 1000
DEG  = 8
ROOT = 0

def run(cmd, timeout=120):
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return proc.stdout + proc.stderr

def main():
    # 1. Generate test graph
    if not GRAPH_GEN.exists():
        print(f"ERROR: {GRAPH_GEN} not found. Build with: make all")
        sys.exit(1)
    if not BFS_EXE.exists():
        print(f"ERROR: {BFS_EXE} not found. Build with: make all")
        sys.exit(1)

    print(f"Generating test graph (V={V}, deg={DEG})...")
    out = run([str(GRAPH_GEN), str(V), str(DEG), str(TEST_GRAPH), "12345"])
    print(out)

    # 2. Run pure MPI BFS (2 ranks)
    print("Running Pure MPI BFS (np=2)...")
    out = run(["mpiexec", "-np", "2", str(BFS_EXE),
               "--mode", "mpi", "--graph", str(TEST_GRAPH),
               "--root", str(ROOT), "--csv"])
    visited_mpi = None
    for line in out.splitlines():
        if line.startswith("RESULT:"):
            parts = line.split(",")
            visited_mpi = int(parts[-1])
            print(f"  Pure MPI visited: {visited_mpi}, time: {parts[5]} ms")
            break

    # 3. Run Hybrid BFS (1 rank × 2 threads)
    print("Running Hybrid BFS (np=1, threads=2)...")
    out = run(["mpiexec", "-np", "1", str(BFS_EXE),
               "--mode", "hybrid", "--graph", str(TEST_GRAPH),
               "--root", str(ROOT), "--threads", "2", "--csv"])
    visited_hyb = None
    for line in out.splitlines():
        if line.startswith("RESULT:"):
            parts = line.split(",")
            visited_hyb = int(parts[-1])
            print(f"  Hybrid visited: {visited_hyb}, time: {parts[5]} ms")
            break

    # 4. Also run serial (np=1 pure MPI) as reference
    print("Running Pure MPI BFS (np=1, serial reference)...")
    out = run(["mpiexec", "-np", "1", str(BFS_EXE),
               "--mode", "mpi", "--graph", str(TEST_GRAPH),
               "--root", str(ROOT), "--csv"])
    visited_serial = None
    for line in out.splitlines():
        if line.startswith("RESULT:"):
            parts = line.split(",")
            visited_serial = int(parts[-1])
            print(f"  Serial visited: {visited_serial}, time: {parts[5]} ms")
            break

    # 5. Compare
    print("\n--- Verification ---")
    all_ok = True
    if visited_mpi is not None and visited_serial is not None:
        if visited_mpi == visited_serial:
            print(f"  [PASS] Pure MPI (np=2) visited = serial: {visited_serial}")
        else:
            print(f"  [FAIL] Pure MPI (np=2) visited {visited_mpi} != serial {visited_serial}")
            all_ok = False
    if visited_hyb is not None and visited_serial is not None:
        if visited_hyb == visited_serial:
            print(f"  [PASS] Hybrid visited = serial: {visited_serial}")
        else:
            print(f"  [FAIL] Hybrid visited {visited_hyb} != serial {visited_serial}")
            all_ok = False

    if all_ok:
        print("\n  All checks passed!")
    else:
        print("\n  Some checks FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
