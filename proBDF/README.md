# Hybrid Parallel BFS

Comparison of **Pure MPI** vs. **Hybrid (MPI + OpenMP)** level-synchronous
Breadth-First Search with **1-D vertex partitioning**.

## Project Structure

```
├── src/
│   ├── common.hpp            Shared types (CsrGraph, BfsStats, Timer)
│   ├── graph_io.{hpp,cpp}    CSR binary read/write & 1-D distribution
│   ├── graph_generator.cpp   Erdős–Rényi random graph generator
│   ├── bfs_mpi.{hpp,cpp}     Pure MPI BFS
│   ├── bfs_hybrid.{hpp,cpp}  Hybrid MPI+OpenMP BFS
│   └── main.cpp              Driver (argument parsing, graph loading, dispatch)
├── scripts/
│   ├── run_experiments.py    Batch experiment runner
│   ├── plot_speedup.py       Speedup & scaling visualisation
│   └── verify_correctness.py Quick correctness smoke test
├── Makefile
├── CMakeLists.txt
└── README.md
```

## Requirements

| Component       | Requirement                              |
|-----------------|------------------------------------------|
| Compiler        | C++17, OpenMP support (GCC 9+ / MSVC 2019+) |
| MPI             | MS-MPI, MPICH, or OpenMPI                |
| Python          | 3.8+ (for scripts)                       |
| Python packages | `pip install pandas matplotlib numpy`    |

## Build

### Option 1: Make (MSYS2 / Linux / WSL)

```bash
make all          # build both executables
make graphs       # generate test graphs (20K–160K vertices)
```

### Option 2: CMake (cross-platform)

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

## Quick Verification

```bash
# Build and generate a small test graph
make all
bin/graph_gen.exe 1000 8 data/test.csr 42

# Verify correctness (serial vs MPI vs Hybrid)
python scripts/verify_correctness.py
```

## Running Experiments

```bash
# 1. Generate graphs
make graphs

# 2. Run all experiments (requires MPI)
python scripts/run_experiments.py --nruns 5

# 3. Generate plots
python scripts/plot_speedup.py
```

Results are written to `results/timing.csv` and `results/speedup.png`.

## Experiment Configuration

| Parameter      | Values                          |
|----------------|---------------------------------|
| Graph sizes    | 20K, 40K, 80K, 160K vertices   |
| Avg. degree    | 16                              |
| Pure MPI       | np = 1, 2, 4, 8                 |
| Hybrid         | (np×th) = 1×2, 1×4, 2×2, 2×4, 4×2 |
| Runs per conf. | 5 (median taken)                |

## Algorithm

**Level-Synchronous BFS** with 1-D vertex partitioning:

1. Each MPI rank owns a contiguous block of vertices
2. Each level:
   - **Local expand**: traverse adjacency lists of current frontier vertices
   - **Communicate**: `MPI_Alltoallv` remote discoveries to owner ranks
   - **Local update**: deduplicate received vertices → next frontier
3. Stop when global frontier is empty

In the **Hybrid** variant, step 2 uses `#pragma omp parallel for`
with thread-local discovery buffers merged after the parallel region.
