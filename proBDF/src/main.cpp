/**
 * main.cpp — Hybrid Parallel BFS Driver
 *
 * Usage:
 *   mpirun -np <P> ./bfs.exe --mode mpi|hybrid --graph <file.csr>
 *                             [--root 0] [--threads 2]
 *
 * Output (to stdout):
 *   Human-readable statistics, followed by a machine-parseable CSV line
 *   prefixed with "RESULT:".
 */
#include "common.hpp"
#include "graph_io.hpp"
#include "bfs_mpi.hpp"
#include "bfs_hybrid.hpp"

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <iostream>
#include <iomanip>

#ifdef _OPENMP
#include <omp.h>
#endif

static void print_usage(const char* prog) {
    std::cerr << "Usage: mpirun -np <P> " << prog << " [options]\n"
              << "Options:\n"
              << "  --mode    <mpi|hybrid>   BFS mode (required)\n"
              << "  --graph   <file.csr>     Input graph in CSR binary format (required)\n"
              << "  --root    <int>          BFS root vertex (default: 0)\n"
              << "  --threads <int>          OpenMP threads per process (hybrid only, default: 2)\n"
              << "  --csv                     Output only the CSV result line\n";
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    // ---- parse arguments ----
    std::string mode     = "mpi";
    std::string graph_file;
    int  root            = 0;
    int  num_threads     = 2;
    bool csv_only        = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--mode" && i + 1 < argc) {
            mode = argv[++i];
        } else if (arg == "--graph" && i + 1 < argc) {
            graph_file = argv[++i];
        } else if (arg == "--root" && i + 1 < argc) {
            root = std::atoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            num_threads = std::atoi(argv[++i]);
        } else if (arg == "--csv") {
            csv_only = true;
        } else if (arg == "--help" || arg == "-h") {
            if (rank == 0) print_usage(argv[0]);
            MPI_Finalize();
            return 0;
        }
    }

    if (graph_file.empty()) {
        if (rank == 0) {
            std::cerr << "Error: --graph is required\n";
            print_usage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    if (mode != "mpi" && mode != "hybrid") {
        if (rank == 0)
            std::cerr << "Error: --mode must be 'mpi' or 'hybrid'\n";
        MPI_Finalize();
        return 1;
    }

    // ---- load & distribute graph ----
    CsrGraph g;
    if (rank == 0) {
        g = load_csr_binary(graph_file);
    }

    // Broadcast V and E so all ranks can allocate
    MPI_Bcast(&g.V, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&g.E, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);

    // Distribute: rank 0 scatters row_ptr and col_idx
    // Step 1: compute local sizes
    int64_t per_rank = (g.V + P - 1) / P;
    g.offset  = per_rank * rank;
    g.local_V = std::min(per_rank, g.V - g.offset);
    if (g.local_V < 0) g.local_V = 0;

    // Step 2: scatter row_ptr (rank 0 has full row_ptr; others receive their
    //         (local_V+1) entries)
    std::vector<int64_t> full_row_ptr;
    if (rank == 0) {
        full_row_ptr = g.row_ptr;
    }

    // Every rank needs to know: for their local vertices, what are the row_ptr values?
    // We can compute local_row_ptr if we know the start edge and end edge.
    // Strategy: scatter the start/end edge indices, then each rank reads col_idx
    // accordingly.

    // Build counts/displacements for scattering row_ptr segments
    std::vector<int> rcounts(P), rdispls(P);
    std::vector<int> ecounts(P), edispls(P);  // for edges (col_idx)

    if (rank == 0) {
        for (int r = 0; r < P; r++) {
            int64_t r_off  = per_rank * r;
            int64_t r_cnt  = std::min(per_rank, g.V - r_off);
            if (r_cnt < 0) r_cnt = 0;
            rcounts[r] = static_cast<int>(r_cnt + 1);   // +1 for row_ptr
            rdispls[r] = static_cast<int>(r_off);        // displacement in row_ptr

            int64_t e_start = full_row_ptr[r_off];
            int64_t e_end   = (r_off + r_cnt < g.V)
                                  ? full_row_ptr[r_off + r_cnt]
                                  : g.E;
            ecounts[r] = static_cast<int>(e_end - e_start);
            edispls[r] = static_cast<int>(e_start);
        }
    }

    // Scatter local row_ptr sizes first
    int local_rp_size = 0;
    MPI_Scatter(rcounts.data(), 1, MPI_INT, &local_rp_size, 1, MPI_INT, 0,
                MPI_COMM_WORLD);

    // Allocate and scatter row_ptr
    g.local_row_ptr.resize(local_rp_size);
    MPI_Scatterv((rank == 0 ? full_row_ptr.data() : nullptr),
                 rcounts.data(), rdispls.data(),
                 MPI_INT64_T, g.local_row_ptr.data(), local_rp_size, MPI_INT64_T,
                 0, MPI_COMM_WORLD);

    // Shift local_row_ptr so it starts at 0
    {
        int64_t base = g.local_row_ptr[0];
        for (size_t i = 0; i < g.local_row_ptr.size(); i++) {
            g.local_row_ptr[i] -= base;
        }
    }

    // Scatter col_idx
    int local_E_size = 0;
    MPI_Scatter(ecounts.data(), 1, MPI_INT, &local_E_size, 1, MPI_INT, 0,
                MPI_COMM_WORLD);

    g.local_col_idx.resize(local_E_size);

    // Rank 0 still has the full col_idx; non-root sendbuf is unused by Scatterv
    MPI_Scatterv((rank == 0 ? g.col_idx.data() : nullptr),
                 ecounts.data(), edispls.data(), MPI_INT,
                 g.local_col_idx.data(), local_E_size, MPI_INT, 0,
                 MPI_COMM_WORLD);

    // Free full-graph data on rank 0
    if (rank == 0) {
        g.row_ptr.clear();
        g.row_ptr.shrink_to_fit();
        g.col_idx.clear();
        g.col_idx.shrink_to_fit();
    }

    // ---- validate root ----
    if (root < 0 || root >= g.V) {
        if (rank == 0)
            std::cerr << "Error: root " << root << " out of range [0, "
                      << g.V << ")\n";
        MPI_Finalize();
        return 1;
    }

    // ---- run BFS ----
    BfsStats stats;
    std::vector<int> parent;

    // Warm-up (optional, helps get consistent timings)
    // We just run once; the experiment script does multiple runs.

    if (mode == "mpi") {
        parent = bfs_mpi(g, root, MPI_COMM_WORLD, stats);
    } else {
#ifdef _OPENMP
        omp_set_num_threads(num_threads);
#endif
        parent = bfs_hybrid(g, root, MPI_COMM_WORLD, num_threads, stats);
    }

    // ---- output ----
    if (rank == 0) {
        int actual_threads = 1;
#ifdef _OPENMP
        actual_threads = (mode == "hybrid") ? num_threads : 1;
#endif

        if (!csv_only) {
            std::cout << "============================================\n";
            std::cout << "  Hybrid Parallel BFS — Results\n";
            std::cout << "============================================\n";
            std::cout << "Mode:        " << mode << "\n";
            std::cout << "Graph:       " << graph_file << "\n";
            std::cout << "V:           " << g.V << "\n";
            std::cout << "E:           " << g.E << "\n";
            std::cout << "MPI ranks:   " << P << "\n";
            if (mode == "hybrid")
                std::cout << "OMP threads: " << actual_threads << "\n";
            std::cout << "Root:        " << root << "\n";
            std::cout << "BFS time:    " << std::fixed << std::setprecision(3)
                      << stats.time_ms << " ms\n";
            std::cout << "Levels:      " << stats.levels << "\n";
            std::cout << "Visited:     " << stats.visited_count << "\n";
            std::cout << "--------------------------------------------\n";
        }

        // Machine-parseable result line
        // Format: RESULT:mode,np,threads,V,E,time_ms,levels,visited
        std::cout << "RESULT:" << mode << "," << P << "," << actual_threads
                  << "," << g.V << "," << g.E << ","
                  << std::fixed << std::setprecision(3) << stats.time_ms
                  << "," << stats.levels << "," << stats.visited_count << "\n";
    }

    MPI_Finalize();
    return 0;
}
