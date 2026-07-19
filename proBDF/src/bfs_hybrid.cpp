/**
 * bfs_hybrid.cpp
 *
 * Hybrid MPI + OpenMP Level-Synchronous BFS with 1-D vertex partitioning.
 *
 * Local expansion is parallelised with OpenMP threads.  Each thread accumulates
 * its own discovery lists, which are merged after the parallel region.
 * Inter-rank communication still uses MPI_Alltoallv, identical to pure MPI.
 */
#include "bfs_hybrid.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <cstring>
#include <cstdlib>

std::vector<int> bfs_hybrid(const CsrGraph& g, int root, MPI_Comm comm,
                            int num_threads, BfsStats& stats) {
    int rank, P;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &P);

    const int64_t local_V = g.local_V;
    const int64_t offset  = g.offset;
    int64_t per_rank      = (g.V + P - 1) / P;

    // ---- shared data ----
    std::vector<int> visited(local_V, 0);
    std::vector<int> parent(local_V, -1);

    // frontier vectors — use std::vector<int> for each level
    std::vector<int> frontier;
    std::vector<int> next_frontier_merged;

    // ---- send / receive buffers (shared across threads) ----
    std::vector<int> send_counts(P, 0);
    std::vector<int> send_displs(P, 0);
    std::vector<int> recv_counts(P, 0);
    std::vector<int> recv_displs(P, 0);
    std::vector<int> send_buf;
    std::vector<int> recv_buf;

    // ---- thread-local buffers (allocated once, reused each level) ----
    int max_threads = num_threads;
#ifndef _OPENMP
    max_threads = 1;
#endif

    // Per-thread: list of (local_vertex) and per-rank send lists
    std::vector<std::vector<int>> thr_next(max_threads);
    std::vector<std::vector<std::vector<int>>> thr_send(max_threads);
    for (int t = 0; t < max_threads; t++) {
        thr_send[t].resize(P);
    }

    // ---- initialise root ----
    int64_t global_frontier_size = 0;

    if (g.owner_rank(root, P) == rank) {
        int64_t local_root = g.global_to_local(root, P);
        visited[local_root] = 1;
        parent[local_root] = -2;
        frontier.push_back(static_cast<int>(local_root));
    }

    {
        int64_t local_sz = static_cast<int64_t>(frontier.size());
        MPI_Allreduce(&local_sz, &global_frontier_size, 1, MPI_INT64_T, MPI_SUM,
                      comm);
    }

    Timer timer;
    int level = 0;
    int64_t total_visited_local = frontier.size();

    // ================================================================
    // BFS main loop
    // ================================================================
    while (global_frontier_size > 0) {
        next_frontier_merged.clear();

        // Clear thread-local buffers
        for (int t = 0; t < max_threads; t++) {
            thr_next[t].clear();
            for (int r = 0; r < P; r++) {
                thr_send[t][r].clear();
            }
        }

        const int fsize = static_cast<int>(frontier.size());

        // ============================================================
        // Phase 1: Parallel local expansion (OpenMP)
        // ============================================================
#pragma omp parallel num_threads(max_threads) default(none)                  \
    shared(frontier, fsize, g, visited, rank, P, per_rank, thr_next,         \
           thr_send, max_threads)
        {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif

            auto& my_next  = thr_next[tid];
            auto& my_sends = thr_send[tid];

#pragma omp for schedule(dynamic, 64) nowait
            for (int i = 0; i < fsize; i++) {
                int loc_u     = frontier[i];
                int u_global  = static_cast<int>(g.local_to_global(loc_u));
                int64_t rbeg  = g.local_row_ptr[loc_u];
                int64_t rend  = g.local_row_ptr[loc_u + 1];

                for (int64_t j = rbeg; j < rend; j++) {
                    int v_global = g.local_col_idx[j];
                    int owner    = v_global / static_cast<int>(per_rank);
                    if (owner >= P) owner = P - 1;

                    if (owner == rank) {
                        int64_t loc_v = v_global - g.rank_offset(rank, P);
                        my_next.push_back(static_cast<int>(loc_v));
                        my_next.push_back(u_global);  // interleave: vertex, parent
                    } else {
                        my_sends[owner].push_back(v_global);
                    }
                }
            }
        }
        // ---- end of parallel region ----

        // ============================================================
        // Phase 1b: Merge thread-local discoveries (serial, outside omp)
        // ============================================================

        // Merge newly discovered local vertices (deduplicate via visited[])
        for (int t = 0; t < max_threads; t++) {
            const auto& nxt = thr_next[t];
            for (size_t k = 0; k < nxt.size(); k += 2) {
                int loc_v  = nxt[k];
                int parent_global = nxt[k + 1];
                if (!visited[loc_v]) {
                    visited[loc_v] = 1;
                    parent[loc_v] = parent_global;
                    next_frontier_merged.push_back(loc_v);
                }
            }
        }

        // Merge per-rank send buffers
        for (int r = 0; r < P; r++) {
            send_counts[r] = 0;
        }
        for (int t = 0; t < max_threads; t++) {
            for (int r = 0; r < P; r++) {
                send_counts[r] += static_cast<int>(thr_send[t][r].size());
            }
        }

        // ============================================================
        // Phase 2: Alltoallv — exchange remote discoveries
        // ============================================================
        send_displs[0] = 0;
        for (int r = 1; r < P; r++) {
            send_displs[r] = send_displs[r - 1] + send_counts[r - 1];
        }
        int total_send = send_displs[P - 1] + send_counts[P - 1];

        send_buf.resize(total_send);
        // Fill send buffer by copying from thread-local buffers
        std::vector<int> write_pos = send_displs;  // mutable copy
        for (int t = 0; t < max_threads; t++) {
            for (int r = 0; r < P; r++) {
                const auto& src = thr_send[t][r];
                if (!src.empty()) {
                    std::copy(src.begin(), src.end(),
                              send_buf.begin() + write_pos[r]);
                    write_pos[r] += static_cast<int>(src.size());
                }
            }
        }

        // Exchange counts first
        MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                     recv_counts.data(), 1, MPI_INT, comm);

        // Compute receive layout
        int total_recv = 0;
        recv_displs[0] = 0;
        for (int r = 0; r < P; r++) {
            total_recv += recv_counts[r];
            if (r > 0) recv_displs[r] = recv_displs[r - 1] + recv_counts[r - 1];
        }
        recv_buf.resize(total_recv);

        // Exchange vertex data
        MPI_Alltoallv(send_buf.data(), send_counts.data(), send_displs.data(),
                      MPI_INT, recv_buf.data(), recv_counts.data(),
                      recv_displs.data(), MPI_INT, comm);

        // ============================================================
        // Phase 3: Process received vertices
        // ============================================================
        for (int i = 0; i < total_recv; i++) {
            int v_global = recv_buf[i];
            int64_t loc_v = v_global - g.rank_offset(rank, P);
            if (loc_v >= 0 && loc_v < local_V && !visited[loc_v]) {
                visited[loc_v] = 1;
                parent[loc_v] = -3;  // remote discovery
                next_frontier_merged.push_back(static_cast<int>(loc_v));
            }
        }

        // ============================================================
        // Phase 4: Prepare next iteration
        // ============================================================
        frontier.swap(next_frontier_merged);

        int64_t local_fsize = static_cast<int64_t>(frontier.size());
        MPI_Allreduce(&local_fsize, &global_frontier_size, 1, MPI_INT64_T,
                      MPI_SUM, comm);

        total_visited_local += local_fsize;
        stats.frontier_sizes.push_back(local_fsize);
        level++;
    }

    stats.time_ms  = timer.elapsed_ms();
    stats.levels   = level;

    int64_t global_visited = 0;
    MPI_Reduce(&total_visited_local, &global_visited, 1, MPI_INT64_T, MPI_SUM,
               0, comm);
    MPI_Bcast(&global_visited, 1, MPI_INT64_T, 0, comm);
    stats.visited_count = global_visited;

    return parent;
}
