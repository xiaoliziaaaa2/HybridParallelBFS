/**
 * bfs_mpi.cpp
 *
 * Pure MPI Level-Synchronous BFS with 1-D vertex partitioning.
 *
 * Communication pattern:
 *   Each level: local expand → MPI_Alltoallv remote discoveries → local update
 */
#include "bfs_mpi.hpp"
#include <algorithm>
#include <cstring>

std::vector<int> bfs_mpi(const CsrGraph& g, int root, MPI_Comm comm,
                         BfsStats& stats) {
    int rank, P;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &P);

    const int64_t local_V = g.local_V;
    const int64_t offset  = g.offset;

    // ---- per-rank data structures ----
    std::vector<int> visited(local_V, 0);   // 0=unvisited, 1=visited
    std::vector<int> parent(local_V, -1);   // global parent vertex ID

    std::vector<int> frontier;              // current level (local vertex indices)
    std::vector<int> next_frontier;         // next level (local vertex indices)

    // ---- send/receive buffers for Alltoallv ----
    // We allocate once and reuse across levels.
    std::vector<std::vector<int>> send_pack(P);   // temp per-rank packing
    std::vector<int> send_counts(P, 0);
    std::vector<int> send_displs(P, 0);
    std::vector<int> recv_counts(P, 0);
    std::vector<int> recv_displs(P, 0);
    std::vector<int> send_buf;
    std::vector<int> recv_buf;

    // ---- initialise root ----
    int64_t global_frontier_size = 0;

    if (g.owner_rank(root, P) == rank) {
        int64_t local_root = g.global_to_local(root, P);
        visited[local_root] = 1;
        parent[local_root] = -2;  // mark as root
        frontier.push_back(static_cast<int>(local_root));
        global_frontier_size = 1;
    }

    // Make sure all ranks agree on initial frontier size
    {
        int64_t local_sz = static_cast<int64_t>(frontier.size());
        MPI_Allreduce(&local_sz, &global_frontier_size, 1, MPI_INT64_T, MPI_SUM,
                      comm);
    }

    // pre-compute per_rank for owner calculations
    int64_t per_rank = (g.V + P - 1) / P;

    Timer timer;
    int level = 0;
    int64_t total_visited_local = frontier.size();

    // ---- BFS loop ----
    while (global_frontier_size > 0) {
        next_frontier.clear();

        // Reset per-rank send buffers
        for (int r = 0; r < P; r++) {
            send_pack[r].clear();
        }

        // ============================================================
        // Phase 1: Local expansion
        // ============================================================
        for (int loc_u : frontier) {
            int64_t u_global = g.local_to_global(loc_u);
            int64_t row_start = g.local_row_ptr[loc_u];
            int64_t row_end   = g.local_row_ptr[loc_u + 1];

            for (int64_t j = row_start; j < row_end; j++) {
                int v_global = g.local_col_idx[j];

                // Determine owner rank of v
                int owner = v_global / static_cast<int>(per_rank);
                if (owner >= P) owner = P - 1;

                if (owner == rank) {
                    // Local vertex
                    int64_t loc_v = v_global - g.rank_offset(rank, P);
                    if (!visited[loc_v]) {
                        visited[loc_v] = 1;
                        parent[loc_v] = static_cast<int>(u_global);
                        next_frontier.push_back(static_cast<int>(loc_v));
                    }
                } else {
                    // Remote vertex — queue for send
                    send_pack[owner].push_back(v_global);
                }
            }
        }

        // ============================================================
        // Phase 2: Alltoallv to exchange remote discoveries
        // ============================================================

        // Build send buffer
        int total_send = 0;
        for (int r = 0; r < P; r++) {
            send_counts[r] = static_cast<int>(send_pack[r].size());
            total_send += send_counts[r];
        }
        send_displs[0] = 0;
        for (int r = 1; r < P; r++) {
            send_displs[r] = send_displs[r - 1] + send_counts[r - 1];
        }

        send_buf.resize(total_send);
        for (int r = 0; r < P; r++) {
            std::copy(send_pack[r].begin(), send_pack[r].end(),
                      send_buf.begin() + send_displs[r]);
        }

        // Exchange counts
        MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                     recv_counts.data(), 1, MPI_INT, comm);

        // Compute receive displacements
        int total_recv = 0;
        recv_displs[0] = 0;
        for (int r = 0; r < P; r++) {
            total_recv += recv_counts[r];
            if (r > 0) recv_displs[r] = recv_displs[r - 1] + recv_counts[r - 1];
        }

        recv_buf.resize(total_recv);

        // Alltoallv — exchange the actual vertex IDs
        MPI_Alltoallv(send_buf.data(), send_counts.data(), send_displs.data(),
                      MPI_INT, recv_buf.data(), recv_counts.data(),
                      recv_displs.data(), MPI_INT, comm);

        // ============================================================
        // Phase 3: Process received vertices
        // ============================================================
        for (int i = 0; i < total_recv; i++) {
            int v_global = recv_buf[i];
            int64_t loc_v = v_global - g.rank_offset(rank, P);
            // loc_v should be in [0, local_V)
            if (loc_v >= 0 && loc_v < local_V && !visited[loc_v]) {
                visited[loc_v] = 1;
                // parent unknown (discovered by remote); mark with special value
                parent[loc_v] = -3;
                next_frontier.push_back(static_cast<int>(loc_v));
            }
        }

        // ============================================================
        // Phase 4: Barrier & prepare next iteration
        // ============================================================
        frontier.swap(next_frontier);

        int64_t local_fsize = static_cast<int64_t>(frontier.size());
        MPI_Allreduce(&local_fsize, &global_frontier_size, 1, MPI_INT64_T,
                      MPI_SUM, comm);

        total_visited_local += local_fsize;
        stats.frontier_sizes.push_back(local_fsize);
        level++;
    }

    stats.time_ms       = timer.elapsed_ms();
    stats.levels        = level;
    stats.visited_count = total_visited_local;

    // Sum visited_count globally so all ranks see the total
    int64_t global_visited = 0;
    MPI_Reduce(&total_visited_local, &global_visited, 1, MPI_INT64_T, MPI_SUM,
               0, comm);
    MPI_Bcast(&global_visited, 1, MPI_INT64_T, 0, comm);
    stats.visited_count = global_visited;

    return parent;
}
