#ifndef BFS_HYBRID_HPP
#define BFS_HYBRID_HPP

#include "common.hpp"
#include <mpi.h>

/// Run level-synchronous BFS using Hybrid MPI + OpenMP.
///
/// Each MPI rank spawns `num_threads` OpenMP threads to parallelise the local
/// neighbour expansion.  Communication across MPI ranks uses Alltoallv,
/// identical to the pure-MPI variant.
///
/// @param g           Local CSR graph
/// @param root        Global root vertex ID
/// @param comm        MPI communicator
/// @param num_threads Number of OpenMP threads per MPI process
/// @param stats       [out] BFS statistics
/// @return            parent array (size local_V)
std::vector<int> bfs_hybrid(const CsrGraph& g, int root, MPI_Comm comm,
                            int num_threads, BfsStats& stats);

#endif // BFS_HYBRID_HPP
