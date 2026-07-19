#ifndef BFS_MPI_HPP
#define BFS_MPI_HPP

#include "common.hpp"
#include <mpi.h>

/// Run level-synchronous BFS using pure MPI.
///
/// @param g      Local CSR graph (with local_row_ptr, local_col_idx populated)
/// @param root   Global root vertex ID
/// @param comm   MPI communicator
/// @param stats  [out] BFS statistics (timing, levels, visited count)
/// @return       parent array (size local_V), parent[v] = global parent vertex ID,
///               or -1 if unreachable / root. Root's parent is set to -2.
std::vector<int> bfs_mpi(const CsrGraph& g, int root, MPI_Comm comm,
                         BfsStats& stats);

#endif // BFS_MPI_HPP
