#ifndef GRAPH_IO_HPP
#define GRAPH_IO_HPP

#include "common.hpp"
#include <string>

/// Save a full CSR graph to a binary file.
/// Format: [V:int64][E:int64][row_ptr:V+1 int64][col_idx:E int]
void save_csr_binary(const std::string& filename, const CsrGraph& g);

/// Load a full CSR graph from a binary file.
CsrGraph load_csr_binary(const std::string& filename);

/// Build local CSR from the full CSR for a given rank.
/// After this call, `g.local_row_ptr` and `g.local_col_idx` are populated
/// and `g.offset` / `g.local_V` are set.
void build_local_csr(CsrGraph& g, int rank, int P);

#endif // GRAPH_IO_HPP
