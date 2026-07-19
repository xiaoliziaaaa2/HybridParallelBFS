#include "graph_io.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>

void save_csr_binary(const std::string& filename, const CsrGraph& g) {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) throw std::runtime_error("Cannot open " + filename + " for writing");

    ofs.write(reinterpret_cast<const char*>(&g.V), sizeof(g.V));
    ofs.write(reinterpret_cast<const char*>(&g.E), sizeof(g.E));
    ofs.write(reinterpret_cast<const char*>(g.row_ptr.data()),
              (g.V + 1) * sizeof(int64_t));
    ofs.write(reinterpret_cast<const char*>(g.col_idx.data()),
              g.E * sizeof(int));
}

CsrGraph load_csr_binary(const std::string& filename) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) throw std::runtime_error("Cannot open " + filename + " for reading");

    CsrGraph g;
    ifs.read(reinterpret_cast<char*>(&g.V), sizeof(g.V));
    ifs.read(reinterpret_cast<char*>(&g.E), sizeof(g.E));

    g.row_ptr.resize(g.V + 1);
    g.col_idx.resize(g.E);

    ifs.read(reinterpret_cast<char*>(g.row_ptr.data()),
             (g.V + 1) * sizeof(int64_t));
    ifs.read(reinterpret_cast<char*>(g.col_idx.data()),
             g.E * sizeof(int));

    return g;
}

void build_local_csr(CsrGraph& g, int rank, int P) {
    g.offset   = g.rank_offset(rank, P);
    g.local_V  = g.rank_count(rank, P);

    int64_t local_start_edge = g.row_ptr[g.offset];
    int64_t local_end_edge   = g.row_ptr[g.offset + g.local_V];
    int64_t local_E          = local_end_edge - local_start_edge;

    // Build local row_ptr (shifted so offset 0 = first local vertex)
    g.local_row_ptr.resize(g.local_V + 1);
    for (int64_t i = 0; i <= g.local_V; i++) {
        g.local_row_ptr[i] = g.row_ptr[g.offset + i] - local_start_edge;
    }

    // Copy local edges
    g.local_col_idx.resize(local_E);
    std::copy(g.col_idx.begin() + local_start_edge,
              g.col_idx.begin() + local_end_edge,
              g.local_col_idx.begin());

    // Free the full CSR data to save memory
    g.row_ptr.clear();
    g.row_ptr.shrink_to_fit();
    g.col_idx.clear();
    g.col_idx.shrink_to_fit();
}
