/**
 * graph_generator.cpp
 *
 * Generates Erdős–Rényi random directed graphs and saves them in CSR binary format.
 *
 * Usage:
 *   graph_gen.exe <V> <avg_degree> <output_file> [seed]
 *
 * Example:
 *   graph_gen.exe 20000 16 ../data/graph_20k.csr
 *   graph_gen.exe 40000 16 ../data/graph_40k.csr 42
 */
#include "graph_io.hpp"
#include <iostream>
#include <random>
#include <string>
#include <cstdlib>
#include <algorithm>

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <V> <avg_degree> <output_file> [seed]\n";
        return 1;
    }

    int64_t V         = std::stoll(argv[1]);
    int     avg_degree = std::stoi(argv[2]);
    std::string outfile = argv[3];
    uint64_t seed     = (argc >= 5) ? std::stoull(argv[4]) : 42ULL;

    std::cout << "Generating Erdos-Renyi graph: V=" << V
              << ", avg_degree=" << avg_degree << ", seed=" << seed << "\n";

    // Edge probability: p = avg_degree / (V - 1)
    double p = static_cast<double>(avg_degree) / (V - 1);

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // Phase 1: Count edges per vertex
    std::vector<int64_t> degree(V, 0);
    std::vector<std::vector<int>> adj(V);  // temporary adjacency lists

    int64_t total_edges = 0;
    for (int64_t u = 0; u < V; u++) {
        for (int64_t v = 0; v < V; v++) {
            if (u == v) continue;  // no self-loops
            if (dist(rng) < p) {
                adj[u].push_back(static_cast<int>(v));
                total_edges++;
            }
        }
        if ((u + 1) % 10000 == 0) {
            std::cout << "  Generated " << (u + 1) << " / " << V
                      << " vertices, " << total_edges << " edges so far\n";
        }
    }

    std::cout << "Total edges: " << total_edges << "\n";

    // Phase 2: Build CSR
    CsrGraph g;
    g.V = V;
    g.E = total_edges;
    g.row_ptr.resize(V + 1);
    g.col_idx.resize(total_edges);

    int64_t pos = 0;
    for (int64_t u = 0; u < V; u++) {
        g.row_ptr[u] = pos;
        for (int v : adj[u]) {
            g.col_idx[pos++] = v;
        }
    }
    g.row_ptr[V] = total_edges;

    // Phase 3: Save to file
    save_csr_binary(outfile, g);
    std::cout << "Graph saved to " << outfile << "\n";

    return 0;
}
